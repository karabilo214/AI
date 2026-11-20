"""Explicit bounded voice session: microphone -> VAD -> partial/final STT events.

Generic audio orchestration only: no Character prompts, LLM calls or automatic sends.
"""

import copy
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from ramvi_service.chat import ChatError
from ramvi_service.contracts.voice import AudioCapture, AudioStream, SpeechRecognizer


class Detector(Protocol):
    def prepare(self) -> None: ...
    def speech(self, pcm: bytes, rate: int) -> bool: ...
    def close(self) -> None: ...


class VoiceSession:
    def __init__(
        self,
        microphone: AudioCapture,
        detector: Detector,
        recognizer: SpeechRecognizer,
        *,
        on_close: Callable[[], None],
        language: str = "auto",
        silence_s: float = 0.8,
        partial_s: float = 3,
        max_utterance_s: float = 30,
        max_session_s: float = 3600,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.microphone, self.detector, self.recognizer = microphone, detector, recognizer
        self.on_close, self.language = on_close, language
        self.silence_s, self.partial_s = silence_s, partial_s
        self.max_utterance_s, self.max_session_s = max_utterance_s, max_session_s
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.audio: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self.jobs: queue.Queue[tuple[str, str, bytes, int, list[int]]] = queue.Queue(maxsize=2)
        self.events: deque[dict[str, Any]] = deque(maxlen=64)
        self.sequence = 0
        self.epoch = 0
        self.playing = False
        self.suppressed_until = 0.0
        self.error: str | None = None
        self.state = "new"
        self.rate = 0
        self.stream: AudioStream | None = None
        self.threads: list[threading.Thread] = []

    def start(self, device: str | None = None) -> None:
        try:
            self.detector.prepare()  # fail before opening the microphone
            self.stream = self.microphone.open(device, self._capture)
            self.rate = self.stream.rate
            if not 8000 <= self.rate <= 192000:
                raise ValueError("unsupported microphone rate")
            self.state = "listening"
            self._emit("session_started")
            self.threads = [
                threading.Thread(target=target, name=name, daemon=True)
                for target, name in [(self._segment, "voice-vad"), (self._recognize, "voice-stt")]
            ]
            for thread in self.threads:
                thread.start()
        except BaseException:
            self.stop()
            raise

    def _emit(self, kind: str, **fields: Any) -> None:
        with self.lock:
            if self.stopped.is_set() and kind not in {"session_stopped", "error"}:
                return
            self.sequence += 1
            self.events.append(
                {"seq": self.sequence, "session_id": self.id, "type": kind, **fields}
            )

    def _capture(self, block: bytes) -> None:
        if self.stopped.is_set() or self.playing or time.monotonic() < self.suppressed_until:
            return
        # PortAudio callback only copies a bounded block, never performs inference.
        if len(block) > 192000 * 2:
            self.error = "voice.session_overflow"
            self.stopped.set()
            return
        try:
            self.audio.put_nowait(block)
        except queue.Full:
            self.error = "voice.session_overflow"
            self.stopped.set()

    def playback(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise ValueError("playback active must be boolean")
        with self.lock:
            if active != self.playing:
                self.epoch += 1  # discard any queued transcription across the playback boundary
            self.playing = active
            self.suppressed_until = time.monotonic() + (0.5 if not active else 0)
            flush = getattr(self.stream, "flush", None)
            if callable(flush):
                flush(self.suppressed_until)
            while not self.audio.empty():
                try:
                    self.audio.get_nowait()
                except queue.Empty:
                    break
        self._emit(
            "capture_suppressed" if active else "capture_resumed", echo_mode="half_duplex_gate"
        )

    def read(self, cursor: int = 0) -> dict[str, Any]:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("invalid voice cursor")
        with self.lock:
            return {
                "session_id": self.id,
                "state": self.state,
                "error": self.error,
                "cursor": self.sequence,
                "gap": bool(self.events and cursor < self.events[0]["seq"] - 1),
                "events": copy.deepcopy([e for e in self.events if e["seq"] > cursor]),
                "echo_mode": "half_duplex_gate",
                "barge_in_during_playback": False,
            }

    def _submit(
        self, kind: str, identifier: str, pcm: bytes, epoch: int, cuts: list[int] | None = None
    ) -> None:
        if kind == "final":
            retained = []
            while True:
                try:
                    old = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if old[0] == "final":
                    retained.append(old)
            for old in retained:
                self.jobs.put_nowait(old)
        elif not self.jobs.empty():
            return  # stale preview is expendable; owner finals are not
        try:
            self.jobs.put_nowait((kind, identifier, pcm, epoch, list(cuts or [])))
        except queue.Full:
            if kind == "final":
                raise ChatError("voice.session_overflow") from None

    def _segment(self) -> None:
        pre = bytearray()
        utterance = bytearray()
        window = bytearray()
        identifier = ""
        cuts: list[int] = []
        quiet = total = next_partial = 0.0
        epoch = self.epoch
        started = time.monotonic()
        try:
            while not self.stopped.is_set():
                if time.monotonic() - started >= self.max_session_s:
                    break
                try:
                    block = self.audio.get(timeout=0.1)
                except queue.Empty:
                    if self.stream is not None and getattr(self.stream, "error", None):
                        raise ChatError("voice.device_error") from None
                    continue
                if epoch != self.epoch:
                    epoch = self.epoch
                    cuts.clear()
                    pre.clear()
                    utterance.clear()
                    window.clear()
                    identifier = ""
                if self.playing or time.monotonic() < self.suppressed_until:
                    continue
                duration = len(block) / (self.rate * 2)
                window += block
                del window[: -self.rate * 2]
                speech = self.detector.speech(bytes(window), self.rate)
                if epoch != self.epoch or self.playing or time.monotonic() < self.suppressed_until:
                    continue
                if not identifier:
                    pre += block
                    del pre[
                        : -self.rate * 2
                    ]  # one second of pre-roll, also protects first syllables
                    if not speech:
                        continue
                    identifier = uuid.uuid4().hex
                    utterance = bytearray(pre)
                    pre.clear()
                    cuts.clear()
                    quiet = total = 0
                    next_partial = self.partial_s
                    self._emit("speech_start", utterance_id=identifier)
                else:
                    utterance += block
                total += duration
                was_quiet = quiet > 0
                quiet = 0 if speech else quiet + duration
                last_cut = cuts[-1] if cuts else 0
                # Short pauses split ASR language windows, not the owner's whole turn.
                # This avoids forcing the first detected language on a later language.
                if quiet and not was_quiet and len(utterance) - last_cut >= self.rate * 4:
                    cuts.append(len(utterance) - len(block) // 2)
                elif len(utterance) - last_cut >= self.rate * 12:
                    cuts.append(len(utterance))
                if quiet >= self.silence_s or total >= self.max_utterance_s:
                    self._emit("speech_end", utterance_id=identifier)
                    if cuts and len(utterance) - cuts[-1] < self.rate * 2:
                        cuts.pop()  # avoid a sub-second final language window
                    self._submit("final", identifier, bytes(utterance), epoch, cuts)
                    identifier = ""
                    utterance.clear()
                    window.clear()
                elif total >= next_partial:
                    self._submit("partial", identifier, bytes(utterance), epoch)
                    next_partial += self.partial_s
        except Exception as error:
            if not self.stopped.is_set():
                self.error = getattr(error, "code", "voice.stt_failed")
        finally:
            self.stop()

    def _recognize(self) -> None:
        try:
            while not self.stopped.is_set():
                try:
                    kind, identifier, pcm, epoch, cuts = self.jobs.get(timeout=0.1)
                except queue.Empty:
                    if self.stream is not None and getattr(self.stream, "error", None):
                        raise ChatError("voice.device_error") from None
                    continue
                if epoch != self.epoch:
                    continue
                recognize_at = time.monotonic()
                try:
                    texts: list[str] = []
                    languages: list[str] = []
                    start = 0
                    for end in [*cuts, len(pcm)]:
                        if self.stopped.is_set() or epoch != self.epoch:
                            break
                        result = self.recognizer.transcribe(
                            pcm[start:end], self.rate, self.language
                        )
                        value = result.get("text", "")
                        if not isinstance(value, str):
                            raise ValueError("invalid voice transcript")
                        if value.strip():
                            texts.append(value.strip())
                            language = result.get("language")
                            if isinstance(language, str) and language not in languages:
                                languages.append(language)
                        start = end
                    result = {
                        "text": " ".join(texts),
                        "language": languages[0] if len(languages) == 1 else "mixed",
                    }
                except ChatError as error:
                    if error.code == "voice.no_speech":
                        if kind == "final" and epoch == self.epoch:
                            self._emit("rejected", utterance_id=identifier, code="voice.no_speech")
                        continue
                    raise
                finally:
                    del pcm
                content = result.get("text", "").strip()
                if not isinstance(content, str) or len(content.encode()) > 4096:
                    raise ValueError("invalid voice transcript")
                if not content and kind == "final" and epoch == self.epoch:
                    self._emit("rejected", utterance_id=identifier, code="voice.no_speech")
                if content and not self.stopped.is_set() and epoch == self.epoch:
                    self._emit(
                        kind,
                        utterance_id=identifier,
                        text=content,
                        language=result.get("language"),
                        languages=languages,
                        provisional=kind == "partial",
                        recognition_ms=(time.monotonic() - recognize_at) * 1000,
                    )
        except Exception as error:
            if not self.stopped.is_set():
                self.error = getattr(error, "code", "voice.stt_failed")
            self.stop()

    def stop(self) -> None:
        # The cleanup guard is separate: an overflowing callback sets stopped first.
        with self.lock:
            if self.state == "stopped":
                return
            self.stopped.set()
            self.state = "stopped"
            stream, self.stream = self.stream, None
        cleanup = ([stream.close] if stream else []) + [
            self.on_close,
            self.detector.close,
            self.recognizer.close,
        ]
        for close in cleanup:
            try:
                close()
            except Exception:
                self.error = self.error or "voice.cleanup_failed"
        for channel in (self.audio, self.jobs):
            while not channel.empty():
                try:
                    channel.get_nowait()
                except queue.Empty:
                    break

        if self.error:
            self._emit("error", code=self.error)
        self._emit("session_stopped")

    def join(self) -> None:
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)
