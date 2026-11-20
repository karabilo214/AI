"""Character-owned voice session policy; audio and recognition stay in scoped services."""

import copy
import threading
import time
from collections import deque
from typing import Any


class VoiceConversation:
    def __init__(self, owner: Any, services: Any = None) -> None:
        self.owner, self.services = owner, services
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = "inactive"
        self.error: str | None = None
        self.capture_id: str | None = None
        self.partial = ""
        self.last_reply: dict[str, Any] | None = None
        self.remote_host: str | None = None
        self.cursor = 0
        self.recognition_ms: float | None = None
        self.last_contact = time.monotonic()

    def capabilities(self) -> dict[str, Any]:
        if self.services is None:
            return {"available": False}
        try:
            caps = self.services.call("voice.capabilities", {})["capabilities"]
            self.remote_host = caps.get("remote_host")
            return {"available": bool(caps.get("session", {}).get("supported")), **caps}
        except (OSError, RuntimeError, NotImplementedError):
            return {"available": False}

    def touch(self) -> None:
        self.last_contact = time.monotonic()

    def view(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(
                {
                    "state": self.state,
                    "recognition_ms": self.recognition_ms,
                    "error": self.error,
                    "partial": self.partial,
                    "reply": self.last_reply,
                    "remote_host": self.remote_host,
                    "echo_mode": "half_duplex_gate",
                    "barge_in_during_playback": False,
                }
            )

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("voice session already active or stopping")
            if self.services is None or not self.owner._model:
                raise RuntimeError("select a model before starting voice")
            # An explicitly opened microphone must not start under an older TTS
            # stream which was opened before this input session existed.
            self.owner._dialogue.cancel()
            self.stop_event = threading.Event()
            self.touch()
            self.state, self.error, self.partial = "starting", None, ""
            self.capture_id, self.last_reply, self.cursor = None, None, 0
            self.thread = threading.Thread(target=self._run, name="character-voice", daemon=True)
            self.thread.start()
            return self.view()

    def end(self, wait: bool = False) -> dict[str, Any]:
        self.stop_event.set()
        with self.lock:
            if self.state not in {"inactive", "error"}:
                self.state = "stopping"
            self.partial = ""
        if wait and self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        return self.view()

    def playback(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise ValueError("playback active must be boolean")
        if self.capture_id and not self.stop_event.is_set():
            self.services.call(
                "voice.session.playback", {"capture_id": self.capture_id, "active": active}
            )

    def _run(self) -> None:
        pending: deque[dict[str, Any]] = deque()
        seen: set[str] = set()
        speaking: set[str] = set()
        latest_utterance: str | None = None
        cursor = 0
        try:
            packet = self.services.call("voice.session.start", {"language": "auto"})
            self.capture_id = packet["session_id"]
            with self.lock:
                self.state = "listening"
            while not self.stop_event.is_set():
                if time.monotonic() - self.last_contact > 5:
                    raise RuntimeError("voice.session_heartbeat_lost")
                packet = self.services.call(
                    "voice.session.read", {"capture_id": self.capture_id, "cursor": cursor}
                )
                if packet.get("gap"):
                    raise RuntimeError("voice.session_overflow")
                cursor = packet["cursor"]
                for event in packet["events"]:
                    if self.stop_event.is_set() or event["session_id"] != self.capture_id:
                        break
                    kind = event["type"]
                    with self.lock:
                        if kind == "speech_start":
                            pending.clear()
                            latest_utterance = event["utterance_id"]
                            speaking.add(latest_utterance)
                            self.owner._dialogue.cancel()
                            self.state, self.partial = "hearing", ""
                        elif kind == "speech_end":
                            speaking.discard(event["utterance_id"])
                            self.state = "hearing" if speaking else "recognizing"
                        elif kind == "partial" and event["utterance_id"] == latest_utterance:
                            self.partial = event["text"]
                        elif (
                            kind == "final"
                            and event["utterance_id"] == latest_utterance
                            and event["utterance_id"] not in seen
                        ):
                            self.recognition_ms = event.get("recognition_ms")
                            seen.add(event["utterance_id"])
                            if len(seen) > 256 or len(pending) >= 2:
                                raise RuntimeError("voice.session_overflow")
                            self.owner._dialogue.cancel()
                            pending.append(event)
                            self.partial = ""
                            self.state = "hearing" if speaking else "listening"
                        elif kind == "rejected" and event["utterance_id"] == latest_utterance:
                            self.partial = ""
                            self.state = "hearing" if speaking else "listening"
                        elif kind == "error":
                            raise RuntimeError(event["code"])
                if packet["state"] == "stopped":
                    break
                with self.lock:
                    thread = self.owner._dialogue.thread
                    if (
                        pending
                        and not speaking
                        and not (thread and thread.is_alive())
                        and not self.stop_event.is_set()
                    ):
                        utterance = pending.popleft()
                        result = self.owner._dialogue.start(utterance["text"], self.owner._model)
                        self.last_reply = {
                            **result,
                            "text": utterance["text"],
                            "utterance_id": utterance["utterance_id"],
                        }
                self.stop_event.wait(0.1)
        except Exception as error:
            if not self.stop_event.is_set():
                code = str(error)
                with self.lock:
                    self.error = (
                        code
                        if code.startswith("voice.") and len(code) < 100
                        else "character.voice_error"
                    )
        finally:
            self.stop_event.set()
            if self.error:
                try:
                    self.owner._speech.pause()
                except (OSError, RuntimeError, PermissionError, ValueError):
                    pass
            if self.services is not None:
                try:
                    # Startup can time out after the service opened our microphone.
                    # Scope ownership still permits cancelling only our own session.
                    self.services.call("voice.session.cancel", {"capture_id": self.capture_id})
                except (OSError, RuntimeError, PermissionError, ValueError):
                    pass
            pending.clear()
            with self.lock:
                self.partial = ""
                self.state = "error" if self.error else "inactive"
