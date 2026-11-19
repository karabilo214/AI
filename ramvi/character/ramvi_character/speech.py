"""Character output policy. Audio failures preserve text and require explicit recovery."""

import copy
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

SEGMENT_QUEUE = 3
SAFE_ERROR = re.compile(
    r"speech\.(?:disabled|remote_text_disabled|key_missing|secrets_unavailable|voice_not_supported|voice_missing|busy|device_unavailable|device_lost|device_timeout|device_close_failed|provider_timeout|audio_timeout|deadline|provider_failed|cleanup_failed|failed|closed|api_http_[1-5]\d\d)\b"
)


class SpeechOutput:
    def __init__(self, services: Any = None, session_id: str | None = None) -> None:
        self.services = services
        self.session_id = session_id or uuid.uuid4().hex
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None
        self.queue: deque[tuple[int, str, dict[str, Any]]] = deque()
        self.enabled = self.armed = False
        # Response identity is independent of the audio attempt. Device/provider
        # failure or a voice change must not invalidate an in-progress text reply.
        self.response_id: str | None = None
        self.generation = 0
        self.audio_generation = 0
        self.audio_response_id: str | None = None
        self.minimum_generation = 0
        self.offered: dict[str, dict[str, Any]] = {}
        self.last_contact = time.monotonic()
        self.state = "off"
        self.mode = "off"
        self.error: str | None = None
        self.last: dict[str, Any] | None = None
        self.preview_mode = False
        self.high_water = self.backpressure_waits = 0

    def catalog(self, refresh: bool = False) -> dict[str, Any]:
        if self.services is None:
            return {"available": False}
        try:
            return {"available": True, **self.services.call("speech.catalog", {"refresh": refresh})}
        except (NotImplementedError, RuntimeError, OSError):
            return {"available": False}

    def _send_fence(self, generation: int, identifier: str | None) -> None:
        if self.services is None:
            return
        try:
            self.services.call(
                "speech.advance", {"generation": generation, "response_id": identifier}
            )
        except (RuntimeError, PermissionError, OSError):
            with self.lock:
                if generation == self.audio_generation and not self.closed.is_set():
                    raise

    def _audio_fence(
        self,
        identifier: str | None,
        pause: bool = False,
        future_only: bool = False,
        expected_generation: int | None = None,
        error: str | None = None,
    ) -> int:
        with self.changed:
            if expected_generation is not None and expected_generation != self.audio_generation:
                return self.audio_generation
            if error is not None:
                self.error = error
                self.mode, self.state = "text", "error"
            notify = self.services is not None and (self.armed or bool(self.thread))
            self.audio_generation += 1
            generation = self.audio_generation
            self.audio_response_id = identifier
            self.queue.clear()
            self.offered.clear()
            self.preview_mode = False
            self.last = None
            if pause:
                self.armed = False
            if future_only:
                self.minimum_generation = self.generation + 1
            self.changed.notify_all()
        if notify:
            self._send_fence(generation, identifier)
        return generation

    def current(self, response_id: str, generation: int) -> bool:
        with self.lock:
            return self.response_id == response_id and self.generation == generation

    def _audio_current(self, identifier: str, generation: int) -> bool:
        with self.lock:
            return self.audio_response_id == identifier and self.audio_generation == generation

    def _degrade(self, error: BaseException | str, expected_generation: int | None = None) -> None:
        code = SAFE_ERROR.search(str(error))
        try:
            self._audio_fence(
                None,
                pause=True,
                future_only=True,
                expected_generation=expected_generation,
                error=code[0] if code else "speech.provider_failed",
            )
        except (RuntimeError, PermissionError, OSError):
            pass

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        self.pause()
        result = self.services.call("speech.configure", {"settings": settings})
        with self.lock:
            self.enabled = result["settings"]["enabled"]
        self._arm()
        return {**result, **self.view()}

    def _arm(self) -> None:
        with self.lock:
            self.armed = self.enabled
            self.last_contact = time.monotonic()
            self.error = None
            self.mode = "speech" if self.enabled else "off"
            self.state = "idle" if self.enabled else "off"
        self._audio_fence(None, future_only=True)
        self._ensure()

    def arm(self) -> dict[str, Any]:
        catalog = self.catalog()
        with self.lock:
            self.enabled = catalog.get("settings", {}).get("enabled", False)
        self._arm()
        return self.view()

    def pause(self) -> dict[str, Any]:
        self._audio_fence(None, pause=True, future_only=True)
        with self.lock:
            self.mode, self.state = "text" if self.enabled else "off", "off"
        return self.view()

    def preview(self) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        with self.lock:
            self.armed = True
            self.last_contact = time.monotonic()
            self.error = None
            self.mode = "speech"
        identifier = "preview-" + uuid.uuid4().hex
        generation = self._audio_fence(identifier, future_only=True)
        with self.changed:
            self.preview_mode = True
            self.queue.append(
                (
                    generation,
                    identifier,
                    {
                        "id": "preview",
                        "text": "Проверка голоса. Hello!",
                        "emotion": "neutral",
                        "intensity": 0,
                        "voice_style": "neutral",
                    },
                )
            )
            self.changed.notify_all()
        self._ensure()
        return self.view()

    def touch(self) -> None:
        self.last_contact = time.monotonic()

    def _ensure(self) -> None:
        with self.lock:
            if self.armed and not (self.thread and self.thread.is_alive()):
                self.closed = threading.Event()
                self.thread = threading.Thread(
                    target=self._run, name="character-speech", daemon=True
                )
                self.thread.start()

    def begin(self, response_id: str) -> int:
        with self.lock:
            self.generation += 1
            self.response_id = response_id
            generation = self.generation
        self._audio_fence(response_id)
        return generation

    def offer(
        self,
        response_id: str,
        record: dict[str, Any],
        generation: int | None = None,
        on_wait: Callable[[], None] | None = None,
    ) -> bool:
        epoch = self.generation if generation is None else generation
        keepalive = time.monotonic() + 1
        while True:
            with self.changed:
                if (
                    not self.armed
                    or not self.enabled
                    or not self.current(response_id, epoch)
                    or epoch < self.minimum_generation
                ):
                    return False
                previous = self.offered.get(record["id"])
                if previous is not None:
                    if previous != record:
                        raise ValueError("speech.segment_conflict")
                    return True
                if len(self.offered) >= 8:
                    raise ValueError("speech.segment_limit")
                if len(self.queue) < SEGMENT_QUEUE:
                    self.offered[record["id"]] = copy.deepcopy(record)
                    self.queue.append((self.audio_generation, response_id, copy.deepcopy(record)))
                    self.high_water = max(self.high_water, len(self.queue))
                    self.changed.notify_all()
                    return True
                self.backpressure_waits += 1
                self.changed.wait(0.05)
            if on_wait is not None and time.monotonic() >= keepalive:
                on_wait()
                keepalive = time.monotonic() + 1

    def cancel(self, pause: bool = False) -> dict[str, Any]:
        with self.lock:
            self.generation += 1
            self.response_id = None
        self._audio_fence(None, pause)
        return self.view()

    def close(self) -> None:
        self.cancel(pause=True)
        self.closed.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)

    def view(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(
                {
                    "session_id": self.session_id,
                    "enabled": self.enabled,
                    "armed": self.armed,
                    "generation": self.generation,
                    "response_id": self.response_id,
                    "audio_generation": self.audio_generation,
                    "mode": self.mode,
                    "state": self.state,
                    "error": self.error,
                    "last": self.last,
                    "queued": len(self.queue),
                    "queue_capacity": SEGMENT_QUEUE,
                    "high_water": self.high_water,
                    "backpressure_waits": self.backpressure_waits,
                }
            )

    def _run(self) -> None:
        job_id: str | None = None
        job_generation = -1
        failed = False
        try:
            while not self.closed.is_set():
                operation_generation = self.audio_generation
                try:
                    if self.armed and time.monotonic() - self.last_contact > 5:
                        self.pause()
                    if job_id:
                        operation_generation = job_generation
                        packet = self.services.call("speech.status", {"job_id": job_id})["job"]
                        if packet:
                            with self.lock:
                                current = job_generation == self.audio_generation
                                if current:
                                    self.state = packet["state"]
                                    metadata = packet["metadata"]
                                    self.last = {
                                        k: copy.deepcopy(v)
                                        for k, v in metadata.items()
                                        if k != "alignment"
                                    }
                                    self.last["timing_count"] = len(
                                        (metadata.get("alignment") or {}).get("starts", [])
                                    )
                            if current and packet["state"] == "error" and not failed:
                                failed = True
                                self._degrade(
                                    packet.get("error") or "speech.failed", job_generation
                                )
                            if packet.get("settled"):
                                job_id = None
                                with self.lock:
                                    if (
                                        self.preview_mode
                                        and job_generation == self.audio_generation
                                    ):
                                        self.preview_mode = False
                                        self.armed = self.enabled
                                        self.mode = "speech" if self.enabled else "off"
                        else:
                            job_id = None
                    if not job_id:
                        with self.changed:
                            item = self.queue.popleft() if self.armed and self.queue else None
                            self.changed.notify_all()
                            if item is None and self.error is None:
                                self.state = "idle" if self.armed else "off"
                        if item:
                            generation, identifier, record = item
                            operation_generation = generation
                            if not self._audio_current(identifier, generation):
                                continue
                            request = {
                                "text": record["text"],
                                "language": "auto",
                                "emotion": record["emotion"],
                                "intensity": record["intensity"],
                                "style": f"Use a {record['emotion']} tone at relative intensity {record['intensity']:.2f}. Delivery: {record['voice_style']}.",
                            }
                            try:
                                result = self.services.call(
                                    "speech.start",
                                    {
                                        "request": request,
                                        "generation": generation,
                                        "response_id": identifier,
                                        "segment_id": record["id"],
                                        "preview": record["id"] == "preview",
                                    },
                                )
                                job_id, job_generation, failed = result["job_id"], generation, False
                            except (RuntimeError, PermissionError, OSError):
                                if self._audio_current(identifier, generation):
                                    raise
                except Exception as error:
                    self._degrade(error, operation_generation)
                    job_id = None
                with self.changed:
                    self.changed.wait(0.05)
        finally:
            if self.services is not None:
                try:
                    self.services.call("speech.cancel", {})
                except (OSError, RuntimeError, PermissionError):
                    pass
            with self.changed:
                self.armed = False
                self.queue.clear()
                self.changed.notify_all()
