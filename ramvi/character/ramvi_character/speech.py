"""Generation-fenced Character policy; only validated segments enter a bounded queue."""

import copy
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

SEGMENT_QUEUE = 3


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
        self.response_id: str | None = None
        self.generation = 0
        self.offered: dict[str, dict[str, Any]] = {}
        self.last_contact = time.monotonic()
        self.state = "off"
        self.error: str | None = None
        self.last: dict[str, Any] | None = None
        self.preview_mode = False
        self.high_water = 0
        self.backpressure_waits = 0

    def catalog(self) -> dict[str, Any]:
        if self.services is None:
            return {"available": False}
        try:
            return {"available": True, **self.services.call("speech.catalog", {})}
        except (NotImplementedError, RuntimeError):
            return {"available": False}

    def _fence(self, response_id: str | None, pause: bool = False) -> int:
        with self.changed:
            notify_core = self.services is not None and (self.armed or bool(self.thread))
            self.generation += 1
            generation = self.generation
            self.response_id = response_id
            self.queue.clear()
            self.offered.clear()
            self.preview_mode = False
            self.last = None
            if pause:
                self.armed = False
            self.changed.notify_all()
        if notify_core:
            try:
                self.services.call(
                    "speech.advance", {"generation": generation, "response_id": response_id}
                )
            except (RuntimeError, PermissionError):
                with self.lock:
                    if generation == self.generation and not self.closed.is_set():
                        raise
                # Another fence overtook this request; it already cancelled this one.
        return generation

    def current(self, response_id: str, generation: int) -> bool:
        with self.lock:
            return self.response_id == response_id and self.generation == generation

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        self.cancel(pause=True)
        result = self.services.call("speech.configure", {"settings": settings})
        with self.lock:
            self.enabled = result["settings"]["enabled"]
            self.armed = self.enabled
            self.last_contact = time.monotonic()
            self.error = None
        self._fence(None)
        self._ensure()
        return {**result, **self.view()}

    def arm(self) -> dict[str, Any]:
        catalog = self.catalog()
        with self.lock:
            self.enabled = catalog.get("settings", {}).get("enabled", False)
            self.armed = self.enabled
            self.last_contact = time.monotonic()
        self._fence(None)
        self._ensure()
        return self.view()

    def preview(self) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        with self.lock:
            self.armed = True
            self.last_contact = time.monotonic()
        identifier = "preview-" + uuid.uuid4().hex
        generation = self._fence(identifier)
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
        return self._fence(response_id)

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
                if not self.armed or not self.enabled or not self.current(response_id, epoch):
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
                    self.queue.append((epoch, response_id, copy.deepcopy(record)))
                    self.high_water = max(self.high_water, len(self.queue))
                    self.changed.notify_all()
                    return True
                self.backpressure_waits += 1
                self.changed.wait(0.05)
            # Keep the upstream lease alive without consuming model chunks or
            # holding the queue lock. UI heartbeat still bounds unattended playback.
            if on_wait is not None and time.monotonic() >= keepalive:
                on_wait()
                keepalive = time.monotonic() + 1

    def cancel(self, pause: bool = False) -> dict[str, Any]:
        self._fence(None, pause)
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
        try:
            while not self.closed.is_set():
                if self.armed and time.monotonic() - self.last_contact > 5:
                    self.cancel(pause=True)
                if job_id:
                    packet = self.services.call("speech.status", {"job_id": job_id})["job"]
                    if packet:
                        with self.lock:
                            if job_generation == self.generation:
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
                                self.error = packet["error"]
                            if packet.get("settled"):
                                job_id = None
                                if self.preview_mode and job_generation == self.generation:
                                    self.preview_mode = False
                                    self.armed = self.enabled
                    else:
                        job_id = None
                if not job_id:
                    with self.changed:
                        item = self.queue.popleft() if self.armed and self.queue else None
                        self.changed.notify_all()
                        if item is None:
                            self.state = "idle" if self.armed else "off"
                    if item:
                        generation, identifier, record = item
                        if not self.current(identifier, generation):
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
                            job_id, job_generation = result["job_id"], generation
                        except (RuntimeError, PermissionError):
                            if self.current(identifier, generation):
                                raise
                with self.changed:
                    self.changed.wait(0.04)
        except Exception as error:
            with self.lock:
                code = str(error)
                self.error = (
                    code if code.startswith("speech.") and len(code) < 100 else "speech.failed"
                )
                self.state = "error"
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
                self.state = "off" if self.error is None else "error"
