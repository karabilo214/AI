"""Character speech policy. Only validated segments enter this queue; providers live in Core."""

import copy
import threading
import time
from collections import deque
from typing import Any


class SpeechOutput:
    def __init__(self, services: Any = None) -> None:
        self.services = services
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None
        self.queue: deque[tuple[str, dict[str, Any]]] = deque()
        self.enabled = False
        self.armed = False
        self.response_id: str | None = None
        self.cancel_pending = False
        self.last_contact = time.monotonic()
        self.state = "off"
        self.error: str | None = None
        self.last: dict[str, Any] | None = None
        self.preview_mode = False

    def catalog(self) -> dict[str, Any]:
        if self.services is None:
            return {"available": False}
        try:
            return {"available": True, **self.services.call("speech.catalog", {})}
        except (NotImplementedError, RuntimeError):
            return {"available": False}

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        result = self.services.call("speech.configure", {"settings": settings})
        with self.lock:
            self.enabled = result["settings"]["enabled"]
            self.armed = self.enabled
            self.last_contact = time.monotonic()
            self.cancel()
            self._ensure()
        return {**result, **self.view()}

    def arm(self) -> dict[str, Any]:
        catalog = self.catalog()
        with self.lock:
            self.enabled = catalog.get("settings", {}).get("enabled", False)
            self.armed = self.enabled
            self.last_contact = time.monotonic()
            self._ensure()
        return self.view()

    def preview(self) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        with self.lock:
            self.cancel()
            self.armed = True
            self.preview_mode = True
            self.response_id = "preview"
            self.last_contact = time.monotonic()
            self.queue.append(
                (
                    "preview",
                    {
                        "id": "preview",
                        "text": "Проверка голоса. Hello!",
                        "emotion": "neutral",
                        "intensity": 0,
                        "voice_style": "neutral",
                    },
                )
            )
            self._ensure()
            self.wake.set()
        return self.view()

    def touch(self) -> None:
        self.last_contact = time.monotonic()

    def _ensure(self) -> None:
        if self.armed and not (self.thread and self.thread.is_alive()):
            self.closed = threading.Event()
            self.thread = threading.Thread(target=self._run, name="character-speech", daemon=True)
            self.thread.start()

    def begin(self, response_id: str) -> None:
        with self.lock:
            self.cancel()
            self.response_id = response_id

    def offer(self, response_id: str, record: dict[str, Any]) -> None:
        with self.lock:
            if not self.armed or not self.enabled or response_id != self.response_id:
                return
            if len(self.queue) >= 8:
                self.error = "speech.queue_full"
                self.cancel()
                return
            self.queue.append((response_id, copy.deepcopy(record)))
            self.wake.set()

    def cancel(self, pause: bool = False) -> dict[str, Any]:
        with self.lock:
            self.queue.clear()
            self.response_id = None
            self.cancel_pending = True
            self.preview_mode = False
            if pause:
                self.armed = False
            self.wake.set()
        return self.view()

    def close(self) -> None:
        self.cancel(pause=True)
        self.closed.set()
        self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)

    def view(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(
                {
                    "enabled": self.enabled,
                    "armed": self.armed,
                    "state": self.state,
                    "error": self.error,
                    "last": self.last,
                    "queued": len(self.queue),
                }
            )

    def _run(self) -> None:
        job_id: str | None = None
        try:
            while not self.closed.is_set():
                if self.armed and time.monotonic() - self.last_contact > 5:
                    self.cancel(pause=True)
                with self.lock:
                    cancelling = self.cancel_pending
                    self.cancel_pending = False
                if cancelling and job_id:
                    self.services.call("speech.cancel", {"job_id": job_id})
                    # Wait for a cancelled native/provider task before starting another one.
                    while not self.closed.is_set():
                        job = self.services.call("speech.status", {"job_id": job_id})["job"]
                        if job is None or job.get("settled"):
                            break
                        self.closed.wait(0.05)
                    job_id = None
                if job_id:
                    packet = self.services.call("speech.status", {"job_id": job_id})["job"]
                    if packet:
                        with self.lock:
                            self.state = packet["state"]
                            metadata = packet["metadata"]
                            self.last = {
                                k: copy.deepcopy(v) for k, v in metadata.items() if k != "alignment"
                            }
                            self.last["timing_count"] = len(
                                (metadata.get("alignment") or {}).get("starts", [])
                            )
                            self.error = packet["error"]
                        if packet.get("settled"):
                            job_id = None
                            if self.preview_mode:
                                self.preview_mode = False
                                self.armed = self.enabled
                if not job_id:
                    with self.lock:
                        item = self.queue.popleft() if self.armed and self.queue else None
                        if item is None:
                            self.state = "idle" if self.armed else "off"
                    if item:
                        identifier, record = item
                        request = {
                            "text": record["text"],
                            "language": "auto",
                            "emotion": record["emotion"],
                            "intensity": record["intensity"],
                            "style": f"Use a {record['emotion']} tone at relative intensity {record['intensity']:.2f}. Delivery: {record['voice_style']}.",
                        }
                        with self.lock:
                            if identifier != self.response_id or not self.armed:
                                continue
                        result = self.services.call(
                            "speech.start",
                            {
                                "request": request,
                                "response_id": identifier,
                                "segment_id": record["id"],
                                "preview": identifier == "preview",
                            },
                        )
                        job_id = result["job_id"]
                self.wake.wait(0.08)
                self.wake.clear()
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
            with self.lock:
                self.armed = False
                self.queue.clear()
                self.state = "off" if self.error is None else "error"
