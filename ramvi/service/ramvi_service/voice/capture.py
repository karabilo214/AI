"""Owned, bounded microphone worker for continuous sessions (Chat recording unchanged)."""

import base64
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ramvi_service.chat import ChatError


class ProcessMicrophone:
    def __init__(self, worker_factory: Callable[[], Any] | None = None) -> None:
        self.worker_factory = worker_factory

    def devices(self) -> list[Any]:
        from ramvi_service.voice.audio import SoundDeviceMicrophone

        return SoundDeviceMicrophone().devices()

    def open(self, device: str | None, on_block: Callable[[bytes], None]) -> "CaptureStream":
        return CaptureStream(device, on_block, self.worker_factory)


class CaptureStream:
    def __init__(
        self,
        device: str | None,
        on_block: Callable[[bytes], None],
        factory: Callable[[], Any] | None = None,
    ) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        service = Path(__file__).resolve().parents[2]
        paths = [str(service), str(service.parent / "engine"), str(service.parent / "storage")]
        paths.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
        self.worker = (
            factory()
            if factory
            else ProcessSupervisor(
                [
                    sys.executable,
                    "-P",
                    "-c",
                    "import sys,runpy; sys.path[:0]=sys.argv[1:]; sys.argv=['ramvi_service.voice.capture_worker']; runpy.run_module('ramvi_service.voice.capture_worker',run_name='__main__')",
                    *paths,
                ],
                timeout=2,
            )
        )
        self.stopped = threading.Event()
        self.error: str | None = None
        self.minimum_time = 0.0
        self.thread: threading.Thread | None = None
        try:
            self.worker.start()
            result = self.worker.request("start", {"device": device}, timeout=2)
            self.rate = result["rate"]
            if (
                isinstance(self.rate, bool)
                or not isinstance(self.rate, int)
                or not 8000 <= self.rate <= 192000
            ):
                raise ValueError("invalid microphone rate")
            self.thread = threading.Thread(
                target=self._read, args=(on_block,), name="session-microphone", daemon=True
            )
            self.thread.start()
        except Exception:
            self.close()
            raise ChatError("voice.device_error") from None

    def _read(self, on_block: Callable[[bytes], None]) -> None:
        received = time.monotonic()
        try:
            while not self.stopped.is_set():
                result = self.worker.request("read", {}, timeout=1)
                for value, captured in zip(result["blocks"], result["captured_at"], strict=True):
                    pcm = base64.b64decode(value, validate=True)
                    if not pcm or len(pcm) > self.rate // 5 * 2 or len(pcm) % 2:
                        raise ValueError("invalid microphone block")
                    received = time.monotonic()
                    if not self.stopped.is_set() and captured >= self.minimum_time:
                        on_block(pcm)
                if time.monotonic() - received > 3:
                    raise TimeoutError("microphone stalled")
        except Exception:
            if not self.stopped.is_set():
                self.error = "voice.device_error"
        finally:
            self.stopped.set()
            try:
                self.worker.stop(timeout=0.5)
            except Exception:
                self.error = "voice.device_error"

    def flush(self, before: float) -> None:
        # Discard child-buffered echo even if IPC delivery resumes after gate tail.
        self.minimum_time = max(self.minimum_time, before)

    def close(self) -> None:
        self.stopped.set()
        try:
            self.worker.stop(timeout=0.5)
        finally:
            if self.thread and self.thread is not threading.current_thread():
                self.thread.join(timeout=2)
