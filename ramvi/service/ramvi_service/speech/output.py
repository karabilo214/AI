"""PCM device facade. All native audio calls are bounded and owned by a worker."""

import base64
import os
import sys
import threading
from pathlib import Path
from typing import Any


class SoundOutput:
    def __init__(self, rate: int, *, worker: Any = None, timeout: float = 2) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        self.rate, self.timeout = rate, timeout
        self.stopped = threading.Event()
        self.opened = False
        service = Path(__file__).resolve().parents[2]
        paths = [str(service), str(service.parent / "engine"), str(service.parent / "storage")]
        paths.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
        self.worker = worker or ProcessSupervisor(
            [
                sys.executable,
                "-P",
                "-c",
                "import sys,runpy; sys.path[:0]=sys.argv[1:]; sys.argv=['ramvi_service.speech.device_worker']; runpy.run_module('ramvi_service.speech.device_worker',run_name='__main__')",
                *paths,
            ],
            timeout=timeout,
        )

    def _request(self, method: str, payload: dict[str, Any], code: str) -> None:
        if self.stopped.is_set():
            raise InterruptedError("audio stopped")
        try:
            self.worker.request(method, payload, timeout=self.timeout)
        except TimeoutError:
            self.interrupt()
            raise TimeoutError("speech.device_timeout") from None
        except Exception:
            self.interrupt()
            raise RuntimeError(code) from None

    def start(self) -> None:
        if self.stopped.is_set():
            raise InterruptedError("audio stopped")
        self.worker.start()
        self.opened = True
        # Cancellation may have run while the interpreter was being spawned.
        if self.stopped.is_set():
            self.worker.stop(timeout=0.5)
            raise InterruptedError("audio stopped")
        self._request("start", {"rate": self.rate}, "speech.device_unavailable")

    def write(self, pcm: bytes) -> None:
        if not pcm or len(pcm) > 8192 or len(pcm) % 2:
            raise ValueError("invalid PCM block")
        self._request("write", {"pcm": base64.b64encode(pcm).decode()}, "speech.device_lost")

    def interrupt(self) -> None:
        # Does not wait for an in-flight RPC lock. Job/process-group termination
        # interrupts open/write/drain as well as ordinary playback.
        self.stopped.set()
        self.worker.stop(timeout=0.5)

    def cancel(self) -> None:
        self.interrupt()

    def close(self) -> None:
        try:
            if self.opened and not self.stopped.is_set():
                self._request("close", {}, "speech.device_close_failed")
        finally:
            self.interrupt()
