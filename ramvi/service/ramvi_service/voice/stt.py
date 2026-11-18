"""Клиент процесса распознавания (`voice/worker.py`) поверх общего `ProcessSupervisor`."""

import os
import sys
import threading
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Protocol

from ramvi_service.chat import ChatError
from ramvi_service.voice.models import VAD


class Transcriber(Protocol):
    def prepare(self) -> None: ...

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]: ...

    def close(self) -> None: ...

    @property
    def running(self) -> bool: ...


def default_threads() -> int:
    """Половина логических ядер, не больше 8: вторая половина — интерфейсу и модели."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


class WhisperProcess:
    """Процесс с моделью: запускается при первой записи, держится до `close()`."""

    def __init__(
        self,
        model: Path,
        threads: int | None = None,
        command: list[str] | None = None,
        timeout: float = 600.0,
    ) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        self.model = Path(model)
        service = Path(__file__).resolve().parents[2]
        paths = [str(service), str(service.parent / "engine")]
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        env = {"PYTHONPATH": os.pathsep.join(paths), "PYTHONUTF8": "1"}
        argv = command or [sys.executable, "-P", "-m", "ramvi_service.voice.worker"]
        threads = threads or default_threads()
        vad = self.model.parent / VAD.file  # детектор речи лежит рядом с моделью
        extra = ["--vad", str(vad)] if vad.is_file() else []
        self.process = ProcessSupervisor(
            [*argv, "--model", str(self.model), "--threads", str(threads), *extra],
            env=env,
            timeout=timeout,
        )
        self.timeout = timeout
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return bool(self.process.status()["state"] == "running")

    def prepare(self) -> None:
        """Запустить процесс и загрузить модель (пока человек говорит)."""
        with self._lock:
            if not self.running:
                self.process.start()

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
        self.prepare()
        block = shared_memory.SharedMemory(
            name=f"ramvi-stt-{uuid.uuid4().hex[:12]}", create=True, size=max(1, len(pcm))
        )
        buf = block.buf
        assert buf is not None
        try:
            buf[: len(pcm)] = pcm
            payload = {"shm": block.name, "size": len(pcm), "rate": rate, "language": language}
            return self.process.request("transcribe", payload, timeout=self.timeout)
        except (RuntimeError, TimeoutError) as error:
            log = " | ".join(self.process.diagnostics()[-3:])
            raise ChatError("voice.stt_failed", detail=f"{error} {log}".strip()[:500]) from None
        finally:
            buf[: len(pcm)] = bytes(len(pcm))  # не оставлять звук в памяти
            del buf
            block.close()
            block.unlink()

    def close(self) -> None:
        with self._lock:
            self.process.stop()
