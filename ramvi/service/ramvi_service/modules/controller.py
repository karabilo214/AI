"""Адаптер к единому процессному супервизору Engine/модулей."""

import os
import sys
from pathlib import Path
from typing import Any, Protocol


class WorkerController(Protocol):
    def call(self, method: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...
    def stop(self) -> None: ...
    def status(self) -> dict[str, Any]: ...


class ProcessModuleWorker:
    def __init__(self, entrypoint: str, *, timeout: float = 10) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        service = Path(__file__).resolve().parents[2]
        engine = service.parent / "engine"
        paths = [str(service), str(engine), str(service.parent.parent)]
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "PYTHONUTF8": "1"}
        self.process = ProcessSupervisor(
            [sys.executable, "-P", "-m", "ramvi_service.modules.worker", entrypoint],
            env=env,
            timeout=timeout,
        )

    def call(self, method: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if method == "initialize":
            self.process.start()
        elif self.process.status()["state"] != "running":
            raise RuntimeError("module worker exited; initialize/recovery required")
        return self.process.request(method, payload, timeout=timeout)

    def stop(self) -> None:
        self.process.stop()

    def status(self) -> dict[str, Any]:
        return self.process.status()
