"""Адаптер к единому процессному супервизору Engine/модулей."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ramvi_service.modules.worker import MAX_LINE


class WorkerController(Protocol):
    def call(self, method: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...
    def stop(self) -> None: ...
    def status(self) -> dict[str, Any]: ...


class ProcessModuleWorker:
    def __init__(
        self, entrypoint: str, *, timeout: float = 10, extra_paths: Sequence[str] = ()
    ) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        service = Path(__file__).resolve().parents[2]
        engine = service.parent / "engine"
        paths = [
            *extra_paths,  # 8.8: код установленного пакета модуля (modules/<id>/code)
            str(service),
            str(engine),
            str(service.parent / "character"),
            str(service.parent.parent),
        ]
        if os.environ.get("PYTHONPATH"):
            paths.extend(os.environ["PYTHONPATH"].split(os.pathsep))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "PYTHONUTF8": "1"}
        self.process = ProcessSupervisor(
            [
                sys.executable,
                "-P",
                "-c",
                "import sys,runpy; sys.path[:0]=sys.argv[2:]; "
                "sys.argv=['ramvi_service.modules.worker',sys.argv[1]]; "
                "runpy.run_module('ramvi_service.modules.worker',run_name='__main__')",
                entrypoint,
                *paths,
            ],
            env=env,
            timeout=timeout,
            max_line=MAX_LINE,
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
