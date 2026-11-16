"""Сервисный контроллер изолированного Engine HTTP-процесса, без импорта CUDA."""

import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.processes import PARENT_WATCH_ENV, ProcessSupervisor
from ramvi_engine.protocol import check_api_version


class EngineProcess:
    def __init__(
        self,
        config: ModelConfig,
        *,
        startup_timeout: float = 180.0,
        command: list[str] | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not 0 < startup_timeout <= 3600:
            raise ValueError("invalid startup timeout")
        self.config, self.startup_timeout = config, startup_timeout
        self._command, self._cwd, self._env = command, cwd, env
        self._supervisor: ProcessSupervisor | None = None
        self._key = secrets.token_urlsafe(32)
        self._instance = secrets.token_hex(16)
        self._port: int | None = None
        self._ready = False
        self._failure: str | None = None
        self._last_exit: dict[str, Any] | None = None
        self._start_gate = threading.Lock()

    @property
    def api_key(self) -> str:
        """Ключ этого экземпляра — только для локального клиента сервиса, не в файлы."""
        return self._key

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("engine process not started")
        return f"http://127.0.0.1:{self._port}"

    def _arguments(self) -> list[str]:
        c = self.config
        args = [
            sys.executable,
            "-P",
            "-m",
            "ramvi_engine",
            "serve",
            "--model",
            c.model,
            "--maxt",
            str(c.maxt),
            "--tokenizer",
            c.tokenizer,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]
        if c.draft:
            args += ["--draft", c.draft]
        if c.resident is not None:
            args += ["--resident", str(c.resident)]
        if not c.cpu:
            args += ["--no-cpu"]
        planning = c.planning
        if planning is None:
            args += ["--plan-mode", "legacy"]
        else:
            args += [
                "--plan-mode",
                planning.mode,
                "--draft-policy",
                planning.draft,
                "--prefill",
                planning.prefill,
            ]
            args += ["--load-attempts", str(planning.max_load_attempts)]
            if planning.reserve_bytes is not None:
                args += ["--reserve-gib", str(planning.reserve_bytes / 2**30)]
            if planning.max_batch is not None:
                args += ["--max-batch", str(planning.max_batch)]
            if not planning.allow_paging:
                args += ["--no-paging"]
        return args

    def call(
        self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 30.0
    ) -> dict[str, Any]:
        if not path.startswith("/") or path.startswith("//") or not 0 < timeout <= 3600:
            raise ValueError("invalid local request")
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
        )

        # Ignore OS HTTP proxy settings: this transport is strictly local.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(
                self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
            ) -> None:
                raise urllib.error.HTTPError(
                    req.full_url, code, "local redirects disabled", headers, fp
                )

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            body = response.read(2**21 + 1)
        if len(body) > 2**21:
            raise RuntimeError("engine response too large")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise RuntimeError("invalid engine response")
        return value

    def start(self) -> dict[str, Any]:
        if not self._start_gate.acquire(blocking=False):
            raise RuntimeError("engine startup already in progress")
        try:
            return self._start()
        finally:
            self._start_gate.release()

    def _start(self) -> dict[str, Any]:
        if self._supervisor and self._supervisor.status()["state"] == "running" and self._ready:
            return self.status()
        self.stop()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = sock.getsockname()[1]
        env = {
            **(self._env or {}),
            "RAMVI_API_KEY": self._key,
            "RAMVI_INSTANCE_ID": self._instance,
            "RAMVI_ENGINE_PORT": str(self._port),
            PARENT_WATCH_ENV: "1",  # 3.3: движок выходит сам, если сервис умер
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1])
            + os.pathsep
            + (self._env or {}).get("PYTHONPATH", os.environ.get("PYTHONPATH", "")),
        }
        command = self._command or self._arguments()
        self._supervisor = ProcessSupervisor(command, cwd=self._cwd, env=env)
        self._supervisor.start()
        self._last_exit = None
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._supervisor.status()["state"] != "running":
                    raise RuntimeError("engine process exited during startup")
                try:
                    health = self.call(
                        "/health", timeout=min(1.0, max(0.01, deadline - time.monotonic()))
                    )
                    if health.get("instance_id") != self._instance:
                        raise RuntimeError("engine instance mismatch")
                    check_api_version(health.get("versions", {}).get("api", ""))
                    self._ready = True
                    self._failure = None
                    return self.status()
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            raise TimeoutError("engine startup timeout")
        except BaseException as exc:
            self._failure = type(exc).__name__
            self.stop()
            raise

    def status(self) -> dict[str, Any]:
        child = (
            self._supervisor.status()
            if self._supervisor
            else {"state": "stopped", "pid": None, "returncode": None}
        )
        if child["state"] == "exited" and self._supervisor is not None:
            self._last_exit = child
            self._supervisor.stop()  # reap descendants/jobs even before the user chooses restart
        alive = child["state"] == "running"
        return {
            "state": "ready"
            if alive and self._ready
            else "starting"
            if alive
            else "failed"
            if self._ready or self._failure
            else "stopped",
            "process": child,
            "failure": self._failure,
            "last_exit": self._last_exit,
            "restart_available": not alive,
        }

    def probe(self, timeout: float = 2.0) -> bool:
        """Отвечает ли именно этот экземпляр движка на /health (для монитора 3.3)."""
        if self._port is None or not self._ready:
            return False
        try:
            health = self.call("/health", timeout=timeout)
        except (OSError, ValueError, RuntimeError):
            return False
        return health.get("instance_id") == self._instance

    def describe(self) -> dict[str, Any]:
        """Публичные сведения для файла состояния: без API-ключа."""
        child = self._supervisor.status() if self._supervisor else {"pid": None}
        return {
            "base_url": None if self._port is None else self.base_url,
            "instance_id": self._instance,
            "pid": child.get("pid"),
            "model": self.config.model,
        }

    def stop(self) -> dict[str, Any]:
        if self._supervisor:
            self._supervisor.stop()
        self._ready = False
        return self.status()

    def restart(self) -> dict[str, Any]:
        self.stop()
        self._instance = secrets.token_hex(16)
        return self.start()

    def diagnostics(self) -> list[str]:
        return self._supervisor.diagnostics() if self._supervisor else []

    def close(self) -> None:
        self.stop()
