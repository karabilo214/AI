"""6.5 (AG-05): короткий дочерний процесс с лимитами времени и вывода.

Процесс запускается без оболочки, со своим stdin = пусто, в отдельной группе
(POSIX) или в Job Object (Windows, тот же механизм, что у supervisor движка).
По окончанию, тайм-ауту или отмене убивается всё дерево — оставшиеся фоновые
потомки тоже, чтобы ресурсы освобождались. stdout и stderr читаются до конца, но
хранится не больше лимита; признак `truncated` сообщает об обрезке.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ramvi_engine.processes import _windows_bootstrap_command, _WindowsJob


@dataclass(frozen=True)
class ProcessLimits:
    """Лимиты одного запуска."""

    timeout: float | None = 120.0  # None — без ограничения (остановит только отмена)
    max_output_bytes: int = 256 * 1024  # на каждый поток


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None  # None — процесс прерван (тайм-аут или отмена)
    stdout: str
    stderr: str
    timed_out: bool
    cancelled: bool
    truncated: bool
    duration: float


class _Capture:
    def __init__(self, pipe: IO[bytes], limit: int) -> None:
        self.data, self.total, self.limit = bytearray(), 0, limit
        self.thread = threading.Thread(
            target=self._run, args=(pipe,), name="ramvi-agent-capture", daemon=True
        )
        self.thread.start()

    def _run(self, pipe: IO[bytes]) -> None:
        try:
            while chunk := os.read(pipe.fileno(), 65536):
                self.total += len(chunk)
                room = self.limit - len(self.data)
                if room > 0:
                    self.data += chunk[:room]
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def _kill_tree(process: subprocess.Popen[bytes], job: Any) -> None:
    if sys.platform == "win32":
        if job is not None:
            job.close()  # TerminateJobObject: весь процесс и его потомки
        elif process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_process(
    argv: list[str],
    cwd: Path,
    limits: ProcessLimits | None = None,
    *,
    env: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
) -> ProcessResult:
    """Запустить процесс и дождаться его в пределах лимитов.

    `FileNotFoundError` — программы нет; остальные ошибки запуска — `OSError`.
    """
    if not argv:
        raise ValueError("empty command")
    limits = limits or ProcessLimits()
    windows = sys.platform == "win32"
    kwargs: dict[str, Any] = (
        {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        }
        if windows
        else {"start_new_session": True}
    )
    started = time.monotonic()
    process = subprocess.Popen(
        _windows_bootstrap_command(argv) if windows else argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if windows else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    job: Any = None
    captures: list[_Capture] = []
    timed_out = cancelled = False
    try:
        if windows:
            job = _WindowsJob(process)  # type: ignore[arg-type,unused-ignore]
            assert process.stdin is not None
            process.stdin.write(b"\x00")  # шлюз bootstrap: процесс уже в Job
            process.stdin.close()
        assert process.stdout is not None and process.stderr is not None
        captures = [
            _Capture(process.stdout, limits.max_output_bytes),
            _Capture(process.stderr, limits.max_output_bytes),
        ]
        deadline = None if limits.timeout is None else started + limits.timeout
        while True:
            try:
                process.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
    finally:
        _kill_tree(process, job)
        process.wait()
        for capture in captures:
            capture.thread.join(2.0)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
    out, err = captures
    return ProcessResult(
        exit_code=None if timed_out or cancelled else process.returncode,
        stdout=out.text(),
        stderr=err.text(),
        timed_out=timed_out,
        cancelled=cancelled,
        truncated=out.total > len(out.data) or err.total > len(err.data),
        duration=time.monotonic() - started,
    )
