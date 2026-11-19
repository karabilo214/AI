"""Ограниченный supervisor дочернего процесса: общий для Engine и module workers."""

import _thread
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WINDOWS_BOOTSTRAP = r"""import os,subprocess,sys
# No target code runs before the supervisor assigns this process to its Job.
if os.read(sys.stdin.fileno(), 1) != b'\x00':
    raise SystemExit(125)
child = subprocess.Popen(sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
raise SystemExit(child.wait())
"""


def _windows_bootstrap_command(command: list[str]) -> list[str]:
    # -I/-S prevents user site/PYTHONPATH startup hooks before Job assignment.
    # A venv python.exe may be a redirector that spawns its real interpreter
    # BEFORE AssignProcessToJobObject. Use the real base interpreter for the gate.
    interpreter = getattr(sys, "_base_executable", None) or sys.executable
    return [interpreter, "-I", "-S", "-u", "-c", _WINDOWS_BOOTSTRAP, *command]


PARENT_WATCH_ENV = "RAMVI_EXIT_WITH_PARENT"


class ParentWatch:
    """Завершить дочерний процесс, когда владелец закрыл его stdin (3.3).

    Супервизор держит stdin ребёнка открытым и ничего в него не пишет. Если
    процесс-владелец падает или его убивают, ОС закрывает канал, ребёнок видит
    EOF: вызывает мягкую остановку и через `grace` секунд завершается сам.
    Так движок не остаётся сиротой, держащей порт и видеопамять, и никому не
    нужно убивать процессы по PID из файла.
    """

    def __init__(self, stream: Any = None, *, grace: float = 10.0) -> None:
        if not 0 <= grace <= 600:
            raise ValueError("invalid grace period")
        self._stream = sys.stdin if stream is None else stream
        self.grace = grace
        self._shutdown: Any = None
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="ramvi-parent-watch", daemon=True)

    @classmethod
    def from_env(cls) -> "ParentWatch | None":
        """Включить сторож, если его попросил супервизор.

        Мягкая остановка — KeyboardInterrupt в главном потоке, как Ctrl+C:
        срабатывают прежние `finally` загрузки и `serve`, модели закрываются.
        """
        if os.environ.get(PARENT_WATCH_ENV) != "1":
            return None
        watch = cls()
        watch.set_shutdown(_thread.interrupt_main)
        return watch.start()

    def start(self) -> "ParentWatch":
        self._thread.start()
        return self

    def set_shutdown(self, callback: Any) -> None:
        """Мягкая остановка (например `server.shutdown`); вызывается из потока сторожа."""
        # Если владелец уже ушёл, процесс и так завершится по истечении grace.
        self._shutdown = callback

    def _watch(self) -> None:
        try:
            fileno = self._stream.fileno()
            if sys.platform == "win32":
                if not self._wait_windows(fileno):
                    return  # stdin не канал (консоль, NUL): следить не за чем
            else:
                while os.read(fileno, 4096):
                    pass  # Владелец ничего не пишет; данные просто игнорируются.
        except (OSError, ValueError):
            pass
        self.closed.set()
        callback = self._shutdown
        if callback is not None:
            # Отдельный поток: мягкая остановка не должна мешать жёсткому сроку.
            threading.Thread(target=callback, name="ramvi-parent-shutdown", daemon=True).start()
        time.sleep(self.grace)
        os._exit(0)

    @staticmethod
    def _wait_windows(fileno: int, interval: float = 0.5) -> bool:
        """Ждать закрытия канала без висящего чтения; False — stdin не канал.

        Висящий синхронный ReadFile на канале stdin блокирует на Windows другие
        операции с этим дескриптором, в том числе его наследование при запуске
        дочерних процессов: настоящий движок при импорте torch/triton замирал.
        Поэтому канал опрашивается PeekNamedPipe и читается, только когда в нём
        уже есть данные.
        """
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        kernel.PeekNamedPipe.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        kernel.GetFileType.argtypes = [wintypes.HANDLE]
        handle = msvcrt.get_osfhandle(fileno)  # type: ignore[attr-defined,unused-ignore]
        if kernel.GetFileType(handle) != 3:  # FILE_TYPE_PIPE
            return False
        while True:
            available = wintypes.DWORD()
            if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
                return True  # ERROR_BROKEN_PIPE: владелец закрыл канал
            if available.value:
                if not os.read(fileno, min(available.value, 4096)):
                    return True
            else:
                time.sleep(interval)


class ProcessSupervisor:
    MAX_LINE = 65536

    def __init__(
        self,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not command or not 0 < timeout <= 3600:
            raise ValueError("invalid command/timeout")
        self.command, self.cwd, self.env, self.timeout = list(command), cwd, env, timeout
        self._process: subprocess.Popen[str] | None = None
        self._lock, self._rpc = threading.RLock(), threading.Lock()
        self._threads: list[threading.Thread] = []
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue(128)
        self._logs: deque[str] = deque(maxlen=100)
        self._error: str | None = None
        self._job: Any = None
        self._reader_stop = threading.Event()

    def status(self) -> dict[str, Any]:
        p = self._process
        code = p.poll() if p else None
        return {
            "state": "running" if p and code is None else "stopped" if p is None else "exited",
            "pid": p.pid if p else None,
            "returncode": code,
            "error": self._error,
        }

    def diagnostics(self) -> list[str]:
        return list(self._logs)

    @staticmethod
    def _windows_lines(pipe: Any, stopped: threading.Event) -> Iterator[str]:
        # Peek before os.read: closing a TextIOWrapper while readline holds its
        # lock can otherwise block stop() indefinitely after a worker crash.
        if sys.platform != "win32":
            raise OSError("Windows pipe reader requires Windows")
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        kernel.PeekNamedPipe.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        handle = msvcrt.get_osfhandle(pipe.fileno())
        pending = bytearray()
        while not stopped.is_set():
            available = wintypes.DWORD()
            if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
                break
            if not available.value:
                stopped.wait(0.01)
                continue
            chunk = os.read(pipe.fileno(), min(available.value, 4096))
            if not chunk:
                break
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending = bytearray(rest)
                yield (line + b"\n").decode("utf-8", errors="replace")
            if len(pending) > ProcessSupervisor.MAX_LINE:
                yield pending.decode("utf-8", errors="replace")
                return
        if pending and not stopped.is_set():
            yield pending.decode("utf-8", errors="replace")

    def _read(self, pipe: Any, responses: bool, stopped: threading.Event) -> None:
        try:
            lines = (
                self._windows_lines(pipe, stopped)
                if sys.platform == "win32"
                else iter(lambda: pipe.readline(self.MAX_LINE + 1), "")
            )
            for line in lines:
                if len(line.encode()) > self.MAX_LINE or not line.endswith("\n"):
                    self._error = "worker output frame too large/incomplete"
                    return
                self._logs.append(line.rstrip()[:2048])
                if responses:
                    try:
                        value = json.loads(line)
                        if (
                            isinstance(value, dict)
                            and "id" in value
                            and ("result" in value or "error" in value)
                        ):
                            self._responses.put_nowait(value)
                    except json.JSONDecodeError:
                        pass  # CLI Engine stdout is a bounded diagnostic log, not RPC.
                    except queue.Full:
                        self._error = "worker response queue full"
                        return
        except (OSError, ValueError):
            return

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()
            self.stop()
            self._error = None
            self._responses = queue.Queue(128)
            self._reader_stop = threading.Event()
            kwargs: dict[str, Any]
            if sys.platform == "win32":
                kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            else:
                kwargs = {"start_new_session": True}
            launch = (
                _windows_bootstrap_command(self.command)
                if sys.platform == "win32"
                else self.command
            )
            p = subprocess.Popen(
                launch,
                cwd=self.cwd,
                env={**os.environ, **(self.env or {})},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **kwargs,
            )
            self._process = p
            try:
                if sys.platform == "win32":
                    self._job = _WindowsJob(p)
                    assert p.stdin is not None
                    p.stdin.write("\x00")
                    p.stdin.flush()
                for pipe, rpc in ((p.stdout, True), (p.stderr, False)):
                    thread = threading.Thread(
                        target=self._read,
                        args=(pipe, rpc, self._reader_stop),
                        name="ramvi-process-reader",
                        daemon=True,
                    )
                    self._threads.append(thread)
                    thread.start()
            except BaseException:
                self.stop()
                raise
            return self.status()

    def request(
        self, method: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        budget = self.timeout if timeout is None else timeout
        if not 0 < budget <= 3600:
            raise ValueError("invalid timeout")
        deadline = time.monotonic() + budget
        if not self._rpc.acquire(timeout=budget):
            raise TimeoutError("worker RPC busy")
        try:
            p = self._process
            if p is None or p.poll() is not None or p.stdin is None:
                raise RuntimeError("worker is not running")
            request_id = uuid.uuid4().hex
            encoded = (
                json.dumps(
                    {"id": request_id, "method": method, "payload": payload}, allow_nan=False
                )
                + "\n"
            )
            if len(encoded.encode()) > self.MAX_LINE:
                raise ValueError("worker request too large")
            sent = threading.Event()
            failures: list[BaseException] = []

            def send() -> None:
                try:
                    assert p.stdin is not None
                    p.stdin.write(encoded)
                    p.stdin.flush()
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    sent.set()

            writer = threading.Thread(target=send, name="ramvi-process-writer", daemon=True)
            writer.start()
            try:
                if not sent.wait(max(0, deadline - time.monotonic())):
                    raise TimeoutError("worker input timeout")
                if failures:
                    raise RuntimeError("worker input failed") from failures[0]
                while True:
                    if self._error or p.poll() is not None:
                        raise RuntimeError(self._error or "worker exited")
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("worker RPC timeout")
                    try:
                        response = self._responses.get(timeout=min(0.05, left))
                    except queue.Empty:
                        continue
                    if response.get("id") != request_id:
                        continue
                    if "error" in response:
                        raise RuntimeError(str(response["error"]))
                    result = response.get("result")
                    if not isinstance(result, dict):
                        raise RuntimeError("invalid worker result")
                    return result
            except BaseException:
                # A concurrent restart may already own another process.
                with self._lock:
                    if self._process is p:
                        self.stop()
                raise
            finally:
                writer.join(timeout=1)
        finally:
            self._rpc.release()

    def stop(self, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            p = self._process
            if p is None:
                return self.status()
            budget = self.timeout if timeout is None else timeout
            if not 0 < budget <= 3600:
                raise ValueError("invalid timeout")
            if self._job is not None:
                self._job.close()
                self._job = None
            elif sys.platform == "win32":
                if p.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                        capture_output=True,
                        timeout=min(budget, 10),
                        check=False,
                    )
            else:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                p.wait(timeout=min(budget, 2))
            except subprocess.TimeoutExpired:
                if sys.platform != "win32":
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    p.kill()
                p.wait(timeout=budget)
            # Root has been reaped. Descendants may still own its process group;
            # an inaccessible stale group must not prevent closing RPC pipes/threads.
            if sys.platform != "win32":
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self._reader_stop.set()
            for thread in self._threads:
                thread.join(timeout=1)
            for pipe in (p.stdin, p.stdout, p.stderr):
                if pipe is not None:
                    pipe.close()
            self._threads.clear()
            self._process = None
            return self.status()

    def restart(self) -> dict[str, Any]:
        with self._lock:
            self.stop()
            return self.start()


class _WindowsJob:
    kernel: Any
    handle: Any

    def __init__(self, process: subprocess.Popen[str]) -> None:
        import ctypes
        from ctypes import wintypes

        if sys.platform != "win32":
            raise OSError("Windows Jobs require Windows")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]

        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t),
                ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        self.kernel, self.handle = kernel, kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle:
            raise OSError("CreateJobObject failed")
        owned_process: Any = process  # CPython Windows handle, absent from portable stubs.
        if not kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ) or not kernel.AssignProcessToJobObject(self.handle, int(owned_process._handle)):
            self.close()
            raise OSError("AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            # Explicit termination also works if another inherited/duplicated
            # handle keeps the Job alive after this owner closes its handle.
            self.kernel.TerminateJobObject(self.handle, 1)
            self.kernel.CloseHandle(self.handle)
            self.handle = None
