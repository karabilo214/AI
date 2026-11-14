"""Настоящие дочерние процессы: RPC, зависание и аварийный рестарт HTTP Engine."""

import http.client
import os
import sys
import time
import urllib.error
from pathlib import Path

import pytest
from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.processes import ProcessSupervisor

RPC = """import json,sys,time
for line in sys.stdin:
 req=json.loads(line)
 if req['method']=='hang':time.sleep(60)
 if req['method']=='crash':sys.exit(17)
 print(json.dumps({'id':req['id'],'result':{'method':req['method'],'payload':req['payload']}}),flush=True)
"""
HTTP = """import json,os
from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path=='/crash':os._exit(17)
  self.send({'instance_id':os.environ['RAMVI_INSTANCE_ID'],'versions':{'api':'1.0'}})
 def do_POST(self):
  if self.headers.get('Authorization')!='Bearer '+os.environ['RAMVI_API_KEY']:
   self.send_response(401);self.end_headers();return
  self.send({'got':json.loads(self.rfile.read(int(self.headers['Content-Length'])))})
 def send(self,data):
  raw=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
HTTPServer(('127.0.0.1',int(os.environ['RAMVI_ENGINE_PORT'])),H).serve_forever()
"""


def test_rpc_restart_and_timeout_stop_worker() -> None:
    worker = ProcessSupervisor([sys.executable, "-u", "-c", RPC], timeout=1)
    try:
        for index in range(3):
            assert worker.start()["state"] == "running"
            assert worker.request("echo", {"number": 100003 + index})["payload"] == {
                "number": 100003 + index
            }
            assert worker.restart()["state"] == "running"
            assert worker.request("echo", {"after_restart": True})["payload"] == {
                "after_restart": True
            }
            worker.stop()
        worker.start()
        with pytest.raises(TimeoutError):
            worker.request("hang", {}, timeout=0.1)
        assert worker.status()["state"] == "stopped"
    finally:
        worker.stop()


def test_crash_is_reported_and_parent_remains_usable() -> None:
    engine = EngineProcess(
        ModelConfig("fixture"), command=[sys.executable, "-u", "-c", HTTP], startup_timeout=5
    )
    try:
        assert engine.start()["state"] == "ready"
        assert engine.call("/echo", {"number": 100003}) == {"got": {"number": 100003}}
        with pytest.raises(
            (http.client.RemoteDisconnected, urllib.error.URLError, ConnectionError)
        ):
            engine.call("/crash", timeout=1)
        deadline = time.monotonic() + 2
        while engine.status()["state"] != "failed" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.status()["state"] == "failed"
        assert engine.status()["restart_available"]
        assert engine.restart()["state"] == "ready"
        assert engine.call("/echo", {"after_crash": True}) == {"got": {"after_crash": True}}
    finally:
        assert engine.stop()["state"] == "stopped"


def test_startup_timeout_reaps_process() -> None:
    engine = EngineProcess(
        ModelConfig("fixture"),
        command=[sys.executable, "-c", "import time;time.sleep(60)"],
        startup_timeout=0.2,
    )
    with pytest.raises(TimeoutError):
        engine.start()
    assert engine.status()["state"] == "failed"
    assert engine.status()["process"]["pid"] is None


def test_stdout_is_bounded_and_cannot_wedge_rpc() -> None:
    worker = ProcessSupervisor(
        [sys.executable, "-u", "-c", "print('x'*70000,flush=True);import time;time.sleep(60)"],
        timeout=1,
    )
    try:
        worker.start()
        with pytest.raises(RuntimeError):
            worker.request("echo", {}, timeout=1)
        assert worker.status()["state"] == "stopped"
    finally:
        worker.stop()


def test_stop_kills_descendant_tree(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows Job ownership is exercised by the worker integration test")
    pidfile = tmp_path / "child.pid"
    source = "import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    worker = ProcessSupervisor([sys.executable, "-c", source, str(pidfile)])
    try:
        worker.start()
        deadline = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child = int(pidfile.read_text())
        worker.stop()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("descendant survived process group stop")
    finally:
        worker.stop()


def test_windows_bootstrap_does_not_execute_target_before_gate(tmp_path: Path) -> None:
    import subprocess

    from ramvi_engine.processes import _windows_bootstrap_command

    marker = tmp_path / "started"
    command = _windows_bootstrap_command(
        [
            sys.executable,
            "-u",
            "-c",
            "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('started');print(sys.stdin.readline().strip(),flush=True)",
            str(marker),
        ]
    )
    assert not any("\x00" in argument for argument in command)
    child = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        # The trusted bootstrap, rather than target code, is waiting for assignment.
        assert child.poll() is None
        assert not marker.exists()
        stdout, stderr = child.communicate("\x00payload\n", timeout=5)
        assert child.returncode == 0, stderr
        assert marker.read_text() == "started" and stdout.strip() == "payload"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_rpc_queue_wait_consumes_the_same_deadline() -> None:
    worker = ProcessSupervisor([sys.executable, "-u", "-c", RPC], timeout=1)
    try:
        worker.start()
        worker._rpc.acquire()
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="busy"):
            worker.request("echo", {}, timeout=0.03)
        assert time.monotonic() - started < 0.5
    finally:
        worker._rpc.release()
        worker.stop()


def test_timed_out_old_request_never_stops_restarted_worker() -> None:
    import threading

    worker = ProcessSupervisor([sys.executable, "-u", "-c", RPC], timeout=1)
    errors = []
    try:
        worker.start()

        def request() -> None:
            try:
                worker.request("hang", {}, timeout=0.2)
            except (RuntimeError, TimeoutError):
                errors.append(True)

        caller = threading.Thread(target=request)
        caller.start()
        time.sleep(0.03)
        worker.restart()
        caller.join(3)
        assert not caller.is_alive() and errors
        assert worker.status()["state"] == "running"
        assert worker.request("echo", {"fresh": 200044})["payload"] == {"fresh": 200044}
    finally:
        worker.stop()


def test_windows_job_reaps_descendant_after_command_exits(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows Job integration")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    pidfile = tmp_path / "owned-descendant.pid"
    source = "import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(str(p.pid))"
    worker = ProcessSupervisor([sys.executable, "-c", source, str(pidfile)])
    handle = None
    try:
        worker.start()
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int(pidfile.read_text())
        # Keep a handle to this exact owned process; PID reuse cannot affect cleanup.
        handle = kernel.OpenProcess(0x00100001, False, pid)  # SYNCHRONIZE | TERMINATE
        assert handle
        worker.stop()
        assert kernel.WaitForSingleObject(handle, 3000) == 0
    finally:
        worker.stop()
        if handle:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                kernel.WaitForSingleObject(handle, 3000)
            kernel.CloseHandle(handle)


def test_windows_stop_with_retained_job_handle(tmp_path: Path) -> None:
    """A retained Job handle must not keep CUDA descendants/pipes alive."""
    if sys.platform != "win32":
        pytest.skip("Windows Job duplicate-handle regression")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    marker = tmp_path / "ready"
    source = "from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text('ready');print('ready',flush=True);time.sleep(60)"
    worker = ProcessSupervisor([sys.executable, "-u", "-c", source, str(marker)])
    duplicate = wintypes.HANDLE()
    try:
        worker.start()
        end = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert marker.exists()
        assert worker._job is not None
        current = kernel.GetCurrentProcess()
        assert kernel.DuplicateHandle(
            current, worker._job.handle, current, ctypes.byref(duplicate), 0, False, 2
        )
        readers = list(worker._threads)
        started = time.monotonic()
        worker._job.close()  # same failure injection as the real Engine crash check
        worker.stop()
        assert time.monotonic() - started < 5
        assert not any(reader.is_alive() for reader in readers)
        assert worker.status()["state"] == "stopped"
    finally:
        if duplicate:
            kernel.CloseHandle(duplicate)
        worker.stop()


def test_windows_gate_bypasses_virtual_environment_redirector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ramvi_engine.processes import _windows_bootstrap_command

    monkeypatch.setattr(sys, "_base_executable", "base-python", raising=False)
    monkeypatch.setattr(sys, "executable", "venv-redirector")
    command = _windows_bootstrap_command(["venv-redirector", "-m", "ramvi_engine"])
    assert command[0] == "base-python"
    assert command[-3:] == ["venv-redirector", "-m", "ramvi_engine"]
