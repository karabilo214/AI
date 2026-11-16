"""3.3: супервизор на настоящих процессах — старт, падение, зависание, владельцы."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.processes import ProcessSupervisor
from ramvi_service.supervisor import (
    OwnershipError,
    ProcessTarget,
    RestartPolicy,
    RunLock,
    Supervisor,
    discover,
)

ROOT = Path(__file__).resolve().parents[1]


def fast(max_restarts: int = 5) -> RestartPolicy:
    return RestartPolicy(max_restarts=max_restarts, base_delay=0.05, max_delay=0.2)


# Поддельный движок: тот же HTTP-контракт и тот же сторож родителя, что у `serve`;
# по концу stdin сторож прерывает serve_forever и срабатывает finally.
ENGINE = """import json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from ramvi_engine.processes import ParentWatch
watch=ParentWatch.from_env()
class H(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path=='/crash':os._exit(17)
  self.send({'instance_id':os.environ['RAMVI_INSTANCE_ID'],'versions':{'api':'1.0'}})
 def send(self,data):
  raw=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
srv=ThreadingHTTPServer(('127.0.0.1',int(os.environ['RAMVI_ENGINE_PORT'])),H)
try:srv.serve_forever()
finally:
 if watch and watch.closed.is_set():open(os.environ['FIXTURE_MARK'],'w').write('graceful')
"""


def engine(command: str = ENGINE) -> EngineProcess:
    return EngineProcess(
        ModelConfig("fixture"), command=[sys.executable, "-u", "-c", command], startup_timeout=5
    )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def until(check: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return False


posix = pytest.mark.skipif(sys.platform == "win32", reason="сигналы POSIX")


def test_restart_policy_backoff_and_limit() -> None:
    policy = RestartPolicy(max_restarts=3, window=10, base_delay=1, max_delay=3)
    assert [policy.next_delay(t) for t in (0, 1, 2)] == [1, 2, 3]
    assert policy.next_delay(3) is None  # 3 перезапуска за окно — хватит
    assert policy.next_delay(10.5) == 3  # первый вышел из окна; серия не сброшена
    policy.succeeded()
    assert policy.next_delay(20) == 1
    policy.reset()
    assert [policy.next_delay(30) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(ValueError):
        RestartPolicy(base_delay=5, max_delay=1)


def test_run_lock_is_exclusive_and_released_when_owner_dies(tmp_path: Path) -> None:
    path = tmp_path / "x.lock"
    first, second = RunLock(path), RunLock(path)
    assert first.acquire() and not second.acquire() and RunLock.held(path)
    first.release()
    assert not RunLock.held(path) and second.acquire()
    second.release()
    code = textwrap.dedent(
        f"""
        import sys,time
        from pathlib import Path
        from ramvi_service.supervisor import RunLock
        lock=RunLock(Path({str(path)!r}))
        assert lock.acquire()
        print('locked',flush=True);time.sleep(60)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "service",)))},
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        assert RunLock.held(path)
        holder.kill()  # без release: блокировку снимает ОС
        holder.wait(5)
        assert not RunLock.held(path)
    finally:
        holder.kill()
        holder.wait(5)


def test_engine_start_discovery_and_auto_restart_after_crash(tmp_path: Path) -> None:
    target = engine()
    changes: list[str] = []
    sup = Supervisor(
        target,
        tmp_path,
        policy=fast(),
        interval=0.05,
        on_change=lambda s: changes.append(s["state"]),
    )
    try:
        assert sup.start()["state"] == "starting"
        assert sup.wait("ready", timeout=10)["state"] == "ready"
        found = discover(tmp_path, "engine")
        assert found is not None and found["live"] and found["state"] == "ready"
        first_pid = found["target"]["pid"]
        assert found["target"]["base_url"] == target.base_url
        assert found["owner"]["pid"] == os.getpid()
        assert target._key not in (tmp_path / "engine.json").read_text()  # ключ не утекает

        with pytest.raises(OSError):
            urllib.request.urlopen(target.base_url + "/crash", timeout=2)
        assert until(lambda: sup.status()["restarts"] == 1, 10)
        status = sup.wait("ready", timeout=10)
        assert status["state"] == "ready"
        assert status["last_failure"]["reason"] == "exited"
        assert status["last_failure"]["process"]["returncode"] == 17
        assert status["target"]["pid"] != first_pid and target.probe()
        assert changes[:2] == ["starting", "ready"] and "restarting" in changes
    finally:
        final = sup.stop()
    assert final["state"] == "stopped" and target.status()["state"] == "stopped"
    found = discover(tmp_path, "engine")
    assert found is not None and not found["live"] and found["state"] == "stopped"


@posix
def test_hung_engine_is_detected_by_probe_and_replaced(tmp_path: Path) -> None:
    target = engine()
    sup = Supervisor(target, tmp_path, policy=fast(), interval=0.05, failure_threshold=2)
    try:
        sup.start()
        pid = sup.wait("ready", timeout=10)["target"]["pid"]
        os.kill(pid, signal.SIGSTOP)  # type: ignore[attr-defined,unused-ignore]
        # процесс жив, но не отвечает
        assert until(lambda: sup.status()["restarts"] == 1, 15)
        status = sup.wait("ready", timeout=10)
        assert status["last_failure"]["reason"] == "no response 2 times"
        assert status["target"]["pid"] != pid and gone(pid, 5)
    finally:
        sup.stop()


def test_crash_loop_stops_at_limit_until_manual_restart(tmp_path: Path) -> None:
    target = engine("import sys;sys.exit(3)")
    sup = Supervisor(target, tmp_path, policy=fast(2))
    try:
        sup.start()
        status = sup.wait("failed", timeout=10)
        assert status["state"] == "failed" and status["restarts"] == 2
        assert status["last_failure"]["reason"] == "start failed"
        time.sleep(0.5)  # в failed больше никаких попыток
        assert sup.status()["restarts"] == 2
        sup.restart()
        assert until(lambda: sup.status()["restarts"] == 4, 10)
        assert sup.wait("failed", timeout=10)["state"] == "failed"
    finally:
        sup.stop()


def test_second_owner_is_refused_and_does_not_touch_running_engine(tmp_path: Path) -> None:
    target, other = engine(), engine()
    first = Supervisor(target, tmp_path, policy=fast(), interval=0.05)
    second = Supervisor(other, tmp_path, policy=fast(), interval=0.05)
    try:
        first.start()
        pid = first.wait("ready", timeout=10)["target"]["pid"]
        with pytest.raises(OwnershipError) as refused:
            second.start()
        assert refused.value.state is not None and refused.value.state["live"]
        second.stop()  # остановка «чужого» имени не должна трогать первый движок
        assert other.status()["state"] == "stopped"
        time.sleep(0.2)
        assert first.status()["state"] == "ready" and target.probe()
        assert first.status()["target"]["pid"] == pid and first.status()["restarts"] == 0
        found = discover(tmp_path, "engine")
        assert found is not None and found["live"] and found["state"] == "ready"
        assert found["owner"]["id"] == first.status()["owner"]["id"]
    finally:
        first.stop()


def test_stale_state_file_pid_is_never_killed(tmp_path: Path) -> None:
    stranger = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        stale = {
            "version": 1,
            "name": "engine",
            "state": "ready",
            "owner": {"id": "old", "pid": stranger.pid, "started_at": 0},
            "target": {"pid": stranger.pid, "base_url": "http://127.0.0.1:9"},
            "updated_at": 0,
        }
        (tmp_path / "engine.json").write_text(json.dumps(stale))
        found = discover(tmp_path, "engine")
        assert found is not None and not found["live"]  # владелец не держит блокировку
        sup = Supervisor(engine(), tmp_path, policy=fast(), interval=0.05)
        try:
            sup.start()
            status = sup.wait("ready", timeout=10)
            assert status["previous"]["target_pid"] == stranger.pid
            assert not status["previous"]["clean_exit"]
        finally:
            sup.stop()
        assert stranger.poll() is None  # чужой процесс с тем же PID жив
    finally:
        stranger.kill()
        stranger.wait(5)


SERVICE = """import sys,time
from pathlib import Path
from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.model_loader import ModelConfig
from ramvi_service.supervisor import Supervisor
target=EngineProcess(ModelConfig('fixture'),command=[sys.executable,'-u','-c',sys.argv[2]],startup_timeout=5)
sup=Supervisor(target,Path(sys.argv[1]),interval=0.05)
sup.start()
print(sup.wait('ready',timeout=10)['target']['pid'],flush=True)
time.sleep(60)
"""


def test_engine_exits_by_itself_when_service_is_killed(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("service", "engine", "storage")),
        "FIXTURE_MARK": str(tmp_path / "mark"),
    }
    service = subprocess.Popen(
        [sys.executable, "-u", "-c", SERVICE, str(tmp_path), ENGINE],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert service.stdout is not None
        pid = int(service.stdout.readline())
        found = discover(tmp_path, "engine")
        assert found is not None and found["live"] and found["target"]["pid"] == pid
        service.kill()  # аварийная смерть сервиса: никакого stop()
        service.wait(5)
        assert gone(pid, 10), "движок остался сиротой"
        # Мягкий путь: serve_forever прерван, finally выполнился (а не os._exit по сроку).
        assert (tmp_path / "mark").read_text() == "graceful"
        found = discover(tmp_path, "engine")
        assert found is not None and not found["live"] and found["state"] == "ready"
        # Новый владелец занимает имя после аварии прежнего.
        sup = Supervisor(engine(), tmp_path, interval=0.05)
        try:
            sup.start()
            assert sup.wait("ready", timeout=10)["previous"]["target_pid"] == pid
        finally:
            sup.stop()
    finally:
        service.kill()
        service.wait(5)


RPC = """import json,sys
for line in sys.stdin:
 req=json.loads(line)
 if req['method']=='crash':sys.exit(5)
 print(json.dumps({'id':req['id'],'result':{'ok':True}}),flush=True)
"""


def test_module_workers_use_same_supervisor_and_stay_independent(tmp_path: Path) -> None:
    def ping(process: ProcessSupervisor) -> bool:
        try:
            return process.request("ping", {}, timeout=1)["ok"] is True
        except (RuntimeError, TimeoutError):
            return False

    workers = {
        name: ProcessSupervisor([sys.executable, "-u", "-c", RPC], timeout=2)
        for name in ("mod-a", "mod-b")
    }
    sups = {
        name: Supervisor(
            ProcessTarget(proc, ping),
            tmp_path,
            name=name,
            policy=fast(),
            interval=0.05,
        )
        for name, proc in workers.items()
    }
    try:
        for sup in sups.values():
            sup.start()
        pids = {n: s.wait("ready", timeout=10)["target"]["pid"] for n, s in sups.items()}
        with pytest.raises(RuntimeError):
            workers["mod-a"].request("crash", {}, timeout=1)
        assert until(lambda: sups["mod-a"].status()["restarts"] == 1, 10)
        assert sups["mod-a"].wait("ready", timeout=10)["target"]["pid"] != pids["mod-a"]
        assert sups["mod-b"].status()["restarts"] == 0
        assert sups["mod-b"].status()["target"]["pid"] == pids["mod-b"]
        sups["mod-a"].stop()
        assert ping(workers["mod-b"]) and sups["mod-b"].status()["state"] == "ready"
    finally:
        for sup in sups.values():
            sup.stop()


PARENT = """import sys,time
from ramvi_engine.processes import ParentWatch
w=ParentWatch(grace=float(sys.argv[1])).start()
if sys.argv[2]=='soft':w.set_shutdown(lambda:(print('soft',file=sys.stderr,flush=True)))
print('ready',flush=True);time.sleep(60)
"""


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_parent_watch_exits_on_stdin_eof(mode: str) -> None:
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", PARENT, "0.2", mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "engine")},
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        start = time.monotonic()
        assert child.stdin is not None
        child.stdin.close()
        assert child.wait(5) == 0
        assert time.monotonic() - start < 3
        assert child.stderr is not None
        assert ("soft" in child.stderr.read()) is (mode == "soft")
    finally:
        child.kill()
        child.wait(5)


def test_invalid_names_and_settings_are_rejected(tmp_path: Path) -> None:
    for name in ("", "../x", "Engine", "a" * 65):
        with pytest.raises(ValueError):
            Supervisor(engine(), tmp_path, name=name)
    with pytest.raises(ValueError):
        Supervisor(engine(), tmp_path, interval=0)
    assert discover(tmp_path, "engine") is None
    (tmp_path / "engine.json").write_text("{broken")
    assert discover(tmp_path, "engine") is None
