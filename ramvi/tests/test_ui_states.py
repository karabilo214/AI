"""3.5 (UI-03): сервис различает причины сбоя и отдаёт их кодами для интерфейса.

Поддельный движок падает при «загрузке» настоящей трассировкой Python
(`tests/fixtures/fake_engine.py`), сервис и HTTP — настоящие процессы/потоки.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from ramvi_service.app import build
from ramvi_service.chat import ChatService
from ramvi_service.contracts.modules import Authority
from ramvi_service.engines import EngineManager, SwitchableProvider, classify_failure
from ramvi_service.resources import ResourceBusy, SharedLLM
from ramvi_service.supervisor import RestartPolicy
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client, add_models

ROOT = Path(__file__).resolve().parents[1]


def traceback(*body: str) -> list[str]:
    return [
        "loading",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        *body,
    ]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (
            traceback("ramvi_engine.planning.automatic.NoMemoryPlan: no fit"),
            ("engine.no_memory", "no fit"),
        ),
        (
            traceback("torch.OutOfMemoryError: CUDA out of memory."),
            ("engine.no_memory", "CUDA out of memory."),
        ),
        (
            traceback("RuntimeError: CUDA error: out of memory"),
            ("engine.no_memory", "CUDA error: out of memory"),
        ),
        (traceback("MemoryError"), ("engine.no_memory", "")),
        (
            traceback("NotImplementedError: архитектура x пока не поддержана"),
            ("engine.incompatible", "архитектура x пока не поддержана"),
        ),
        (
            traceback("PermissionError: [Errno 13] Permission denied: 'm.gguf'"),
            ("engine.no_access", "[Errno 13] Permission denied: 'm.gguf'"),
        ),
        (
            traceback("FileNotFoundError: модель m не найдена"),
            ("engine.model_missing", "модель m не найдена"),
        ),
        (traceback("RuntimeError: weights corrupted"), None),
        (["just a log line", "RuntimeError: not after traceback"], None),
        ([], None),
    ],
)
def test_classify_failure(lines: list[str], expected: tuple[str, str] | None) -> None:
    assert classify_failure(lines) == expected


def test_classify_uses_last_exception_of_chained_traceback() -> None:
    lines = [
        *traceback("RuntimeError: CUDA error: device-side assert"),
        "",
        "The above exception was the direct cause of the following exception:",
        "",
        *traceback("NotImplementedError: тип Q3 у blk.0 не поддерживается")[1:],
        "    raise X",  # строка с отступом после исключения не считается исключением
    ]
    assert classify_failure(lines) == ("engine.incompatible", "тип Q3 у blk.0 не поддерживается")


def test_classify_ignores_log_lines_after_exception() -> None:
    lines = [*traceback("NotImplementedError: тип Q3"), "INFO: shutting down", "Done: 1"]
    assert classify_failure(lines) == ("engine.incompatible", "тип Q3")
    # Журнал без трассировки после обработанной ошибки не выдаёт себя за исключение.
    assert classify_failure([*traceback("    frame only")[:-1], '  File "y.py"']) is None


@pytest.fixture
def client(tmp_path: Path):  # type: ignore[no-untyped-def]
    add_models(
        tmp_path,
        "model-a",
        "model-oom",
        "model-cuda-oom",
        "model-incompatible",
        "model-denied",
        "model-missing",
        "model-crash",
    )
    fast = lambda: RestartPolicy(max_restarts=1, base_delay=0.05, max_delay=0.05)  # noqa: E731
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE, restart_policy=fast)
    server.start()
    try:
        yield Client(server.port)
    finally:
        server.close()
        engines.stop()


def send(client: Client, model_id: str, text: str = "привет") -> tuple[list[dict], float]:  # type: ignore[type-arg]
    chat_id = client.ok("POST", "/chats", {"model_id": model_id})["chat"]["id"]
    started = time.monotonic()
    events = client.events(f"/chats/{chat_id}/send", {"text": text})
    return events, time.monotonic() - started


@pytest.mark.parametrize(
    ("model_id", "code"),
    [
        ("model-oom", "engine.no_memory"),
        ("model-cuda-oom", "engine.no_memory"),
        ("model-incompatible", "engine.incompatible"),
        ("model-denied", "engine.no_access"),
        ("model-missing", "engine.model_missing"),
    ],
)
def test_definite_failure_is_reported_at_once(client: Client, model_id: str, code: str) -> None:
    events, elapsed = send(client, model_id)

    assert [e["type"] for e in events] == ["user", "status", "done"]
    done = events[-1]
    assert done["finish"] == "error" and done["message"] is None
    assert done["error"]["code"] == code
    params = done["error"]["params"]
    assert params["model_id"] == model_id
    assert Path(params["path"]).is_absolute() and params["path"].endswith(f"{model_id}.gguf")
    assert elapsed < 10, "однозначный сбой не должен ждать перезапусков"
    engine = client.ok("GET", "/status")["engine"]
    assert engine["state"] == "stopped" and engine["problem"]["code"] == code


def test_unknown_failure_is_retried_then_generic(client: Client) -> None:
    events, _ = send(client, "model-crash")

    error = events[-1]["error"]
    assert error["code"] == "engine.start_failed"
    assert client.ok("GET", "/status")["engine"]["problem"] is None


def test_good_model_after_failure_clears_problem(client: Client) -> None:
    send(client, "model-oom")
    events, _ = send(client, "model-a")

    assert events[-1]["error"] is None and events[-1]["finish"] == "stop"
    engine = client.ok("GET", "/status")["engine"]
    assert engine["state"] == "ready" and engine["problem"] is None


def test_full_queue_is_api_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    add_models(tmp_path, "model-a")

    with open_core(tmp_path) as db:
        provider = SwitchableProvider(EngineManager(Models(db), tmp_path / "run"))
        llm = SharedLLM(Authority(), provider)

        def full(*args: object, **kwargs: object) -> str:
            raise ResourceBusy("очередь ресурсов заполнена")

        monkeypatch.setattr(llm._broker, "acquire", full)
        chat = ChatService(db, llm)
        chat_id = chat.create("model-a")["id"]
        events = list(chat.send(chat_id, "x"))

    assert events[-1]["error"] == {"code": "api.busy", "params": {}}


def run_service(data_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "RAMVI_UI_TOKEN": TOKEN,
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("service", "engine", "storage")),
    }
    return subprocess.run(
        [sys.executable, "-P", "-m", "ramvi_service.app", "--data-dir", str(data_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def test_data_dir_that_is_a_file_reports_no_access(tmp_path: Path) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("не папка", encoding="utf-8")

    result = run_service(blocker / "inner")

    assert result.returncode == 3, result.stderr
    event = json.loads(result.stdout.splitlines()[0])
    assert event["event"] == "error" and event["code"] == "storage.no_access"
    assert event["params"]["path"] == str((blocker / "inner").resolve())
    assert event["params"]["detail"]


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="права POSIX, не root")
def test_read_only_data_dir_reports_no_access(tmp_path: Path) -> None:
    data = tmp_path / "ro"
    data.mkdir()
    data.chmod(0o500)
    try:
        result = run_service(data)
    finally:
        data.chmod(0o700)

    assert result.returncode == 3, result.stderr
    assert json.loads(result.stdout.splitlines()[0])["code"] == "storage.no_access"


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="права POSIX, не root")
def test_existing_database_in_read_only_dir_reports_no_access(tmp_path: Path) -> None:
    data = tmp_path / "ro"
    add_models(data, "model-a")  # база уже создана
    data.chmod(0o500)
    try:
        result = run_service(data)
    finally:
        data.chmod(0o700)

    assert result.returncode == 3, (result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[0])["code"] == "storage.no_access"


def test_concurrent_status_during_failure_does_not_deadlock(client: Client) -> None:
    """Пока движок падает, /status отвечает (монитор не держит блокировку менеджера)."""
    done = threading.Event()
    statuses: list[str] = []

    def poll() -> None:
        while not done.is_set():
            statuses.append(client.ok("GET", "/status")["engine"]["state"])

    thread = threading.Thread(target=poll)
    thread.start()
    try:
        send(client, "model-oom")
    finally:
        done.set()
        thread.join(10)
    assert not thread.is_alive() and statuses


def test_definite_failure_skips_default_restart_delays(tmp_path: Path) -> None:
    """С политикой по умолчанию 5 перезапусков заняли бы 1+2+4+8+16 = 31 с."""
    add_models(tmp_path, "model-oom")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        events, elapsed = send(Client(server.port), "model-oom")
    finally:
        server.close()
        engines.stop()
    assert events[-1]["error"]["code"] == "engine.no_memory"
    assert elapsed < 5, elapsed
