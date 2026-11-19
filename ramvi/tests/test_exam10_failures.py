"""8.6, экзамен 10 ТЗ: OOM, переполненный диск, повреждённая модель, обрыв клиента API,
недоступный драйвер, ошибка распознавания речи, падение движка — диагностируются понятным
кодом без повреждения данных. После каждого сбоя: база цела (`PRAGMA integrity_check`),
прежняя история на месте, следующий запрос работает."""

import http.client
import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.supervisor import RestartPolicy
from ramvi_storage.database import Database, DiskFull
from test_chat import ENGINE, TOKEN, Client, add_models, text_of  # noqa: E402


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[Any, Any, Client, Path]]:
    add_models(tmp_path, "model-a", "model-oom", "model-crash")
    fast = lambda: RestartPolicy(max_restarts=1, base_delay=0.05, max_delay=0.05)  # noqa: E731
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE, restart_policy=fast)
    server.start()
    try:
        yield server, engines, Client(server.port), tmp_path
    finally:
        server.close()
        engines.stop()


def history_is_intact(client: Client, db: Path, chat_id: str, expected: list[str]) -> None:
    with sqlite3.connect(db) as check:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    contents = [m["content"] for m in client.ok("GET", f"/chats/{chat_id}/messages")["messages"]]
    assert contents[: len(expected)] == expected


def first_dialog(client: Client) -> tuple[str, list[str]]:
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    events = client.events(f"/chats/{chat_id}/send", {"text": "раз"})
    assert events[-1]["finish"] == "stop"
    return chat_id, ["раз", text_of(events, "text")]


def test_oom_and_corrupt_model_are_diagnosed_history_kept(
    service: tuple[Any, Any, Client, Path],
) -> None:
    server, _, client, _ = service
    chat_id, history = first_dialog(client)
    for model_id, code in (
        ("model-oom", "engine.no_memory"),
        ("model-crash", "engine.start_failed"),
    ):
        other = client.ok("POST", "/chats", {"model_id": model_id})["chat"]["id"]
        done = client.events(f"/chats/{other}/send", {"text": "x"})[-1]
        assert done["finish"] == "error" and done["error"]["code"] == code
        history_is_intact(client, server.chat.chats_repo.db.path, chat_id, history)
    assert client.events(f"/chats/{chat_id}/send", {"text": "два"})[-1]["finish"] == "stop"


def test_engine_crash_mid_answer_saves_partial_and_recovers(
    service: tuple[Any, Any, Client, Path],
) -> None:
    server, engines, client, _ = service
    chat_id, history = first_dialog(client)
    events = client.events(f"/chats/{chat_id}/send", {"text": "CRASH_MID длинный ответ"})
    done = events[-1]
    assert done["finish"] == "error" and done["error"] is not None, done
    history_is_intact(
        client, server.chat.chats_repo.db.path, chat_id, history + ["CRASH_MID длинный ответ"]
    )
    supervisor = engines._supervisor
    deadline = time.monotonic() + 15
    while True:
        result = client.events(f"/chats/{chat_id}/send", {"text": "три"})[-1]
        if result["finish"] == "stop" or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert result["finish"] == "stop", result
    assert supervisor is not None and supervisor.status()["restarts"] >= 1


def test_full_disk_is_diagnosed_and_database_survives(
    service: tuple[Any, Any, Client, Path],
) -> None:
    server, _, client, _ = service
    chat_id, history = first_dialog(client)
    db = server.chat.chats_repo.db
    pages = db.all("PRAGMA page_count")[0][0]
    db.execute(f"PRAGMA max_page_count={pages}")  # база больше не растёт — как полный диск
    big = "ж" * 30_000
    status, body = client.request("POST", f"/chats/{chat_id}/send", {"text": big})
    assert body["error"]["code"] == "storage.disk_full", (status, body)
    assert status == 507
    db.execute("PRAGMA max_page_count=1073741823")
    history_is_intact(client, server.chat.chats_repo.db.path, chat_id, history)
    assert client.events(f"/chats/{chat_id}/send", {"text": "после"})[-1]["finish"] == "stop"


def test_api_client_disconnect_mid_stream_does_not_break_service(
    service: tuple[Any, Any, Client, Path],
) -> None:
    server, _, client, _ = service
    chat_id, history = first_dialog(client)
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request(
        "POST",
        f"/chats/{chat_id}/send",
        json.dumps({"text": "LONG оборвать"}),
        {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    response = conn.getresponse()
    response.read(200)
    conn.close()  # клиент ушёл посреди ответа
    deadline = time.monotonic() + 10
    while client.ok("GET", "/status").get("busy") and time.monotonic() < deadline:
        time.sleep(0.05)
    history_is_intact(client, server.chat.chats_repo.db.path, chat_id, history)
    for _ in range(50):
        done = client.events(f"/chats/{chat_id}/send", {"text": "дальше"})[-1]
        if done["finish"] == "stop":
            break
        time.sleep(0.1)
    assert done["finish"] == "stop"


def test_full_disk_while_saving_answer_is_reported(
    service: tuple[Any, Any, Client, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _, client, _ = service
    chat_id, history = first_dialog(client)
    repo = server.chat.chats_repo
    real = repo.add_message

    def full_for_answer(chat: str, role: str, *args: Any, **kwargs: Any) -> Any:
        if role == "assistant":
            raise DiskFull("database or disk is full")
        return real(chat, role, *args, **kwargs)

    monkeypatch.setattr(repo, "add_message", full_for_answer)
    done = client.events(f"/chats/{chat_id}/send", {"text": "вопрос"})[-1]
    assert done["finish"] == "error" and done["error"]["code"] == "storage.disk_full"
    assert done["error"]["params"]["path"] == str(repo.db.path.parent)
    monkeypatch.undo()
    history_is_intact(client, repo.db.path, chat_id, history + ["вопрос"])


def test_database_full_rolls_back_whole_transaction(tmp_path: Path) -> None:
    """Без места запись отменяется целиком: ни полстроки, ни сломанной базы; потом пишет снова."""
    db = Database(tmp_path / "t.sqlite")
    db.execute("CREATE TABLE t (v TEXT)")
    db.execute("INSERT INTO t VALUES ('было')")
    db.execute(f"PRAGMA max_page_count={db.all('PRAGMA page_count')[0][0]}")
    with pytest.raises(DiskFull), db.transaction():
        db.execute("INSERT INTO t VALUES ('маленькое')")
        db.execute("INSERT INTO t VALUES (?)", ("x" * 100_000,))
    assert [r[0] for r in db.all("SELECT v FROM t")] == ["было"]
    assert db.all("PRAGMA integrity_check")[0][0] == "ok"
    db.execute("PRAGMA max_page_count=1073741823")
    db.execute("INSERT INTO t VALUES ('после')")
    assert [r[0] for r in db.all("SELECT v FROM t")] == ["было", "после"]
    db.close()
