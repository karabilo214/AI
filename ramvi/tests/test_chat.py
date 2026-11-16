"""3.4 (UI-02): чат через HTTP API сервиса с настоящим процессом поддельного движка."""

import http.client
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.ui_api import UIServer
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "fixtures" / "fake_engine.py"
TOKEN = "t" * 32
ENGINE = [sys.executable, "-u", str(FAKE)]


def add_models(data_dir: Path, *ids: str) -> None:
    with open_core(data_dir) as db:
        for i, model_id in enumerate(ids):
            Models(db).upsert(
                Model(model_id, "local", "qwen3", f"models/{model_id}.gguf", float(i))
            )


class Client:
    def __init__(self, port: int, token: str = TOKEN) -> None:
        self.port, self.token = port, token

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        conn = self._conn()
        raw = None if body is None else json.dumps(body).encode()
        conn.request(
            method,
            "/ui/v1" + path,
            raw,
            {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            | (headers or {}),
        )
        response = conn.getresponse()
        value = json.loads(response.read())
        conn.close()
        return response.status, value

    def ok(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        status, value = self.request(method, path, body)
        assert status == 200, value
        return value

    def stream(
        self, path: str, body: dict[str, Any] | None = None
    ) -> Generator[dict[str, Any], None, None]:
        conn = self._conn()
        conn.request(
            "POST",
            "/ui/v1" + path,
            json.dumps(body or {}).encode(),
            {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise AssertionError(json.loads(response.read()))
        try:
            for line in response:
                if line.startswith(b"data: "):
                    yield json.loads(line[6:])
        finally:
            conn.close()

    def events(self, path: str, body: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self.stream(path, body))


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager, Client]]:
    add_models(tmp_path, "model-a", "model-b")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        yield server, engines, Client(server.port)
    finally:
        server.close()
        engines.stop()


def text_of(events: list[dict[str, Any]], kind: str) -> str:
    return "".join(e["text"] for e in events if e["type"] == kind)


def test_send_streams_reasoning_and_text_and_persists(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    chat = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]
    events = client.events(f"/chats/{chat['id']}/send", {"text": "Привет, как дела?\nвторая"})
    assert events[0]["type"] == "user" and events[0]["message"]["content"].startswith("Привет")
    assert {"type": "status", "state": "loading", "model_id": "model-a"} in events
    assert text_of(events, "reasoning") == "думаю ещё"
    assert text_of(events, "text") == "[model-a] echo: Привет, как дела? files=0"
    done = events[-1]
    assert done["type"] == "done" and done["finish"] == "stop" and done["error"] is None
    assert done["message"]["reasoning"] == "думаю ещё"
    messages = client.ok("GET", f"/chats/{chat['id']}/messages")["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "[model-a] echo: Привет, как дела? files=0"
    assert messages[1]["token_usage"] == {"completion_tokens": 4}
    chats = client.ok("GET", "/chats")["chats"]
    assert chats[0]["title"] == "Привет, как дела?" and not chats[0]["generating"]
    assert engines.status() | {"last_failure": None} == {
        "model_id": "model-a",
        "loading": None,
        "state": "ready",
        "last_failure": None,
    }
    # Второй ответ — без загрузки: модель уже в движке.
    events = client.events(f"/chats/{chat['id']}/send", {"text": "ещё"})
    assert not [e for e in events if e["type"] == "status"]


def test_stop_saves_partial_answer_and_chat_is_free_again(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, _, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    seen: list[dict[str, Any]] = []
    for event in client.stream(f"/chats/{chat_id}/send", {"text": "LONG"}):
        seen.append(event)
        if event["type"] == "text" and len([e for e in seen if e["type"] == "text"]) == 5:
            assert client.ok("GET", "/chats")["chats"][0]["generating"]
            status, busy = client.request("POST", f"/chats/{chat_id}/clear")
            assert status == 409 and busy["error"]["code"] == "chat.busy"
            assert client.ok("POST", f"/chats/{chat_id}/stop") == {"stopped": True}
    done = seen[-1]
    assert done["type"] == "done" and done["finish"] == "cancelled"
    partial = done["message"]["content"]
    assert partial.startswith("word word word") and len(partial.split()) < 400
    messages = client.ok("GET", f"/chats/{chat_id}/messages")["messages"]
    assert messages[-1]["finish"] == "cancelled" and messages[-1]["content"] == partial
    assert client.ok("POST", f"/chats/{chat_id}/stop") == {"stopped": False}
    events = client.events(f"/chats/{chat_id}/send", {"text": "после"})
    assert events[-1]["finish"] == "stop"


def test_regenerate_replaces_last_answer(service: tuple[UIServer, EngineManager, Client]) -> None:
    _, _, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    first = client.events(f"/chats/{chat_id}/send", {"text": "вопрос"})[-1]["message"]
    events = client.events(f"/chats/{chat_id}/regenerate")
    assert events[0] == {"type": "regenerate", "removed": [first["id"]]}
    second = events[-1]["message"]
    messages = client.ok("GET", f"/chats/{chat_id}/messages")["messages"]
    assert [m["id"] for m in messages][1:] == [second["id"]] and second["id"] != first["id"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    empty = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    status, error = client.request("POST", f"/chats/{empty}/regenerate")
    assert status == 400 and error["error"]["code"] == "chat.nothing_to_regenerate"


def test_switch_model_without_restarting_service(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    server, engines, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.events(f"/chats/{chat_id}/send", {"text": "раз"})
    pid_a = engines._supervisor.status()["target"]["pid"]  # type: ignore[union-attr]
    assert (
        client.ok("POST", f"/chats/{chat_id}/model", {"model_id": "model-b"})["chat"]["model_id"]
        == "model-b"
    )
    events = client.events(f"/chats/{chat_id}/send", {"text": "два"})
    assert {"type": "status", "state": "loading", "model_id": "model-b"} in events
    assert text_of(events, "text").startswith("[model-b] echo: два")
    assert engines.model_id == "model-b"
    assert engines._supervisor.status()["target"]["pid"] != pid_a  # type: ignore[union-attr]
    status, error = client.request("POST", f"/chats/{chat_id}/model", {"model_id": "нет"})
    assert status == 404 and error["error"]["code"] == "chat.model_not_found"
    assert client.ok("GET", "/status")["engine"]["model_id"] == "model-b"
    assert server.port == client.port  # тот же процесс и тот же API


def test_history_and_model_choice_survive_service_restart(tmp_path: Path) -> None:
    add_models(tmp_path, "model-a", "model-b")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        client = Client(server.port)
        chat_id = client.ok("POST", "/chats", {"model_id": "model-b"})["chat"]["id"]
        client.events(f"/chats/{chat_id}/send", {"text": "запомни"})
        before = client.ok("GET", f"/chats/{chat_id}/messages")["messages"]
    finally:
        server.close()
        engines.stop()
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        client = Client(server.port)
        (chat,) = client.ok("GET", "/chats")["chats"]
        assert chat["id"] == chat_id and chat["model_id"] == "model-b"
        assert client.ok("GET", f"/chats/{chat_id}/messages")["messages"] == before
        # Новый чат без явной модели берёт модель последнего чата.
        assert client.ok("POST", "/chats")["chat"]["model_id"] == "model-b"
    finally:
        server.close()
        engines.stop()


def test_attachments_limits_and_prompt(service: tuple[UIServer, EngineManager, Client]) -> None:
    _, _, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    files = [{"name": "C:\\docs\\a.txt", "text": "первый"}, {"name": "b.md", "text": "# второй"}]
    events = client.events(f"/chats/{chat_id}/send", {"text": "смотри", "attachments": files})
    assert [f["name"] for f in events[0]["message"]["attachments"]] == ["a.txt", "b.md"]
    assert text_of(events, "text").endswith("files=2")
    limits = client.ok("GET", "/status")["limits"]
    cases = [
        ([{"name": "x", "text": "a"}] * (limits["max_files"] + 1), "chat.too_many_files"),
        ([{"name": "big", "text": "я" * limits["max_file_bytes"]}], "chat.file_too_large"),
        ([{"name": "bin", "text": "a\x00b"}], "chat.attachment_not_text"),
        ([{"name": "", "text": "a"}], "chat.attachment_invalid"),
        ([{"name": "x", "text": "a", "path": "/etc"}], "chat.attachment_invalid"),
    ]
    for attachments, code in cases:
        status, error = client.request(
            "POST", f"/chats/{chat_id}/send", {"text": "x", "attachments": attachments}
        )
        assert (status, error["error"]["code"]) == (400, code)
    status, error = client.request("POST", f"/chats/{chat_id}/send", {"text": "   "})
    assert error["error"]["code"] == "chat.empty_message"
    assert len(client.ok("GET", f"/chats/{chat_id}/messages")["messages"]) == 2


def test_clear_delete_and_missing_chat(service: tuple[UIServer, EngineManager, Client]) -> None:
    _, _, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.events(f"/chats/{chat_id}/send", {"text": "x"})
    assert client.ok("POST", f"/chats/{chat_id}/clear") == {"removed": 2}
    assert client.ok("GET", f"/chats/{chat_id}/messages") == {"messages": []}
    assert (
        client.ok("POST", f"/chats/{chat_id}/title", {"title": "  Мой   чат "})["chat"]["title"]
        == "Мой чат"
    )
    assert client.ok("DELETE", f"/chats/{chat_id}") == {"deleted": True}
    status, error = client.request("GET", f"/chats/{chat_id}/messages")
    assert status == 404 and error["error"]["code"] == "chat.not_found"
    status, error = client.request("GET", "/nope")
    assert status == 404


def test_access_requires_token_and_rejects_browsers(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    server, _, client = service
    assert Client(server.port, "x" * 32).request("GET", "/chats")[0] == 401
    status, error = client.request("GET", "/chats", headers={"Origin": "http://evil.example"})
    assert (status, error["error"]["code"]) == (403, "api.forbidden")
    status, _ = client.request("GET", "/chats", headers={"Host": f"evil.example:{server.port}"})
    assert status == 403


def test_client_disconnect_cancels_generation(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, _, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    stream = client.stream(f"/chats/{chat_id}/send", {"text": "LONG"})
    for event in stream:
        if event["type"] == "text":
            break
    stream.close()  # интерфейс закрылся посреди ответа
    deadline = time.monotonic() + 10
    while client.ok("GET", "/chats")["chats"][0]["generating"]:
        assert time.monotonic() < deadline, "генерация не остановилась"
        time.sleep(0.05)
    messages = client.ok("GET", f"/chats/{chat_id}/messages")["messages"]
    assert messages[-1]["finish"] == "cancelled" and len(messages[-1]["content"].split()) < 400


def test_engine_crash_is_restarted_and_next_answer_works(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.events(f"/chats/{chat_id}/send", {"text": "раз"})
    supervisor = engines._supervisor
    assert supervisor is not None
    url = engines.provider().base_url
    with pytest.raises((OSError, http.client.HTTPException)):
        conn = http.client.HTTPConnection(url.split("//")[1], timeout=2)
        conn.request("GET", "/crash")
        conn.getresponse()
    deadline = time.monotonic() + 15
    while supervisor.status()["restarts"] < 1 or supervisor.status()["state"] != "ready":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    events = client.events(f"/chats/{chat_id}/send", {"text": "два"})
    assert events[-1]["finish"] == "stop" and text_of(events, "text").startswith("[model-a]")
    assert engines.provider().base_url != url  # новый порт после перезапуска


def test_two_chats_share_one_engine_queue(service: tuple[UIServer, EngineManager, Client]) -> None:
    _, _, client = service
    a = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    b = client.ok("POST", "/chats", {"model_id": "model-b"})["chat"]["id"]
    results: dict[str, list[dict[str, Any]]] = {}

    def run(chat_id: str, text: str) -> None:
        results[chat_id] = client.events(f"/chats/{chat_id}/send", {"text": text})

    threads = [
        threading.Thread(target=run, args=(a, "из A")),
        threading.Thread(target=run, args=(b, "из B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    # Смена модели ждёт конца чужого ответа: оба ответа полные и от своей модели.
    assert text_of(results[a], "text") == "[model-a] echo: из A files=0", results[b][-1]
    assert text_of(results[b], "text") == "[model-b] echo: из B files=0", results[b][-1]
    assert results[a][-1]["finish"] == results[b][-1]["finish"] == "stop"


def test_service_process_starts_and_exits_with_shell(tmp_path: Path) -> None:
    add_models(tmp_path, "model-a")
    env = {
        **os.environ,
        "RAMVI_UI_TOKEN": TOKEN,
        "RAMVI_EXIT_WITH_PARENT": "1",
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("service", "engine", "storage")),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ramvi_service.app",
            "--data-dir",
            str(tmp_path),
            "--engine-command-json",
            json.dumps(ENGINE),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        ready = json.loads(process.stdout.readline())
        assert ready["event"] == "ready"
        client = Client(ready["port"])
        chat_id = client.ok("POST", "/chats")["chat"]["id"]
        assert client.events(f"/chats/{chat_id}/send", {"text": "hi"})[-1]["finish"] == "stop"
        found = json.loads((tmp_path / "run" / "engine.json").read_text())
        engine_pid = found["target"]["pid"]
        process.stdin.close()  # оболочка закрылась
        assert process.wait(15) == 0
        deadline = time.monotonic() + 10
        while True:
            try:
                os.kill(engine_pid, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < deadline, "движок пережил сервис"
            time.sleep(0.05)
        assert json.loads((tmp_path / "run" / "engine.json").read_text())["state"] == "stopped"
    finally:
        process.kill()
        process.wait(5)
