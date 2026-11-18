"""7.1 (API-01): внешний API — ключи, модель, общая очередь, поток, отмена, discovery."""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from ramvi_service.api_server import ApiOptions
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.ui_api import UIServer
from test_chat import ENGINE, TOKEN, add_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Api:
    def __init__(self, port: int, key: str | None) -> None:
        self.port, self.key = port, key

    def call(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, key: str | None = ""
    ) -> tuple[int, Any]:
        token = self.key if key == "" else key
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        kind = response.getheader("Content-Type") or ""
        return response.status, json.loads(raw) if "json" in kind else raw.decode()


def chat_body(text: str, **extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}], "temperature": 0, **extra}


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager, Api, Any]]:
    add_models(tmp_path, "model-a", "model-b")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    record, key = server.api.keys.create("ide", {"chat", "models"})
    try:
        yield server, engines, Api(server.api.port, key), record
    finally:
        server.close()
        engines.stop()


def test_keys_are_required(api: tuple[UIServer, EngineManager, Api, Any]) -> None:
    server, engines, client, record = api
    engines.ensure("model-a")
    assert client.call("GET", "/v1/models", key=None)[0] == 401
    assert client.call("GET", "/v1/models", key="wrong-key")[0] == 401
    status, body = client.call("POST", "/v1/chat/completions", chat_body("hi"), key=None)
    assert status == 401 and body["error"]["type"] == "api.unauthorized"
    _, models_only = server.api.keys.create("viewer", {"models"})
    assert client.call("GET", "/v1/models", key=models_only)[0] == 200
    assert client.call("POST", "/v1/chat/completions", chat_body("hi"), key=models_only)[0] == 401
    server.api.keys.revoke(record.id)
    assert client.call("GET", "/v1/models")[0] == 401
    assert client.call("GET", "/")[0] == 200  # «жив ли сервис» — без ключа и без данных


def test_no_model_and_wrong_model(api: tuple[UIServer, EngineManager, Api, Any]) -> None:
    _, engines, client, _ = api
    status, body = client.call("POST", "/v1/chat/completions", chat_body("hi"))
    assert status == 503 and body["error"]["type"] == "api.no_model"
    engines.ensure("model-a")
    status, body = client.call("POST", "/v1/chat/completions", chat_body("hi", model="model-b"))
    assert status == 404 and body["error"]["type"] == "api.model_not_loaded"
    assert body["error"]["params"] == {"asked": "model-b", "loaded": "model-a"}
    assert engines.model_id == "model-a"  # чужую модель запрос не переключил


def test_models_listing(api: tuple[UIServer, EngineManager, Api, Any]) -> None:
    _, engines, client, _ = api
    engines.ensure("model-a")
    assert client.call("GET", "/v1/models")[1]["data"][0]["id"] == "model-a"
    assert client.call("GET", "/api/tags")[1]["models"][0]["name"] == "model-a"


def test_openai_stream_anthropic_and_ollama(api: tuple[UIServer, EngineManager, Api, Any]) -> None:
    _, engines, client, _ = api
    engines.ensure("model-a")
    status, text = client.call("POST", "/v1/chat/completions", chat_body("ping", model="model-a"))
    assert status == 200 and "data: " in text and "echo:" in text and "[DONE]" in text
    # Имя модели в запросе можно не указывать.
    assert client.call("POST", "/v1/chat/completions", chat_body("ping"))[0] == 200
    status, message = client.call("POST", "/v1/messages", chat_body("hello", model="model-a"))
    assert status == 200 and "/v1/messages echo: hello" in message["content"][0]["text"]
    status, ollama = client.call("POST", "/api/chat", chat_body("hey", model="MODEL-A"))
    assert status == 200 and "/api/chat echo: hey" in ollama["message"]["content"]


def test_external_requests_share_the_model_queue(
    api: tuple[UIServer, EngineManager, Api, Any],
) -> None:
    server, engines, client, _ = api
    engines.ensure("model-a")
    llm = server.api.llm
    other = llm.authority.issue("chat", "local_user", "busy")
    llm.retain(other)
    released = threading.Event()

    def hold() -> None:
        with llm.exclusive(other):
            time.sleep(0.7)
        released.set()

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.1)
    started = time.monotonic()
    status, _ = client.call("POST", "/v1/chat/completions", chat_body("wait"))
    waited = time.monotonic() - started
    holder.join()
    assert status == 200 and released.is_set() and waited >= 0.5


def test_client_disconnect_cancels_generation(
    api: tuple[UIServer, EngineManager, Api, Any],
) -> None:
    _, engines, client, _ = api
    engines.ensure("model-a")
    sock = socket.create_connection(("127.0.0.1", client.port), timeout=10)
    body = json.dumps(chat_body("LONG answer please")).encode()
    sock.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Authorization: Bearer {client.key}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    assert b"200" in sock.recv(4096)
    sock.close()  # клиент ушёл посреди длинного ответа (400 слов ≈ 4 с)
    engine = urlsplit(engines.provider().base_url)

    def cancelled() -> list[str]:
        conn = http.client.HTTPConnection(engine.hostname or "127.0.0.1", engine.port, timeout=5)
        conn.request("GET", "/debug/cancelled")
        value: list[str] = json.loads(conn.getresponse().read())["cancelled"]
        conn.close()
        return value

    deadline = time.monotonic() + 3
    while not cancelled() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(cancelled()) == 1  # движок получил отмену именно этого запроса
    status, _ = client.call("POST", "/v1/chat/completions", chat_body("next"))
    assert status == 200


def test_discovery_file_and_loopback_only(api: tuple[UIServer, EngineManager, Api, Any]) -> None:
    server, _, client, _ = api
    data = json.loads(server.api.discovery_file.read_text(encoding="utf-8"))
    assert data["port"] == client.port and data["host"] == "127.0.0.1"
    assert data["url"] == f"http://127.0.0.1:{client.port}" and data["pid"] == os.getpid()
    assert client.key is not None and client.key not in json.dumps(data)
    assert server.api._server is not None
    assert server.api._server.server_address[0] == "127.0.0.1"
    server.api.close()
    assert not server.api.discovery_file.exists()


def test_headless_service_serves_the_api_without_window(tmp_path: Path) -> None:
    add_models(tmp_path, "model-a")
    env = {
        **os.environ,
        "RAMVI_EXIT_WITH_PARENT": "1",
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("service", "engine", "storage")),
    }
    env.pop("RAMVI_UI_TOKEN", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ramvi_service.app",
            "--headless",
            "--data-dir",
            str(tmp_path),
            "--api-port",
            "0",
            "--model",
            "model-a",
            "--engine-command-json",
            json.dumps(ENGINE),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    try:
        assert process.stdout is not None
        ready = json.loads(process.stdout.readline())
        assert ready["event"] == "ready" and ready["api"].startswith("http://127.0.0.1:")
        port = int(ready["api"].rsplit(":", 1)[1])
        assert Api(port, None).call("GET", "/")[1]["service"] == "ramvi"
        discovery = json.loads((tmp_path / "run" / "api.json").read_text(encoding="utf-8"))
        assert discovery["port"] == port
    finally:
        assert process.stdin is not None
        process.stdin.close()  # «окно закрылось» — сервис выходит и всё за собой убирает
        process.wait(timeout=30)
    assert process.returncode == 0
    assert not (tmp_path / "run" / "api.json").exists()
