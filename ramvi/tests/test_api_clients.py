"""7.2 (API-02): клиенты внешнего API — счётчики, отзыв, ключ показывается один раз."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.api_server import ApiOptions, parse_usage
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.ui_api import UIServer
from test_api_server import Api, chat_body  # noqa: E402
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager]]:
    add_models(tmp_path, "model-a")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    engines.ensure("model-a")
    try:
        yield server, engines
    finally:
        server.close()
        engines.stop()


def test_parse_usage_in_all_formats() -> None:
    openai_stream = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert parse_usage(b"", openai_stream) == (12, 5)
    anthropic = b'{"type":"message","usage":{"input_tokens":7,"output_tokens":3}}'
    assert parse_usage(anthropic, anthropic) == (7, 3)
    anthropic_stream_head = (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"usage":{"input_tokens":20,"output_tokens":1}}}\n\n'
    )
    anthropic_stream_tail = (
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
    )
    assert parse_usage(anthropic_stream_head, anthropic_stream_tail) == (20, 9)
    ollama = b'{"message":{"content":"x"},"done":false}\n{"done":true,"prompt_eval_count":4,"eval_count":6}\n'
    assert parse_usage(b"", ollama) == (4, 6)
    pretty = b'{\n  "usage": {\n    "prompt_tokens": 2,\n    "completion_tokens": 1\n  }\n}'
    assert parse_usage(pretty, pretty) == (2, 1)
    assert parse_usage(b"", b"not json") == (0, 0)
    # Ollama от движка RAMVI: все токены ответа — total_output_tokens, а не eval_count.
    ramvi_ollama = b'{"done":true,"prompt_eval_count":30,"eval_count":4,"total_output_tokens":5}'
    assert parse_usage(b"", ramvi_ollama) == (30, 5)
    # Служебные метрики движка в ответе не влияют на счёт.
    with_metrics = (
        b'{"usage":{"prompt_tokens":3,"completion_tokens":2},'
        b'"ramvi_metrics":{"completion_tokens":999,"prompt_tokens":888}}'
    )
    assert parse_usage(with_metrics, with_metrics) == (3, 2)
    # Поток OpenAI без include_usage: счёт — из метрик этой генерации.
    stream_no_usage = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"ramvi_metrics":{"generation_id":"g1","input_tokens":30,"output_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert parse_usage(b"", stream_no_usage) == (30, 5)


def test_counters_are_kept_per_client(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    a_record, a_key = server.api.keys.create("ide-a", {"chat", "models"})
    b_record, b_key = server.api.keys.create("ide-b", {"chat"})
    a, b = Api(server.api.port, a_key), Api(server.api.port, b_key)
    assert a.call("POST", "/v1/chat/completions", chat_body("one"))[0] == 200
    assert a.call("POST", "/api/chat", chat_body("two"))[0] == 200
    assert b.call("POST", "/v1/messages", chat_body("three"))[0] == 200
    assert a.call("POST", "/v1/chat/completions", chat_body("bad", temperature=0.7))[0] == 400
    first, second = server.api.keys.get(a_record.id), server.api.keys.get(b_record.id)
    # Поддельный движок: OpenAI-поток — 4 слова ответа; /api/chat и /v1/messages — 7 и 3.
    assert (first.requests, first.prompt_tokens, first.completion_tokens, first.errors) == (
        3,
        7,
        4 + 3,
        1,
    )
    assert (second.requests, second.prompt_tokens, second.completion_tokens, second.errors) == (
        1,
        7,
        3,
        0,
    )
    assert first.busy_seconds > 0 and second.busy_seconds > 0
    assert first.last_seen is not None and first.state == "active"


def _long_stream(port: int, key: str) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=15)
    body = json.dumps(chat_body("LONG answer")).encode()
    sock.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Authorization: Bearer {key}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    return sock


def _read_all(sock: socket.socket) -> bytes:
    data = b""
    while chunk := sock.recv(65536):
        data += chunk
    sock.close()
    return data


def test_revoking_one_client_stops_it_and_not_the_other(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    a_record, a_key = server.api.keys.create("ide-a", {"chat"})
    b_record, b_key = server.api.keys.create("ide-b", {"chat"})
    a_sock = _long_stream(server.api.port, a_key)
    time.sleep(0.3)  # A генерирует длинный ответ
    b_result: dict[str, Any] = {}
    b_thread = threading.Thread(
        target=lambda: b_result.update(
            answer=Api(server.api.port, b_key).call("POST", "/v1/chat/completions", chat_body("B"))
        )
    )
    b_thread.start()  # B ждёт в общей очереди
    time.sleep(0.2)
    assert server.api.running(a_record.id) == 1 and server.api.running(b_record.id) == 1
    stopped = server.api.revoke(a_record.id)
    assert stopped == 1
    a_data = _read_all(a_sock)
    assert a_data.count(b'"content": " word"') < 300  # ответ A оборван, а не досчитан до конца
    b_thread.join(timeout=20)
    status, text = b_result["answer"]
    assert status == 200 and '"content": " B"' in text and "[DONE]" in text
    assert (
        Api(server.api.port, a_key).call("POST", "/v1/chat/completions", chat_body("x"))[0] == 401
    )
    assert (
        Api(server.api.port, b_key).call("POST", "/v1/chat/completions", chat_body("y"))[0] == 200
    )
    assert server.api.keys.get(b_record.id).errors == 0


def test_revoke_without_stopping_lets_the_current_answer_finish(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    record, key = server.api.keys.create("ide", {"chat"})
    sock = _long_stream(server.api.port, key)
    time.sleep(0.3)
    assert server.api.revoke(record.id, cancel_running=False) == 0
    data = _read_all(sock)
    assert data.count(b'"content": " word"') == 399 and b"[DONE]" in data
    assert Api(server.api.port, key).call("POST", "/v1/chat/completions", chat_body("x"))[0] == 401


def test_revoke_while_waiting_in_queue(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    record, key = server.api.keys.create("ide", {"chat"})
    llm = server.api.llm
    other = llm.authority.issue("chat", "local_user", "busy")
    llm.retain(other)
    holding = llm.exclusive(other)
    holding.__enter__()
    result: dict[str, Any] = {}
    waiter = threading.Thread(
        target=lambda: result.update(
            answer=Api(server.api.port, key).call("POST", "/v1/chat/completions", chat_body("w"))
        )
    )
    waiter.start()
    time.sleep(0.3)
    started = time.monotonic()
    assert server.api.revoke(record.id) == 1
    waiter.join(timeout=10)
    holding.__exit__(None, None, None)
    assert result["answer"][0] == 401 and time.monotonic() - started < 5


def test_key_is_shown_only_once(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    ui = Client(server.port)
    created = ui.ok("POST", "/connections/keys", {"name": "VS Code", "permissions": ["chat"]})
    key = created["key"]
    assert key.startswith("ramvi-") and created["client"]["key_hint"] == f"ramvi-…{key[-4:]}"
    listing = ui.ok("GET", "/connections")
    assert listing["api"]["running"] and listing["api"]["url"].startswith("http://127.0.0.1:")
    (client,) = listing["clients"]
    assert client["name"] == "VS Code" and client["state"] == "active" and client["running"] == 0
    assert key not in json.dumps(listing)
    status, error = ui.request("POST", "/connections/keys", {"name": "", "permissions": ["chat"]})
    assert status == 400 and error["error"]["code"] == "client_keys.no_name"
    revoked = ui.ok("POST", f"/connections/keys/{client['id']}/revoke", {"cancel_running": True})
    assert revoked == {"revoked": client["id"], "stopped": 0}
    assert ui.ok("GET", "/connections")["clients"][0]["state"] == "revoked"
    assert ui.request("POST", "/connections/keys/000000000000/revoke", {})[0] == 404
