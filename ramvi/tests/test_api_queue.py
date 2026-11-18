"""7.3 (API-03): очередь клиентов — чередование, лимиты, клиенты не смешивают ответы."""

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.api_server import ApiOptions
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


class Holder:
    """Чужая генерация (например, чат) занимает модель, пока не отпустят."""

    def __init__(self, server: UIServer) -> None:
        llm = server.api.llm
        self.owner = llm.authority.issue("chat", "local_user", "busy")
        llm.retain(self.owner)
        self.slot = llm.exclusive(self.owner)
        self.slot.__enter__()

    def release(self) -> None:
        self.slot.__exit__(None, None, None)


def _send(api: Api, text: str, results: list[tuple[str, int, str, float]]) -> threading.Thread:
    def run() -> None:
        status, body = api.call("POST", "/v1/chat/completions", chat_body(text))
        results.append(
            (text, status, body if isinstance(body, str) else str(body), time.monotonic())
        )

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _wait_pending(server: UIServer, client_id: str, count: int) -> None:
    deadline = time.monotonic() + 5
    while server.api.gate.pending(client_id) < count and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.api.gate.pending(client_id) == count


def test_clients_take_turns(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    a_rec, a_key = server.api.keys.create("busy ide", {"chat"})
    b_rec, b_key = server.api.keys.create("other ide", {"chat"})
    holder = Holder(server)
    results: list[tuple[str, int, str, float]] = []
    threads = [_send(Api(server.api.port, a_key), f"A{i}", results) for i in range(4)]
    _wait_pending(server, a_rec.id, 4)
    threads.append(_send(Api(server.api.port, b_key), "B0", results))
    _wait_pending(server, b_rec.id, 1)
    holder.release()
    for thread in threads:
        thread.join(timeout=30)
    order = [text for text, *_ in sorted(results, key=lambda r: r[3])]
    assert all(status == 200 for _, status, _, _ in results)
    # B прислал один запрос после четырёх запросов A — и ждёт только один из них.
    assert order.index("B0") == 1, order


def test_per_client_queue_limit(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    server.api.options = replace(server.api.options, per_client_queue=2)
    rec, key = server.api.keys.create("ide", {"chat"})
    holder = Holder(server)
    results: list[tuple[str, int, str, float]] = []
    threads = [_send(Api(server.api.port, key), f"q{i}", results) for i in range(2)]
    _wait_pending(server, rec.id, 2)
    status, body = Api(server.api.port, key).call("POST", "/v1/chat/completions", chat_body("x"))
    assert status == 429 and body["error"]["type"] == "api.client_queue_full"
    holder.release()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(status for _, status, _, _ in results) == [200, 200]


def test_max_clients_limit(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    server.api.options = replace(server.api.options, max_clients=1)
    a_rec, a_key = server.api.keys.create("first", {"chat"})
    _, b_key = server.api.keys.create("second", {"chat"})
    holder = Holder(server)
    results: list[tuple[str, int, str, float]] = []
    thread = _send(Api(server.api.port, a_key), "a", results)
    _wait_pending(server, a_rec.id, 1)
    status, body = Api(server.api.port, b_key).call("POST", "/v1/chat/completions", chat_body("b"))
    assert status == 503 and body["error"]["type"] == "api.too_many_clients"
    holder.release()
    thread.join(timeout=30)
    assert results[0][1] == 200
    assert server.api.gate.clients() == 0  # ворота освободились


def test_clients_do_not_mix_answers_or_counters(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    a_rec, a_key = server.api.keys.create("a", {"chat"})
    b_rec, b_key = server.api.keys.create("b", {"chat"})
    a, b = Api(server.api.port, a_key), Api(server.api.port, b_key)
    alone = {
        text: a.call("POST", "/v1/chat/completions", chat_body(text))[1]
        for text in ("alpha question", "bravo question")
    }
    before = (server.api.keys.get(a_rec.id), server.api.keys.get(b_rec.id))
    results: list[tuple[str, int, str, float]] = []
    threads = []
    for _ in range(3):
        threads.append(_send(a, "alpha question", results))
        threads.append(_send(b, "bravo question", results))
    for thread in threads:
        thread.join(timeout=60)
    for text, status, body, _ in results:
        assert status == 200
        assert body == alone[text]  # тот же ответ, что в одиночку: токены не смешались
        other = "bravo" if text.startswith("alpha") else "alpha"
        assert other not in body
    after = (server.api.keys.get(a_rec.id), server.api.keys.get(b_rec.id))
    words = 4  # поддельный движок: [модель] echo: <вопрос> files=0
    assert after[0].completion_tokens - before[0].completion_tokens == 3 * words
    assert after[1].completion_tokens - before[1].completion_tokens == 3 * words
    assert after[1].requests == 3 and after[0].requests == before[0].requests + 3


def test_queue_status_is_reported(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    queue: dict[str, Any] = Client(server.port).ok("GET", "/connections")["queue"]
    assert queue == {
        "max_concurrent": 1,
        "per_client_queue": 8,
        "max_clients": 16,
        "active_clients": 0,
    }


def test_many_simultaneous_connections_are_not_dropped(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    _, key = server.api.keys.create("ide", {"models"})
    statuses: list[int] = []
    lock = threading.Lock()

    def ask() -> None:
        status = Api(server.api.port, key).call("GET", "/v1/models")[0]
        with lock:
            statuses.append(status)

    threads = [threading.Thread(target=ask) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert statuses == [200] * 40
