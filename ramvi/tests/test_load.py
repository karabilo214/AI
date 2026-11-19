"""8.6: нагрузочный тест. Одновременно: 6 чатов в интерфейсе по 5 сообщений, 4 внешних
клиента API по 6 запросов, опрос состояния каждые 50 мс. Проверяем: ни одного сбоя, ответы
не перепутаны (каждый ответ — эхо своего вопроса), счётчики клиентов точные, история
каждого чата полная и по порядку, база цела, потоки сервиса не утекают."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from ramvi_service.api_server import ApiOptions
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.ui_api import UIServer
from test_api_server import Api, chat_body  # noqa: E402
from test_chat import ENGINE, TOKEN, Client, add_models, text_of  # noqa: E402

CHATS, MESSAGES = 6, 5
CLIENTS, REQUESTS = 4, 6


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


def test_chats_and_api_clients_under_load(svc: tuple[UIServer, EngineManager]) -> None:
    server, _ = svc
    ui = Client(server.port)
    threads_before = threading.active_count()
    chats = [ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"] for _ in range(CHATS)]
    keys = [server.api.keys.create(f"client{i}", {"chat"}) for i in range(CLIENTS)]
    failures: list[str] = []
    lock = threading.Lock()
    stop = threading.Event()

    def fail(text: str) -> None:
        with lock:
            failures.append(text)

    def chat_user(n: int) -> None:
        for m in range(MESSAGES):
            question = f"chat{n} msg{m}"
            events = Client(server.port).events(f"/chats/{chats[n]}/send", {"text": question})
            answer = text_of(events, "text")
            if events[-1]["finish"] != "stop" or question not in answer:
                fail(f"чат {n}/{m}: {events[-1]}")

    def api_client(n: int) -> None:
        api = Api(server.api.port, keys[n][1])
        for r in range(REQUESTS):
            question = f"client{n} req{r}"
            status, body = api.call("POST", "/v1/chat/completions", chat_body(question))
            if status != 200 or question not in str(body):
                fail(f"клиент {n}/{r}: {status} {str(body)[:200]}")

    def poller() -> None:
        while not stop.is_set():
            status, _ = Client(server.port).request("GET", "/status")
            if status != 200:
                fail(f"/status {status}")
            time.sleep(0.05)

    workers = [threading.Thread(target=chat_user, args=(n,)) for n in range(CHATS)]
    workers += [threading.Thread(target=api_client, args=(n,)) for n in range(CLIENTS)]
    watch = threading.Thread(target=poller)
    started = time.monotonic()
    watch.start()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=180)
    elapsed = time.monotonic() - started
    stop.set()
    watch.join(timeout=10)
    assert not any(w.is_alive() for w in workers), "зависание под нагрузкой"
    assert failures == []

    for n, chat_id in enumerate(chats):
        messages = ui.ok("GET", f"/chats/{chat_id}/messages")["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"] * MESSAGES
        for m in range(MESSAGES):
            assert messages[2 * m]["content"] == f"chat{n} msg{m}"
            assert f"chat{n} msg{m}" in messages[2 * m + 1]["content"]
    words = 4  # поддельный движок: [модель] echo: <вопрос> files=0
    for record, _ in keys:
        state = server.api.keys.get(record.id)
        assert state.requests == REQUESTS and state.completion_tokens == REQUESTS * words
        assert state.errors == 0
    db = server.chat.chats_repo.db
    assert db.all("PRAGMA integrity_check")[0][0] == "ok"
    deadline = time.monotonic() + 10
    while threading.active_count() > threads_before + 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert threading.active_count() <= threads_before + 2, "потоки сервиса не освободились"
    total = CHATS * MESSAGES + CLIENTS * REQUESTS
    print(f"нагрузка: {total} запросов за {elapsed:.1f} с")
