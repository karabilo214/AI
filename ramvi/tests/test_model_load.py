"""3.17 (UI-MODEL-LOAD-01): загрузка при выборе, модель держится, «Выгрузить модель».

Настоящий процесс поддельного движка: процесс модели виден по pid, выгрузка — по
тому, что процесс завершён.
"""

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.supervisor import RestartPolicy
from ramvi_service.ui_api import UIServer
from test_chat import ENGINE, TOKEN, Client, add_models, text_of
from test_supervisor import alive


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager, Client]]:
    add_models(tmp_path, "model-a", "model-b", "model-oom")
    fast = lambda: RestartPolicy(max_restarts=1, base_delay=0.05, max_delay=0.05)  # noqa: E731
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE, restart_policy=fast)
    starts: list[str] = []
    make = engines._make_engine

    def counted(model: Any, draft: Any = None) -> Any:
        starts.append(model.id)
        return make(model, draft)

    engines._make_engine = counted
    engines.starts = starts  # type: ignore[attr-defined]
    server.start()
    try:
        yield server, engines, Client(server.port)
    finally:
        server.close()
        engines.stop()


def wait_engine(client: Client, timeout: float = 20) -> dict[str, Any]:
    """Ждать, пока загрузка закончится (готово или ошибка), как это делает оболочка."""
    deadline = time.monotonic() + timeout
    while True:
        engine: dict[str, Any] = client.ok("GET", "/status")["engine"]
        if engine["loading"] is None and engine["state"] != "starting":
            return engine
        assert time.monotonic() < deadline, engine
        time.sleep(0.05)


def pid(engines: EngineManager) -> int:
    assert engines._supervisor is not None
    return int(engines._supervisor.status()["target"]["pid"])


def test_model_loads_on_choice_without_a_message(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    answer = client.ok("POST", "/engine/load", {"model_id": "model-a"})
    assert answer["started"] is True and answer["engine"]["loading"] == "model-a"
    engine = wait_engine(client)
    assert engine["model_id"] == "model-a" and engine["state"] == "ready"
    assert engines.starts == ["model-a"]  # type: ignore[attr-defined]
    # Повторный выбор той же модели ничего не перезагружает.
    assert client.ok("POST", "/engine/load", {"model_id": "model-a"})["started"] is False
    # Первое сообщение уже без загрузки.
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    events = client.events(f"/chats/{chat_id}/send", {"text": "привет"})
    assert not [e for e in events if e["type"] == "status"]
    assert text_of(events, "text").startswith("[model-a.gguf] echo: привет")
    assert engines.starts == ["model-a"]  # type: ignore[attr-defined]


def test_model_stays_loaded_after_answer_pause_and_chat_switch(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    client.ok("POST", "/engine/load", {"model_id": "model-a"})
    wait_engine(client)
    first = pid(engines)
    a = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.events(f"/chats/{a}/send", {"text": "раз"})
    time.sleep(2)  # пауза: выгрузки по простою нет
    b = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.ok("GET", f"/chats/{a}/messages")
    client.events(f"/chats/{b}/send", {"text": "два"})
    assert pid(engines) == first and alive(first)
    assert engines.starts == ["model-a"]  # type: ignore[attr-defined]
    assert client.ok("GET", "/status")["engine"]["model_id"] == "model-a"


def test_choosing_another_model_unloads_previous(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    client.ok("POST", "/engine/load", {"model_id": "model-a"})
    wait_engine(client)
    first = pid(engines)
    client.ok("POST", "/engine/load", {"model_id": "model-b"})
    engine = wait_engine(client)
    assert engine["model_id"] == "model-b" and engine["state"] == "ready"
    assert not alive(first)
    assert engines.starts == ["model-a", "model-b"]  # type: ignore[attr-defined]


def test_unload_button_frees_model_and_next_message_loads_again(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    client.ok("POST", "/engine/load", {"model_id": "model-a"})
    wait_engine(client)
    first = pid(engines)
    answer = client.ok("POST", "/engine/unload")
    assert answer["unloaded"] is True
    assert answer["engine"]["model_id"] is None and answer["engine"]["state"] == "stopped"
    assert not alive(first)
    assert client.ok("POST", "/engine/unload")["unloaded"] is False  # уже выгружена
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    events = client.events(f"/chats/{chat_id}/send", {"text": "снова"})
    assert {"type": "status", "state": "loading", "model_id": "model-a"} in events
    assert events[-1]["finish"] == "stop"


def test_unload_during_answer_stops_it_first(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    seen: list[dict[str, Any]] = []
    started = threading.Event()

    def read() -> None:
        for event in client.stream(f"/chats/{chat_id}/send", {"text": "LONG"}):
            seen.append(event)
            if event["type"] == "text":
                started.set()

    reader = threading.Thread(target=read)
    reader.start()
    assert started.wait(20)
    first = pid(engines)
    assert client.ok("POST", "/engine/unload")["unloaded"] is True
    reader.join(20)
    assert seen[-1]["type"] == "done" and seen[-1]["finish"] == "cancelled"
    assert not alive(first) and engines.model_id is None


def test_choice_during_answer_waits_for_its_end(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    seen: list[dict[str, Any]] = []
    started = threading.Event()

    def read() -> None:
        for event in client.stream(f"/chats/{chat_id}/send", {"text": "LONG"}):
            seen.append(event)
            if event["type"] == "text":
                started.set()

    reader = threading.Thread(target=read)
    reader.start()
    assert started.wait(20)
    client.ok("POST", "/engine/load", {"model_id": "model-b"})
    assert client.ok("GET", "/status")["engine"]["loading"] == "model-b"  # ждёт очереди
    reader.join(30)
    assert seen[-1]["finish"] == "stop"  # ответ не оборван сменой модели
    assert len(text_of(seen, "text").split()) >= 400
    engine = wait_engine(client)
    assert engine["model_id"] == "model-b" and engines.starts == ["model-a", "model-b"]  # type: ignore[attr-defined]


def test_message_sent_during_load_waits_and_does_not_load_twice(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, engines, client = service
    chat_id = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    client.ok("POST", "/engine/load", {"model_id": "model-a"})
    events = client.events(f"/chats/{chat_id}/send", {"text": "сразу"})
    assert events[-1]["finish"] == "stop"
    assert text_of(events, "text").startswith("[model-a.gguf] echo: сразу")
    wait_engine(client)
    assert engines.starts == ["model-a"]  # type: ignore[attr-defined]


def test_load_failure_and_unknown_model_are_reported(
    service: tuple[UIServer, EngineManager, Client],
) -> None:
    _, _, client = service
    status, error = client.request("POST", "/engine/load", {"model_id": "нет"})
    assert status == 404 and error["error"]["code"] == "chat.model_not_found"
    status, error = client.request("POST", "/engine/load", {})
    assert status == 400 and error["error"]["code"] == "api.bad_request"
    client.ok("POST", "/engine/load", {"model_id": "model-oom"})
    engine = wait_engine(client)
    assert engine["problem"]["code"] == "engine.no_memory"
    # Выгрузка снимает прежнюю причину.
    client.ok("POST", "/engine/unload")
    assert client.ok("GET", "/status")["engine"]["problem"] is None
