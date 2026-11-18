"""7.6 (CORE-API-01): версии возможностей и событий модулей; подмена module_id не работает;
включённый модуль (как Character) не меняет ответы Chat/Engine."""

import http.client
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from ramvi_service.api_server import ApiOptions, strip_reserved
from ramvi_service.app import build
from ramvi_service.contracts.modules import Manifest
from ramvi_service.engines import EngineManager
from ramvi_service.events import EventSpec
from ramvi_service.ui_api import UIServer
from test_api_server import Api, chat_body  # noqa: E402
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402

FIXTURE = "ramvi.tests.module_fixture:create"


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager]]:
    add_models(tmp_path, "model-a", "model-b")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    engines.ensure("model-a")
    try:
        yield server, engines
    finally:
        server.close()
        assert server.sections is not None
        server.sections.host.close()
        engines.stop()


def _register_character(server: UIServer, *, approve_all: bool = False) -> Any:
    assert server.sections is not None
    host = server.sections.host
    host.register(
        Manifest(
            "character",
            "1.2.3",
            FIXTURE,
            capabilities=(),
            permissions=("llm.text", "context.screen"),
        ),
        approved_permissions=("llm.text", "context.screen") if approve_all else ("llm.text",),
    )
    host.events.register_schema(
        "character", "emotion", EventSpec({"name": str}, {"intensity": float})
    )
    return host


def test_capabilities_are_versioned_and_host_defines_permissions(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    _register_character(server)
    caps = Client(server.port).ok("GET", "/capabilities")
    assert caps["schema"] == "ramvi.capabilities/1" and caps["module_api"] == "1.0"
    assert caps["events"] == {"version": 1, "type": "module.<module_id>.<name>"}
    assert caps["core"]["chat"] and caps["core"]["agent"]
    assert caps["core"]["external_api"]["formats"] == ["openai", "anthropic", "ollama"]
    (module,) = [m for m in caps["modules"] if m["module_id"] == "character"]
    assert module["version"] == "1.2.3" and module["api_version"] == "1.0"
    assert module["permissions_requested"] == ["llm.text", "context.screen"]
    assert module["permissions_approved"] == ["llm.text"]  # решает Host, не манифест
    assert module["events"] == [
        {
            "module_id": "character",
            "name": "emotion",
            "type": "module.character.emotion",
            "version": 1,
            "required": {"name": "str"},
            "optional": {"intensity": "float"},
        }
    ]


def test_external_clients_see_a_reduced_description(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    _register_character(server)
    _, key = server.api.keys.create("ide", {"models"})
    status, caps = Api(server.api.port, key).call("GET", "/ramvi/v1/capabilities")
    assert status == 200 and caps["schema"] == "ramvi.capabilities/1"
    (module,) = [m for m in caps["modules"] if m["module_id"] == "character"]
    assert set(module) == {"module_id", "version", "api_version", "state"}
    assert Api(server.api.port, None).call("GET", "/ramvi/v1/capabilities")[0] == 401


def test_reserved_fields_are_dropped() -> None:
    body = {
        "messages": [],
        "module_id": "character",
        "owner": "x",
        "session_id": "s",
        "ramvi_model_id": "model-b",
        "RAMVI_MIN_CONTEXT": 1,
        "temperature": 0,
    }
    assert sorted(strip_reserved(body)) == [
        "RAMVI_MIN_CONTEXT",
        "module_id",
        "owner",
        "ramvi_model_id",
        "session_id",
    ]
    assert body == {"messages": [], "temperature": 0}


def _engine_body(engines: EngineManager) -> dict[str, Any]:
    target = urlsplit(engines.provider().base_url)
    conn = http.client.HTTPConnection(target.hostname or "127.0.0.1", target.port, timeout=5)
    conn.request("GET", "/debug/last_body")
    body: dict[str, Any] = json.loads(conn.getresponse().read())["body"]
    conn.close()
    return body


def test_spoofed_module_id_gives_nothing(svc: tuple[UIServer, EngineManager]) -> None:
    server, engines = svc
    _register_character(server)
    record, key = server.api.keys.create("ide", {"chat"})
    spoofed = chat_body(
        "hello",
        module_id="character",
        owner="character",
        session_id="other",
        ramvi_model_id="model-b",
        ramvi_min_context=999_999,
    )
    status, text = Api(server.api.port, key).call("POST", "/v1/chat/completions", spoofed)
    assert status == 200 and '"content": " hello"' in text
    assert engines.model_id == "model-a"  # модель не переключилась
    reached = _engine_body(engines)
    assert not {"module_id", "owner", "session_id", "ramvi_model_id", "ramvi_min_context"} & set(
        reached
    )
    assert server.api.keys.get(record.id).requests == 1  # учтено на свой ключ
    assert server.sections is not None
    character = server.sections.host.describe()[0]
    assert character["permissions_approved"] == ["llm.text"]  # права модуля не изменились


def _answers(server: UIServer, key: str) -> dict[str, Any]:
    api = Api(server.api.port, key)
    ui = Client(server.port)
    chat = ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]
    events = ui.events(f"/chats/{chat['id']}/send", {"text": "same question"})
    text = "".join(e.get("text", "") for e in events if e["type"] == "text")
    return {
        "openai": api.call("POST", "/v1/chat/completions", chat_body("same question"))[1],
        "anthropic": api.call("POST", "/v1/messages", chat_body("same question"))[1],
        "ollama": api.call("POST", "/api/chat", chat_body("same question"))[1],
        "models": api.call("GET", "/v1/models")[1],
        "chat": text,
    }


def test_active_character_module_does_not_change_answers(
    svc: tuple[UIServer, EngineManager],
) -> None:
    server, _ = svc
    _, key = server.api.keys.create("ide", {"chat", "models"})
    before = _answers(server, key)
    host = _register_character(server, approve_all=True)  # без одобрения Host не запустит
    host.activate("character")
    assert host.status("character")["character"]["state"] == "active"
    after = _answers(server, key)
    assert after == before
