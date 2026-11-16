"""3.11 (CORE-UI-01): реестр разделов, ленивое подключение, независимость разделов."""

import json
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Manifest
from ramvi_service.modules.host import ModuleHost
from ramvi_service.sections import SectionRegistry, section_of
from test_chat import ENGINE, TOKEN, Client, add_models

FIXTURE = "ramvi.tests.module_fixture:create"


def manifest(module_id: str, view: str = "agent", **section: Any) -> Manifest:
    ui = {
        "section": {
            "view": view,
            "title_key": f"section.{view}",
            "icon": view,
            "order": 20,
            **section,
        }
    }
    return Manifest(module_id, "1.0.0", FIXTURE, ui=ui)


@pytest.fixture
def registry(tmp_path: Path):  # type: ignore[no-untyped-def]
    host = ModuleHost(tmp_path)
    try:
        yield SectionRegistry(host)
    finally:
        host.close()


def by_id(registry: SectionRegistry) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in registry.list()}


def test_only_chat_without_modules_character_may_be_absent(
    registry: SectionRegistry, tmp_path: Path
) -> None:
    registry.discover(tmp_path / "no-modules-here")

    assert [s["id"] for s in registry.list()] == ["chat"]
    assert by_id(registry)["chat"]["state"] == "ready"
    assert registry.open("chat")["view"] == "chat"


def test_module_is_loaded_lazily_on_first_open(registry: SectionRegistry) -> None:
    registry.install(manifest("agent"))

    agent = by_id(registry)["agent"]
    assert (agent["state"], agent["view"], agent["module_id"]) == ("available", "agent", "agent")
    assert registry.host.status("agent")["agent"]["worker"] is None  # процесс не запущен

    opened = registry.open("agent")

    assert opened["state"] == "active" and opened["error"] is None
    assert registry.host.status("agent")["agent"]["worker"]["state"] == "running"


def test_switching_sections_keeps_module_state_and_process(registry: SectionRegistry) -> None:
    registry.install(manifest("agent"))
    registry.install(manifest("character", view="character", order=30))
    registry.open("agent")
    pid = registry.host.status("agent")["agent"]["worker"]["pid"]
    value = registry.host.invoke("agent", "value")["value"]

    for section in ("chat", "character", "chat", "agent", "agent"):
        registry.open(section)

    assert registry.host.status("agent")["agent"]["worker"]["pid"] == pid  # не перезапускался
    assert registry.host.invoke("agent", "value")["value"] == value  # activate не повторялся
    assert [s["id"] for s in registry.list()] == ["chat", "agent", "character"]


def test_broken_module_is_a_section_error_not_a_crash(registry: SectionRegistry) -> None:
    registry.install(
        Manifest(
            "character",
            "1.0.0",
            "not_installed_character.factory:create",
            ui={"section": {"view": "character", "title_key": "section.character"}},
        )
    )

    opened = registry.open("character")

    assert opened["state"] == "error" and opened["error"]
    assert registry.open("chat")["state"] == "ready"  # чат работает независимо


def test_discover_installed_modules_and_report_broken(
    registry: SectionRegistry, tmp_path: Path
) -> None:
    modules = tmp_path / "modules"
    (modules / "agent").mkdir(parents=True)
    (modules / "agent" / "manifest.json").write_text(
        json.dumps(
            {
                "module_id": "agent",
                "version": "1.0.0",
                "entrypoint": FIXTURE,
                "permissions": [],
                "ui": {
                    "section": {
                        "view": "agent",
                        "title_key": "section.agent",
                        "icon": "agent",
                        "order": 20,
                    }
                },
            }
        )
    )
    (modules / "broken").mkdir()
    (modules / "broken" / "manifest.json").write_text("{not json")
    (modules / "plain").mkdir()  # модуль без раздела интерфейса
    (modules / "plain" / "manifest.json").write_text(
        json.dumps({"module_id": "plain", "version": "1.0.0", "entrypoint": FIXTURE})
    )

    registry.discover(modules)

    assert [s["id"] for s in registry.list()] == ["chat", "agent"]
    assert "plain" in registry.host.status()  # установлен, но раздела нет
    assert [Path(p["path"]).parent.name for p in registry.problems] == ["broken"]


@pytest.mark.parametrize(
    "section",
    [
        {"view": "chat", "title_key": "section.x"},
        {"view": "terminal", "title_key": "section.x"},
        {"view": "agent", "title_key": "Агент"},
        {"view": "agent", "title_key": "section.agent", "icon": "Bad Icon"},
        {"view": "agent", "title_key": "section.agent", "order": True},
    ],
)
def test_invalid_section_descriptions_are_rejected(section: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        section_of(Manifest("m", "1.0.0", FIXTURE, ui={"section": section}))


def test_unknown_section_and_duplicate(registry: SectionRegistry) -> None:
    registry.install(manifest("agent"))
    with pytest.raises(ChatError) as error:
        registry.open("nothing")
    assert error.value.code == "section.not_found"
    with pytest.raises(ValueError):
        registry.install(manifest("agent"))


def test_sections_over_http_and_chat_unaffected(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "modules" / "agent").mkdir(parents=True)
    (app / "modules" / "agent" / "manifest.json").write_text(
        json.dumps(
            {
                "module_id": "agent",
                "version": "1.0.0",
                "entrypoint": FIXTURE,
                "ui": {
                    "section": {
                        "view": "agent",
                        "title_key": "section.agent",
                        "icon": "agent",
                        "order": 20,
                    }
                },
            }
        )
    )
    add_models(app / "data", "model-a")
    server, engines = build(app / "data", TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        client = Client(server.port)
        listed = client.ok("GET", "/sections")["sections"]
        assert [(s["id"], s["state"]) for s in listed] == [
            ("chat", "ready"),
            ("agent", "available"),
        ]
        chat = client.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
        assert client.ok("POST", "/sections/agent/open")["section"]["state"] == "active"
        assert client.events(f"/chats/{chat}/send", {"text": "hi"})[-1]["finish"] == "stop"
        status, error = client.request("POST", "/sections/nope/open")
        assert status == 400 and error["error"]["code"] == "section.not_found"
    finally:
        server.close()
        assert server.sections is not None
        server.sections.host.close()
        engines.stop()
