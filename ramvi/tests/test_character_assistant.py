"""9.28 (CHAR-DESKTOP-03): Application Assistant — персонаж помогает с выбранным окном."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.delegation import Delegation
from ramvi_service.chat import ChatError
from ramvi_service.context import ContextProvider
from test_agent_desktop import Setup
from test_character_agent import with_character
from test_character_dialogue import Services, active, finish

TARGET = {"application": "notepad.exe", "window_id": "0x1", "title": "заметки.txt — Блокнот"}


class Window(Services):
    """Сервисы с выбранным окном: снимок интерфейса или ошибка доступа/адаптера."""

    def __init__(self, failure: type[Exception] | None = None) -> None:
        super().__init__()
        self.failure = failure
        self.captures: list[dict[str, Any]] = []

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "context.capture":
            self.captures.append(payload)
            if self.failure is not None:
                raise self.failure("no")
            return {
                "snapshot": {
                    "text": "Превет мир",
                    "structure": {
                        "elements": [
                            {"name": "Меню", "role": "MenuBar"},
                            {"name": "", "role": "Pane", "focused": True},  # контейнер без значения
                            {
                                "name": "Текстовый редактор",
                                "role": "Edit",
                                "focused": True,
                                "value": "Превет мир, как дила?",
                            },
                        ]
                    },
                    "source": "ui",
                }
            }
        return super().call(method, payload)


def prompt(svc: Services) -> str:
    return "\n".join(m["content"] for m in svc.requests[-1]["messages"])


def application(svc: Services) -> dict[str, Any]:
    for message in svc.requests[-1]["messages"]:
        if '"application_context"' in message["content"]:
            return dict(json.loads(message["content"]))
    raise AssertionError("no application_context in the request")


def test_selected_text_of_the_chosen_window_goes_to_the_reply_as_data() -> None:
    svc = Window()
    m = active(svc)
    m.dialogue("start", {"text": "Вот здесь ошибка?", "model": "one"})
    finish(m)
    assert '"application_context"' not in prompt(svc) and not svc.captures  # окно не выбрано
    assert m.dialogue("assistant", {"target": TARGET})["assistant"] == TARGET
    m.dialogue("start", {"text": "Вот здесь ошибка?", "model": "one"})
    assert finish(m)["job"]["state"] == "completed"
    context = application(svc)
    assert context["status"] == "ok" and context["trust"] == "untrusted_data"
    assert context["selected_text"] == "Превет мир" and context["title"] == TARGET["title"]
    assert context["focused_text"] == "Превет мир, как дила?"  # поле с курсором — целиком
    assert all("value" not in e for e in context["elements"])  # в списке элементов — только имена
    assert svc.captures[-1] == {"kind": "ui", "application": "notepad.exe", "window_id": "0x1"}
    assert "Превет мир" not in m.dialogue()["history"][-1]["user"]  # в историю — только слова
    m.dialogue("assistant", {"target": None})
    m.dialogue("start", {"text": "А теперь?", "model": "one"})
    finish(m)
    assert '"application_context"' not in svc.requests[-1]["messages"][-1]["content"]
    m.dispose()


@pytest.mark.parametrize(
    ("failure", "status"),
    [(PermissionError, "no_access"), (NotImplementedError, "unavailable"), (TimeoutError, "busy")],
)
def test_without_access_or_adapter_the_character_is_told_what_is_missing(
    failure: type[Exception], status: str
) -> None:
    svc = Window(failure)
    m = active(svc)
    m.dialogue("assistant", {"target": TARGET})
    m.dialogue("start", {"text": "Что тут не так?", "model": "one"})
    assert finish(m)["job"]["state"] == "completed"  # обычный разговор продолжается
    context = application(svc)
    assert context["status"] == status and "selected_text" not in context
    m.dispose()


def test_bad_target_is_refused() -> None:
    m = active(Window())
    for target in ({"application": "x"}, {**TARGET, "title": ""}, "notepad"):
        assert "error" in m.dialogue("assistant", {"target": target})
    m.dispose()


def test_handed_over_task_carries_the_window_context_but_history_keeps_the_owner_goal() -> None:
    sent: list[str] = []
    finished = threading.Event()

    class Gateway:
        def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
            if method == "agent.create":
                sent.append(payload["goal"])
                return {"task_id": "t1"}
            if method == "agent.events":
                finished.set()
                return {
                    "events": [{"type": "agent.done", "status": "done", "result": "ok"}],
                    "cursor": 1,
                }
            if method == "agent.status":
                return {"state": "done", "result": {"result": "ok"}}
            return {}

    delegation = Delegation(Gateway(), lambda task: None, poll=0.01)
    delegation.bind("b1")
    view = delegation.start("Исправь опечатку", '[RAMVI: context] {"selected_text": "Превет"}')
    assert sent and sent[0].startswith("Исправь опечатку\n\n[RAMVI: context]")
    assert view["task"]["goal"] == "Исправь опечатку"


class Adapter:
    def __init__(self) -> None:
        self.application = "notepad.exe"

    def current_target(self, *, describe: bool = False) -> dict[str, Any]:
        return {"application": self.application, "window_id": "0x1", "title": "Блокнот"}

    def capture(self, application: str, window: str, kind: str, cancel: Any) -> dict[str, Any]:
        return {"text": "", "structure": {"elements": []}}


def test_owner_grants_the_window_to_character_and_a_new_window_revokes_it(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        adapter = Adapter()
        setup.host.context = ContextProvider(setup.host.authority, setup.host.grants, adapter)
        with_character(setup)
        owner = setup.host._entry("character").owner
        assert owner is not None
        with pytest.raises(ChatError):  # окно ещё не выбрано
            setup.route(
                "POST",
                "target/module",
                module="character",
                ui=True,
                screen=False,
                egress=False,
                control=False,
            )
        setup.route("POST", "target")
        given = setup.route(
            "POST",
            "target/module",
            module="character",
            ui=True,
            screen=False,
            egress=True,
            control=True,
        )
        assert given["target"]["application"] == "notepad.exe"
        setup.host.grants.require(owner, "context.ui", "notepad.exe")
        setup.host.grants.require(owner, "egress.text", "notepad.exe")
        setup.host.grants.require(owner, "agent.control", "notepad.exe")  # поручения — в окне
        with pytest.raises(PermissionError):
            setup.host.grants.require(owner, "context.screen", "notepad.exe")
        adapter.application = "calc.exe"
        setup.route("POST", "target")  # другое окно — доступ к прежнему отозван сразу
        with pytest.raises(PermissionError):
            setup.host.grants.require(owner, "context.ui", "notepad.exe")
        setup.route(
            "POST",
            "target/module",
            module="character",
            ui=True,
            screen=False,
            egress=False,
            control=False,
        )
        setup.host.grants.require(owner, "context.ui", "calc.exe")
        setup.route(
            "POST",
            "target/module",
            module="character",
            ui=False,
            screen=False,
            egress=False,
            control=False,
        )
        with pytest.raises(PermissionError):
            setup.host.grants.require(owner, "context.ui", "calc.exe")
    finally:
        setup.close()


class CloudWindow(Window):
    """Облачная модель: текст окна уходит ей только с разрешением на отправку."""

    def __init__(self, egress: bool) -> None:
        super().__init__()
        self.egress = egress

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "llm.capabilities":
            return {"available": True, "capabilities": {"streaming": True, "external": True}}
        result = super().call(method, payload)
        if method == "context.capture":
            result["egress_text"] = self.egress
        return result


@pytest.mark.parametrize("egress", [False, True])
def test_cloud_model_gets_window_text_only_with_the_send_permission(egress: bool) -> None:
    svc = CloudWindow(egress)
    m = active(svc)
    m.dialogue("assistant", {"target": TARGET})
    m.dialogue("start", {"text": "Что тут не так?", "model": "one"})
    assert finish(m)["job"]["state"] == "completed"  # ответ доходит в обоих случаях
    context = application(svc)
    if egress:
        assert context["status"] == "ok" and context["selected_text"] == "Превет мир"
    else:
        assert context["status"] == "cloud_not_allowed" and "selected_text" not in context
        assert "Превет мир" not in prompt(svc)
    m.dispose()
