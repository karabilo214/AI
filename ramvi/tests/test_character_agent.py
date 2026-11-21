"""9.21 (CHAR-21): Character поручает явную задачу Agent через публичный Agent Gateway."""

import ast
import json
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.delegation import Delegation
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Manifest
from ramvi_service.module_data import load_migrations
from ramvi_storage.repositories import AgentRuns
from test_agent_desktop import Setup
from test_character_dialogue import Services, active, finish

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "modules" / "character" / "manifest.json").read_text(encoding="utf-8")
)


def with_character(setup: Setup) -> None:
    setup.host.register(
        Manifest(
            "character",
            MANIFEST["version"],
            MANIFEST["entrypoint"],
            permissions=("data", "llm", "agent"),
        ),
        approved_permissions=("data", "llm", "agent"),
        sql_migrations=load_migrations(ROOT / "modules" / "character" / "migrations"),
    )
    setup.host.activate("character")


def character(setup: Setup, operation: str, **arguments: Any) -> dict[str, Any]:
    return dict(
        setup.host.invoke("character", "dialogue", {"operation": operation, "arguments": arguments})
    )


def delegation(setup: Setup, predicate: Any) -> dict[str, Any]:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        view = character(setup, "delegation")["delegation"]
        if predicate(view):
            return dict(view)
        time.sleep(0.02)
    raise AssertionError("delegation state was not reached")


def enable(setup: Setup, write: bool = True) -> str:
    from ramvi_service.agent.desktop import defaults

    policy = defaults()
    policy["permissions"]["write"] = write
    setup.route("POST", "settings", **policy)
    binding = setup.route(
        "POST",
        "delegate",
        module="character",
        enabled=True,
        workspace=str(setup.project),
        model_id="test-model",
    )["binding_id"]
    assert character(setup, "agent_bind", binding_id=binding)["delegation"]["enabled"]
    return str(binding)


@pytest.mark.parametrize("answer", ["approve", "deny"])
def test_explicit_task_uses_agent_confirmations_journal_and_reports(
    tmp_path: Path, answer: str
) -> None:
    setup = Setup(tmp_path)
    try:
        with_character(setup)
        enable(setup)
        started = character(setup, "delegate", goal="Fix add")["delegation"]
        assert started["error"] is None and started["task"]["state"] == "running"
        task_id = started["task"]["task_id"]
        # Подтверждение — то же, что у задачи самого Agent, и видно, от кого задача.
        pending = setup.wait(lambda s: s["pending"] is not None)["pending"]
        assert pending["source_module"] == "character" and "+    return a + b" in pending["diff"]
        waiting = delegation(setup, lambda v: v["task"]["phase"] == "waiting_approval")
        assert waiting["task"]["state"] == "running"
        assert "a - b" in setup.file.read_text(
            encoding="utf-8"
        )  # без решения владельца файл не тронут
        busy = character(setup, "delegate", goal="Another task")["delegation"]
        assert busy["error"] == "agent_busy"
        setup.route("POST", "decision", id=pending["id"], action=answer)
        done = delegation(setup, lambda v: v["task"]["state"] != "running")["task"]
        assert ("a + b" in setup.file.read_text(encoding="utf-8")) == (answer == "approve")
        assert done["state"] == "done" and done["result"] == "Done"

        runs = AgentRuns(setup.db)
        origin = next(e for e in runs.events(task_id) if e["kind"] == "agent.origin")
        assert origin["payload"]["module_id"] == "character"
        assert runs.tool_calls(task_id)[0].permission == (
            "approved" if answer == "approve" else "denied"
        )
        history = character(setup, "status")["history"]
        assert history[-1]["id"] == task_id and history[-1]["user"] == "[agent task] Fix add"
        assert history[-1]["assistant"] == "Done" and history[-1]["status"] == "completed"
    finally:
        setup.close()


def test_character_cancels_its_own_task_and_settings_change_revokes_binding(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    try:
        with_character(setup)
        enable(setup)
        task_id = character(setup, "delegate", goal="Fix add")["delegation"]["task"]["task_id"]
        setup.wait(lambda s: s["pending"] is not None)
        character(setup, "delegate_cancel")
        ended = delegation(setup, lambda v: v["task"]["state"] != "running")["task"]
        assert ended["state"] == "cancelled" and "a - b" in setup.file.read_text(encoding="utf-8")
        assert AgentRuns(setup.db).get(task_id).status == "cancelled"
        assert setup.route("GET", "state")["pending"] is None
        assert character(setup, "status")["history"][-1]["status"] == "cancelled"

        enable(setup)
        from ramvi_service.agent.desktop import defaults

        setup.route("POST", "settings", **defaults())  # новые права — старая привязка не годна
        refused = character(setup, "delegate", goal="Fix add")["delegation"]
        assert refused["error"] == "agent_unavailable" and not refused["enabled"]
        assert "a - b" in setup.file.read_text(encoding="utf-8")

        enable(setup)
        setup.route("POST", "delegate", module="character", enabled=False)
        assert character(setup, "delegate", goal="Fix add")["delegation"]["error"] == (
            "agent_unavailable"
        )
    finally:
        setup.close()


def test_delegate_route_accepts_only_modules_that_request_agent(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        setup.host.register(
            Manifest("plain", "1.0.0", "ramvi.tests.module_fixture:create"),
        )
        setup.host.activate("plain")  # запущен, но права agent не просил
        for module in ("agent", "plain", "missing"):
            with pytest.raises(ChatError):
                setup.route(
                    "POST",
                    "delegate",
                    module=module,
                    enabled=True,
                    workspace=str(setup.project),
                    model_id="test-model",
                )
        with_character(setup)
        with pytest.raises(ChatError):  # путь, права и стратегия не задаются модулем
            setup.route("POST", "delegate", module="character", enabled=True, write=True)
    finally:
        setup.close()


class NoAgent(Services):
    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method.startswith("agent."):
            raise NotImplementedError("Agent is not connected")
        return super().call(method, payload)


def test_without_agent_the_ordinary_dialogue_goes_on() -> None:
    m = active(NoAgent())
    assert m.dialogue("delegate", {"goal": "Fix add"})["delegation"]["error"] == "agent_off"
    m.dialogue("agent_bind", {"binding_id": "stale"})
    refused = m.dialogue("delegate", {"goal": "Fix add"})["delegation"]
    assert refused["error"] == "agent_unavailable" and not refused["enabled"]
    assert m.dialogue("delegate_cancel")["delegation"]["task"] is None
    assert "error" in m.dialogue("delegate", {"goal": "  "})
    assert "error" in m.dialogue("agent_bind", {"binding_id": 5})
    m.dialogue("start", {"text": "Привет", "model": "one"})
    assert finish(m)["job"]["state"] == "completed"  # обычный разговор продолжается
    m.dispose()


class FakeAgent:
    """Агент по сценарию: события приходят частями, отмена — только своей задачи."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.polls = 0
        self.cancelled = False

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, payload))
        if method == "agent.create":
            return {"task_id": "t1"}
        if method == "agent.cancel":
            assert payload == {"task_id": "t1"}
            self.cancelled = True
            return {"ok": True}
        if method == "agent.events":
            self.polls += 1
            finished = self.cancelled or self.polls > 3
            return {
                "events": [{"type": "agent.step", "cursor": self.polls}],
                "cursor": self.polls,
                "gap": False,
                "done": finished,
            }
        if method == "agent.status":
            if self.cancelled:
                result = {
                    "status": "cancelled",
                    "stop_reason": "cancelled",
                    "result": "",
                    "steps": 1,
                }
                return {"state": "cancelled", "phase": "cancelled", "result": result, "error": None}
            if self.polls > 3:
                result = {"status": "done", "stop_reason": "done", "result": "x" * 5000, "steps": 4}
                return {"state": "done", "phase": "done", "result": result, "error": None}
            return {"state": "running", "phase": "running", "result": None, "error": None}
        return {"ok": True}


def test_follower_reads_feed_bounds_result_and_forgets_handle() -> None:
    agent = FakeAgent()
    finished: list[dict[str, Any]] = []
    d = Delegation(agent, finished.append, poll=0.001)
    d.bind("b1")
    d.start("Сделай отчёт")
    assert d.thread is not None
    d.thread.join(2)
    (task,) = finished
    assert task["state"] == "done" and len(task["result"]) == 2048 and task["steps"] == 4
    assert ("agent.forget", {"task_id": "t1"}) in agent.calls
    assert agent.calls[0] == ("agent.create", {"goal": "Сделай отчёт", "binding_id": "b1"})
    # Повтор после завершения — новая задача; отмена шлётся только для своей.
    agent2 = FakeAgent()
    stopped: list[dict[str, Any]] = []
    d2 = Delegation(agent2, stopped.append, poll=0.05)
    d2.bind("b2")
    d2.start("Долгая задача")
    d2.stop()
    assert agent2.cancelled and not (d2.thread and d2.thread.is_alive())
    assert stopped == []  # после выключения модуля в разговор ничего не дописывается


def test_character_package_does_not_import_agent() -> None:
    for path in (ROOT / "character" / "ramvi_character").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.startswith("ramvi_service") for n in names), path


def test_agent_and_character_confirmations_wait_in_turn(tmp_path: Path) -> None:
    from test_agent_desktop import response

    setup = Setup(tmp_path)

    def complete(request: Any, cancel: Any) -> dict[str, Any]:
        if any(m.get("role") == "tool" for m in request["messages"]):
            return response("finish", summary="Done")
        return response("edit_file", path="calc.py", old_text="a - b", new_text="a + b")

    setup.provider.complete = complete  # type: ignore[method-assign]
    try:
        with_character(setup)
        enable(setup)
        setup.start()
        first = setup.wait(lambda s: s["pending"] is not None)["pending"]
        character(setup, "delegate", goal="Fix add")
        delegation(setup, lambda v: v["task"]["state"] == "running")
        time.sleep(0.3)  # вторая просьба не затирает первую, пока та не решена
        assert setup.route("GET", "state")["pending"]["id"] == first["id"]
        setup.route("POST", "decision", id=first["id"], action="deny")
        second = setup.wait(
            lambda s: s["pending"] is not None and s["pending"]["id"] != first["id"]
        )["pending"]
        assert {first["source_module"], second["source_module"]} == {"agent", "character"}
        setup.route("POST", "decision", id=second["id"], action="deny")
        assert delegation(setup, lambda v: v["task"]["state"] != "running")["task"]["state"] == (
            "done"
        )
    finally:
        setup.close()


def test_deactivating_character_cancels_its_task(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        with_character(setup)
        enable(setup)
        task_id = character(setup, "delegate", goal="Fix add")["delegation"]["task"]["task_id"]
        setup.wait(lambda s: s["pending"] is not None)
        setup.host.deactivate("character")
        setup.wait(lambda s: s["pending"] is None)
        deadline = time.monotonic() + 4
        while AgentRuns(setup.db).get(task_id).status == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert AgentRuns(setup.db).get(task_id).status == "cancelled"
        assert "a - b" in setup.file.read_text(encoding="utf-8")
    finally:
        setup.close()


class FailingAgent(Services):
    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "agent.create":
            return {"task_id": "t9"}
        if method == "agent.events":
            return {"events": [], "cursor": 0, "gap": False, "done": True}
        if method == "agent.status":
            result = {"status": "failed", "stop_reason": "stuck", "result": "half", "steps": 2}
            return {"state": "failed", "phase": "failed", "result": result, "error": None}
        if method.startswith("agent."):
            return {"ok": True}
        return super().call(method, payload)


def test_failed_task_is_reported_as_error_without_partial_text() -> None:
    m = active(FailingAgent())
    m.dialogue("agent_bind", {"binding_id": "b"})
    m.dialogue("delegate", {"goal": "Сломанная задача"})
    thread = m._delegation.thread
    assert thread is not None
    thread.join(2)
    task = m.dialogue("delegation")["delegation"]["task"]
    assert task["state"] == "failed" and task["stop_reason"] == "stuck"
    turn = m.dialogue("status")["history"][-1]
    assert turn["status"] == "error" and turn["assistant"] == ""
    m.dispose()


class EndlessAgent(Services):
    def __init__(self) -> None:
        super().__init__()
        self.stops = 0

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "agent.create":
            return {"task_id": "t5"}
        if method == "agent.cancel":
            self.stops += 1
            return {"ok": True}
        if method == "agent.events":
            return {"events": [], "cursor": 0, "gap": False, "done": False}
        if method == "agent.status":
            return {"state": "running", "phase": "running", "result": None, "error": None}
        return super().call(method, payload)


def test_module_deactivation_stops_its_follower() -> None:
    svc = EndlessAgent()
    m = active(svc)
    m.dialogue("agent_bind", {"binding_id": "b"})
    m.dialogue("delegate", {"goal": "Долгая задача"})
    thread = m._delegation.thread
    m.deactivate()
    assert svc.stops == 1 and thread is not None and not thread.is_alive()
