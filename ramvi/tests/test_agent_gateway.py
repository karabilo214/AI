"""6.10: Character и текстовый клиент проходят один runtime и настоящие подтверждения."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.approval import Decision
from ramvi_service.agent.gateway import AgentGateway
from ramvi_service.agent.runtime import ModelReply
from ramvi_service.context import GrantStore
from ramvi_service.contracts.modules import Authority, Manifest
from ramvi_service.modules.host import ModuleHost
from ramvi_storage.database import Database
from test_agent_runtime import FIX, Env, Script, done, reply
from test_agent_runtime import db as db  # noqa: F401


def wait(gateway: AgentGateway, owner: Any, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        state = gateway.status(owner, task_id)
        if state["state"] != "running":
            return state
        time.sleep(0.01)
    raise AssertionError("task did not finish")


@pytest.mark.parametrize("approve", [True, False])
def test_actual_workers_delegate_through_same_policy(
    tmp_path: Path, db: Database, approve: bool
) -> None:
    host = ModuleHost(tmp_path / "host", timeout=4)
    gateway = AgentGateway(host.authority, host.grants)
    host.agent = gateway
    for name in ("character", "chat"):
        host.register(
            Manifest(
                name,
                "1.0.0",
                "ramvi.tests.module_fixture:create",
                capabilities=("agent",),
                permissions=("agent",),
            ),
            approved_permissions=("agent",),
        )
    try:
        for name in ("character", "chat"):
            host.activate(name)
        owner = host._entry("character").owner
        other = host._entry("chat").owner
        assert owner is not None and other is not None
        env = Env(
            tmp_path,
            db,
            Script(reply("1. Fix", FIX), done("ok")),
            [Decision("approve" if approve else "deny")],
        )
        binding = gateway.bind(owner, lambda: env.runtime)
        task_id = host.invoke(
            "character",
            "agent",
            {
                "method": "create",
                "payload": {
                    "goal": "Fix add",
                    "binding_id": binding,
                },
            },
        )["task_id"]
        result = wait(gateway, owner, task_id)
        assert result["source_module"] == "character"
        assert result["result"]["run_id"] == task_id
        assert ("a + b" in (env.root / "src/calc.py").read_text(encoding="utf-8")) == approve
        assert env.runs.tool_calls(task_id)[0].permission == ("approved" if approve else "denied")
        events = host.invoke(
            "character",
            "agent",
            {
                "method": "events",
                "payload": {
                    "task_id": task_id,
                },
            },
        )
        assert events["done"] and events["events"][-1]["type"] == "agent.done"
        origin = next(e for e in env.runs.events(task_id) if e["kind"] == "agent.origin")
        assert origin["payload"] == {
            "principal_id": "local_user",
            "module_id": "character",
            "session_id": owner.session_id,
        }
        chat_scope = host._entry("chat").services
        character_scope = host._entry("character").services
        assert chat_scope is not None and character_scope is not None
        for method in ("status", "events", "cancel"):
            with pytest.raises(PermissionError):
                chat_scope.call("agent." + method, {"task_id": task_id})
        with pytest.raises(PermissionError):
            gateway.create(other, "Other", binding)
        with pytest.raises(ValueError):
            character_scope.call(
                "agent.create",
                {"goal": "Bypass", "binding_id": binding, "permissions": {"write": True}},
            )
        gateway.forget(owner, task_id)
        with pytest.raises(PermissionError):
            gateway.status(owner, task_id)
    finally:
        host.close()
        gateway.close()
    assert not any(t.name == "ramvi-agent-task" for t in threading.enumerate())


@pytest.mark.parametrize("stop", ["cancel", "revoke", "scope_stop", "close"])
def test_cancellation_and_release_leave_no_task_thread(
    tmp_path: Path, db: Database, stop: str
) -> None:
    authority = Authority()
    owner = authority.issue("caller", "user", "session")
    grants = GrantStore(authority)
    gateway = AgentGateway(authority, grants, timeout=4)
    entered = threading.Event()

    def slow(cancel: threading.Event) -> ModelReply:
        entered.set()
        assert cancel.wait(3)
        raise InterruptedError()

    env = Env(tmp_path, db, Script(slow))
    binding = gateway.bind(owner, lambda: env.runtime)
    with pytest.raises(PermissionError):
        gateway.create(owner, "Task", binding)
    grants.allow(owner, "agent")
    try:
        task_id = gateway.create(owner, "Task", binding)
        assert entered.wait(2)
        with pytest.raises(RuntimeError):
            gateway.create(owner, "Parallel", binding)
        if stop == "cancel":
            gateway.cancel(owner, task_id)
        elif stop == "revoke":
            grants.revoke(owner, "agent")
        elif stop == "scope_stop":
            gateway.release_owner(owner)
        else:
            gateway.close()
        if stop in ("cancel", "revoke"):
            assert wait(gateway, owner, task_id)["state"] == "cancelled"
    finally:
        gateway.close()
    assert not any(t.name == "ramvi-agent-task" for t in threading.enumerate())


def test_absent_gateway_is_explicit_and_character_optional(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=3)
    host.register(
        Manifest("chat", "1.0.0", "ramvi.tests.module_fixture:create", capabilities=("agent",))
    )
    with pytest.raises(NotImplementedError):
        host.activate("chat")
    host.close()


def test_gateway_uses_trusted_model_readiness(tmp_path: Path, db: Database) -> None:
    from ramvi_service.agent.readiness import ModelFacts

    authority = Authority()
    owner = authority.issue("chat", "user", "session")
    grants = GrantStore(authority)
    grants.allow(owner, "agent")
    gateway = AgentGateway(authority, grants)
    env = Env(tmp_path, db, Script(done("must not be called")))
    from ramvi_storage.repositories import Model, Models

    Models(db).upsert(Model("small-context", "test", "qwen3", "test.gguf", 0))
    try:
        binding = gateway.bind(
            owner, lambda: env.runtime, model_id="small-context", facts=ModelFacts(True, 128)
        )
        task = gateway.create(owner, "Inspect", binding)
        assert wait(gateway, owner, task)["result"]["stop_reason"] == "model_not_ready"
        assert env.runs.get(task).workspace == str(env.root)
        assert env.runs.get(task).model_id == "small-context"
    finally:
        gateway.close()


def test_factory_failure_is_a_terminal_event_without_error_text() -> None:
    authority = Authority()
    owner = authority.issue("chat", "user", "session")
    grants = GrantStore(authority)
    grants.allow(owner, "agent")
    gateway = AgentGateway(authority, grants)

    def broken() -> Any:
        raise RuntimeError("private path or credential must not be exposed")

    try:
        binding = gateway.bind(owner, broken)
        task = gateway.create(owner, "Task", binding)
        result = wait(gateway, owner, task)
        assert result["error"] == "RuntimeError" and result["phase"] == "failed"
        (event,) = gateway.events(owner, task)["events"]
        assert event == {
            "type": "agent.gateway_failed",
            "task_id": task,
            "cursor": 1,
            "error": "RuntimeError",
        }
    finally:
        gateway.close()


def test_host_infers_authority_and_grants_from_injected_agent(tmp_path: Path) -> None:
    authority = Authority()
    grants = GrantStore(authority)
    gateway = AgentGateway(authority, grants)
    host = ModuleHost(tmp_path, agent=gateway)
    try:
        assert host.authority is authority and host.grants is grants
    finally:
        host.close()
        gateway.close()
