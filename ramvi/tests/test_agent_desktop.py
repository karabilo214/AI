"""Trusted desktop routes with real gateway, journal, module worker and file approval."""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.desktop import AgentDesktop, defaults, validate_policy
from ramvi_service.agent.gateway import AgentGateway
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Authority, Manifest, ProviderCapabilities
from ramvi_service.modules.host import ModuleHost
from ramvi_service.resources import SharedLLM
from ramvi_service.sections import SectionRegistry
from ramvi_storage import open_core
from ramvi_storage.repositories import AgentRuns, Model, Models


def response(name: str, **args: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": "1. Inspect\n2. Fix\n3. Check",
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


class Provider:
    capabilities = ProviderCapabilities(tools=True)

    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[Any] = []

    def complete(self, request: Any, cancel: Any) -> dict[str, Any]:
        assert request["ramvi_min_context"] == 8192
        assert request["ramvi_model_id"] == "test-model"
        self.calls += 1
        self.messages.append(request["messages"])
        if self.calls == 1:
            return response("edit_file", path="calc.py", old_text="a - b", new_text="a + b")
        return response("finish", summary="Done")

    def cancel(self) -> None:
        pass

    def unload(self) -> None:
        pass


class Setup:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.db = open_core(root / "data")
        Models(self.db).upsert(Model("test-model", "test", "qwen3", "fixture.gguf", 0))
        authority = Authority()
        self.provider = Provider()
        self.llm = SharedLLM(authority, self.provider)
        self.host = ModuleHost(root / "host", llm=self.llm)
        self.host.agent = AgentGateway(authority, self.host.grants)
        self.sections = SectionRegistry(self.host)
        self.sections.install(
            Manifest(
                "agent",
                "1.0.0",
                "ramvi_service.agent.module:create",
                capabilities=("agent",),
                ui={"section": {"view": "agent", "title_key": "section.agent"}},
            )
        )
        self.desktop = AgentDesktop(
            root / "data", AgentRuns(self.db), self.host, self.sections, self.llm
        )
        self.project = root / "project"
        self.project.mkdir()
        self.file = self.project / "calc.py"
        self.file.write_text("def add(a, b):\n    return a - b\n")

    def route(self, method: str, endpoint: str, **body: Any) -> dict[str, Any]:
        return self.desktop.route(method, "/agent/" + endpoint, body)

    def start(self) -> str:
        return str(
            self.route(
                "POST", "start", goal="Fix add", workspace=str(self.project), model_id="test-model"
            )["task_id"]
        )

    def wait(self, predicate: Any) -> dict[str, Any]:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            state = self.route("GET", "state")
            if predicate(state):
                return state
            time.sleep(0.01)
        raise AssertionError("state was not reached")

    def close(self) -> None:
        self.desktop.close()
        self.host.close()
        self.host.agent.close()
        self.db.close()


@pytest.mark.parametrize("answer", ["approve", "deny", "stop"])
def test_real_pending_diff_decision_and_rollback(tmp_path: Path, answer: str) -> None:
    setup = Setup(tmp_path)
    try:
        policy = defaults()
        policy["permissions"]["write"] = True
        setup.route("POST", "settings", **policy)
        task_id = setup.start()
        state = setup.wait(lambda s: s["pending"] is not None)
        pending = state["pending"]
        assert "-    return a - b" in pending["diff"] and "+    return a + b" in pending["diff"]
        assert "a - b" in setup.file.read_text()
        with pytest.raises(ChatError):
            setup.route("POST", "decision", id="expired", action="approve")
        setup.route("POST", "decision", id=pending["id"], action=answer, reason="Keep the file")
        ended = setup.wait(lambda s: s["task"]["state"] != "running")
        assert ("a + b" in setup.file.read_text()) == (answer == "approve")
        assert setup.route("GET", "history")["runs"][0]["id"] == task_id
        detail = setup.route("GET", "runs/" + task_id)
        if answer == "approve":
            change = detail["changes"][0]
            assert change["files"] == ["calc.py"]
            assert setup.route("POST", "why", path="calc.py", run_id=task_id)["reasons"]
            setup.route("POST", "rollback/" + change["id"], run_id=task_id)
            assert "a - b" in setup.file.read_text()
        elif answer == "stop":
            assert ended["task"]["state"] == "cancelled"
        setup.route("POST", "clear", confirm=True)
        assert setup.route("GET", "history")["runs"] == []
    finally:
        setup.close()


def test_save_policy_stops_waiting_task_and_persists_defaults(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        policy = defaults()
        policy["permissions"]["write"] = True
        setup.route("POST", "settings", **policy)
        task_id = setup.start()
        setup.wait(lambda s: s["pending"] is not None)
        policy["permissions"]["write"] = False
        policy["limits"]["max_steps"] = None
        setup.route("POST", "settings", **policy)
        assert "a - b" in setup.file.read_text()
        assert AgentRuns(setup.db).get(task_id).status == "cancelled"
        persisted = json.loads((tmp_path / "data/agent_settings.json").read_text())
        assert persisted["limits"]["max_steps"] is None and not persisted["permissions"]["write"]
    finally:
        setup.close()


def test_invalid_policy_and_absent_module_cannot_start(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        policy = defaults()
        policy["limits"]["max_steps"] = True
        with pytest.raises(ValueError):
            validate_policy(policy)
        setup.sections.remove("agent")
        with pytest.raises(ChatError):
            setup.start()
        assert setup.provider.calls == 0
    finally:
        setup.close()


def test_attached_text_is_data_and_does_not_expand_workspace(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        policy = defaults()
        policy["permissions"]["write"] = True
        setup.route("POST", "settings", **policy)
        setup.route(
            "POST",
            "start",
            goal="Fix add using the attached specification",
            workspace=str(setup.project),
            model_id="test-model",
            attachments=[{"name": "spec.txt", "text": "The result must be a plus b."}],
        )
        state = setup.wait(lambda s: s["pending"] is not None)
        assert "The result must be a plus b" in setup.provider.messages[0][-1]["content"]
        assert "untrusted data" in setup.provider.messages[0][-1]["content"]
        assert not (setup.project / "spec.txt").exists()
        setup.route("POST", "decision", id=state["pending"]["id"], action="deny")
        setup.wait(lambda s: s["task"]["state"] != "running")
        assert "a - b" in setup.file.read_text()
    finally:
        setup.close()


def test_oversized_context_attachments_are_rejected_before_model(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    try:
        with pytest.raises(ChatError, match="agent.context_attachments"):
            setup.route(
                "POST",
                "start",
                goal="Read the file",
                workspace=str(setup.project),
                model_id="test-model",
                attachments=[{"name": "large.txt", "text": "x" * 30000}],
            )
        assert setup.provider.calls == 0
    finally:
        setup.close()
