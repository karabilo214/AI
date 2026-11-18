"""6.6 (AG-06): готовность модели, подсказки к ошибкам, неверные вызовы."""

import tempfile
from pathlib import Path
from typing import Any

from ramvi_service.agent.readiness import (
    ModelFacts,
    ReadinessCheck,
    facts_from_fit,
    required_context,
)
from ramvi_service.agent.runtime import (
    FINISH_TOOL,
    SYSTEM_PROMPT,
    AgentRuntime,
    RunLimits,
    ToolCall,
)
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import ToolResult
from ramvi_service.agent.workspace import Permissions, Workspace
from test_agent_runtime import Script, done, reply  # noqa: E402


def _tools(permissions: Permissions | None = None) -> list[dict[str, Any]]:
    root = Path(tempfile.mkdtemp())
    runner = build_tools(Workspace(root, permissions or Permissions.full()))
    return [*runner.openai_tools(), FINISH_TOOL]


def _codes(items: tuple[dict[str, Any], ...]) -> list[str]:
    return [item["code"] for item in items]


def test_required_context_matches_what_worked_on_real_model() -> None:
    """Проверка 6.1: при 2048 ответ обрезался, при 8192 задачи решались."""
    need = required_context(SYSTEM_PROMPT, _tools())
    assert 2048 < need <= 8192
    check = ReadinessCheck()
    assert _codes(check.check(ModelFacts(True, 2048), SYSTEM_PROMPT, _tools()).errors) == [
        "agent.ready.context"
    ]
    assert check.check(ModelFacts(True, 8192), SYSTEM_PROMPT, _tools()).ok
    # Только чтение — схем меньше, требование ниже.
    assert required_context(SYSTEM_PROMPT, _tools(Permissions())) < need


def test_errors_and_warnings() -> None:
    check = ReadinessCheck()
    no_tools = check.check(ModelFacts(False, 32768), SYSTEM_PROMPT, _tools())
    assert _codes(no_tools.errors) == ["agent.ready.no_tools"]
    short_model = check.check(ModelFacts(True, 32768, model_context=4096), SYSTEM_PROMPT, [])
    assert _codes(short_model.errors) == ["agent.ready.model_context"]
    small = check.check(ModelFacts(True, 32768, size_b=0.6), SYSTEM_PROMPT, _tools())
    assert small.ok and _codes(small.warnings) == ["agent.ready.small_model"]
    assert small.json()["warnings"][0]["params"] == {"size_b": 0.6, "recommended_b": 7.0}
    big = check.check(ModelFacts(True, 32768, size_b=8.0), SYSTEM_PROMPT, _tools())
    assert big.ok and not big.warnings


def test_facts_from_model_fit_verdict() -> None:
    verdict = {
        "context": 8192,
        "model": {"size_b": 14.0},
        "task": {"reasons": [{"code": "task.reason.context", "params": {"model": 4096}}]},
    }
    facts = facts_from_fit(verdict, ("chat", "tools", "json"))
    assert facts == ModelFacts(True, 8192, model_context=4096, size_b=14.0)
    assert not facts_from_fit({"context": 8192}, ("chat",)).tools


def test_not_ready_model_does_not_start(tmp_path: Path) -> None:
    script = Script(done("never"))
    events: list[dict[str, Any]] = []
    runtime = AgentRuntime(script, build_tools(Workspace(tmp_path)), emit=events.append)
    result = runtime.run("x", facts=ModelFacts(tools=False, context=8192))
    assert (result.status, result.stop_reason, result.error) == (
        "failed",
        "model_not_ready",
        "agent.ready.no_tools",
    )
    assert script.seen == []  # к модели не обращались
    assert [e["type"] for e in events] == ["agent.readiness", "agent.done"]


def test_small_model_warning_is_shown_but_run_goes_on(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    runtime = AgentRuntime(Script(done("ok")), build_tools(Workspace(tmp_path)), emit=events.append)
    result = runtime.run("x", facts=ModelFacts(True, 8192, size_b=0.6))
    assert result.status == "done"
    readiness = events[0]
    assert readiness["type"] == "agent.readiness" and readiness["ok"]
    assert readiness["warnings"][0]["code"] == "agent.ready.small_model"


def test_hints_tell_the_model_whether_to_retry() -> None:
    fix = ToolResult.failure("agent.tool.edit_not_found", path="a.py").for_model()
    assert fix.endswith("(Fix the call and try again.)")
    stop = ToolResult.failure("agent.access.secret", path=".env").for_model()
    assert "Do not repeat this call" in stop and "finish with success=false" in stop
    denied = ToolResult.failure("agent.approval.denied", reason="no").for_model()
    assert "Follow the reason" in denied


def test_repeated_invalid_calls_are_a_checkpoint(tmp_path: Path) -> None:
    calls = [ToolCall(f"c{i}", "read_file", "{bad json " + str(i)) for i in range(5)]
    script = Script(*[reply("", c) for c in calls], done("never"))
    asked: list[str] = []

    def on_limit(request: Any) -> bool:
        asked.append(request.reason)
        return False

    runtime = AgentRuntime(
        script, build_tools(Workspace(tmp_path)), limits=RunLimits(), on_limit=on_limit
    )
    result = runtime.run("x")
    assert asked == ["invalid_calls"]
    assert (result.status, result.stop_reason, result.steps) == ("cancelled", "invalid_calls", 3)
