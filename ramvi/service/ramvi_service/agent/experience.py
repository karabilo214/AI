"""AG-08: проверяемые переходы, связанные с хронологией, без обучения весов.

Оценка — наблюдение runtime, не награда модели и не доказательство решения задачи.
Состояние — ссылка на запуск/шаг и отпечаток входа; полный prompt не дублируется.
"""

import hashlib
import json
from typing import Any

from ramvi_service.agent.strategy import ToolCall
from ramvi_service.agent.tools import ToolResult, redact

SCHEMA = "ramvi.agent.experience/1"


def decision_state(
    run_id: str, step: int, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    encoded = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
    return {
        "run_id": run_id,
        "step": step,
        "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "message_count": len(messages),
        "available_tools": [t["function"]["name"] for t in tools],
        "evidence": "agent_runs + agent_events through agent.decision at this step",
    }


def experience(
    state: dict[str, Any],
    call: ToolCall | None,
    *,
    result: ToolResult | None = None,
    outcome: str = "",
    content: str = "",
) -> dict[str, Any]:
    if call is None:
        action: dict[str, Any] = {"kind": "message"}
    else:
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError:
            args = None
        action = {
            "kind": "tool",
            "call_id": call.id,
            "tool": call.name,
            "args": redact(args) if isinstance(args, dict) else {"invalid_json_object": True},
        }
    observation: dict[str, Any] = {"content": content[:2000]}
    if result is not None:
        outcome = "tool_ok" if result.ok else "tool_error"
        observation = {
            "ok": result.ok,
            "error": result.error,
            "content": result.content[:2000],
            "truncated": result.truncated or len(result.content) > 2000,
            "change_id": result.data.get("change_id"),
            "applied": result.data.get("applied", []),
            "exit_code": result.data.get("exit_code"),
        }
    return {
        "schema": SCHEMA,
        "state": state,
        "action": action,
        "result": observation,
        "evaluation": {"source": "runtime", "outcome": outcome, "task_verified": False},
    }
