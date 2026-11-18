"""AG-08: сменная стратегия и опыт; реальные права, журнал и изменения."""

from pathlib import Path
from typing import Any, Literal

import pytest
from ramvi_service.agent.approval import Decision
from ramvi_service.agent.journal import AgentJournal
from ramvi_service.agent.strategy import LLMStrategy, ModelReply, ToolCall
from ramvi_storage import open_core
from ramvi_storage.database import Database
from ramvi_storage.repositories import AgentRuns
from test_agent_runtime import FIX, Env, Script, call, done, reply

# Общая фикстура, как в соседних тестах агента.
from test_agent_runtime import db as db  # noqa: F401


def records(env: Env, run_id: str) -> list[dict[str, Any]]:
    return AgentJournal(env.runs).experience(run_id)


@pytest.mark.parametrize("value", [17, 913, 100_032])
def test_strategy_uses_observation_to_recover(tmp_path: Path, db: Database, value: int) -> None:
    """Новые значения/запуски: ошибка → наблюдение → исправленное действие."""

    class Recover:
        def decide(self, messages: Any, tools: Any, cancel: Any, max_tokens: Any) -> ModelReply:
            if messages[-1]["role"] == "user":
                return reply("1. Inspect", call("read_file", path="missing.txt"))
            observation = messages[-1]["content"]
            if "agent.tool.not_found" in observation or "ERROR" in observation:
                return reply("Correct the path", call("read_file", path="fresh.txt"))
            assert str(value) in observation
            return done(f"Read {value}")

    env = Env(tmp_path, db, Recover())
    (env.root / "fresh.txt").write_text(str(value))
    result = env.runtime.run("Read fresh.txt")
    assert result.status == "done"
    experience = records(env, result.run_id)
    assert [e["evaluation"]["outcome"] for e in experience] == [
        "tool_error",
        "tool_ok",
        "claimed_success",
    ]
    assert len({e["state"]["input_sha256"] for e in experience}) == 3
    assert all(not e["evaluation"]["task_verified"] for e in experience)
    assert experience[1]["action"]["args"] == {"path": "fresh.txt"}


def test_lazy_finish_is_only_a_claim(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(done("Fixed it")))
    result = env.runtime.run("Fix add")
    assert "a - b" in (env.root / "src/calc.py").read_text()
    (record,) = records(env, result.run_id)
    assert record["evaluation"] == {
        "source": "runtime",
        "outcome": "claimed_success",
        "task_verified": False,
    }


@pytest.mark.parametrize("decision", ["approve", "deny", "stop"])
def test_strategy_cannot_bypass_approval(
    tmp_path: Path, db: Database, decision: Literal["approve", "deny", "stop"]
) -> None:
    env = Env(tmp_path, db, Script(reply("", FIX), done("ok")), [Decision(decision)])
    result = env.runtime.run("Fix add")
    text = (env.root / "src/calc.py").read_text()
    assert ("a + b" in text) == (decision == "approve")
    record = records(env, result.run_id)[0]
    assert (
        record["evaluation"]["outcome"]
        == {"approve": "tool_ok", "deny": "tool_error", "stop": "cancelled"}[decision]
    )
    if decision == "approve":
        assert record["result"]["applied"][0]["path"] == "src/calc.py"
        assert record["result"]["change_id"]


def test_swapped_strategy_still_obeys_access(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(reply("", call("read_file", path="../outside")), done("ok")))
    result = env.runtime.run("Read outside")
    record = records(env, result.run_id)[0]
    assert record["evaluation"]["outcome"] == "tool_error"
    assert record["result"]["error"].startswith("agent.access.")


def test_persistence_and_cleanup(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(reply("", call("list_dir")), done("ok")))
    run = env.runtime.run("Inspect")
    expected = records(env, run.run_id)
    reopened = open_core(tmp_path / "data")
    try:
        journal = AgentJournal(AgentRuns(reopened))
        assert journal.experience(run.run_id) == expected
        assert journal.clear() == 1
        assert journal.experience(run.run_id) == []
    finally:
        reopened.close()


def test_llm_remains_the_default_adapter() -> None:
    # Старые импорты runtime остаются совместимыми.
    from ramvi_service.agent.runtime import LLMStrategy as Legacy

    assert Legacy is LLMStrategy
    assert ToolCall("id", "list_dir", "{}").name == "list_dir"


def test_strategy_receives_detached_history(tmp_path: Path, db: Database) -> None:
    class Mutating(Script):
        def decide(self, messages: Any, tools: Any, cancel: Any, max_tokens: Any) -> ModelReply:
            result = super().decide(messages, tools, cancel, max_tokens)
            messages[0]["content"] = "tampered"
            tools.clear()
            return result

    script = Mutating(reply("", call("list_dir")), done("ok"))
    env = Env(tmp_path, db, script)
    env.runtime.run("Inspect")
    from ramvi_service.agent.runtime import SYSTEM_PROMPT

    assert script.seen[1][0]["content"] == SYSTEM_PROMPT
    assert "list_dir" in script.tools[1]


def test_no_action_and_invalid_finish_are_recorded(tmp_path: Path, db: Database) -> None:
    env = Env(
        tmp_path,
        db,
        Script(
            reply("I will do it"),
            reply("", ToolCall("bad-finish", "finish", "[]")),
            done("Cannot do it", success=False),
        ),
    )
    result = env.runtime.run("Fix add")
    assert result.stop_reason == "not_possible"
    assert [e["evaluation"]["outcome"] for e in records(env, result.run_id)] == [
        "no_action",
        "tool_error",
        "claimed_failure",
    ]


def test_experience_redacts_arguments_and_isolates_runs(tmp_path: Path, db: Database) -> None:
    env = Env(
        tmp_path,
        db,
        Script(
            reply("", call("unknown", api_key="do-not-store")),
            done("ok"),
        ),
    )
    first = env.runtime.run("First")
    first_records = records(env, first.run_id)
    assert first_records[0]["action"]["args"]["api_key"] == "***"
    env.runtime.strategy = Script(done("Second"))
    second = env.runtime.run("Second")
    second_records = records(env, second.run_id)
    assert len(second_records) == 1
    assert second_records[0]["state"]["run_id"] == second.run_id
    assert records(env, first.run_id) == first_records
