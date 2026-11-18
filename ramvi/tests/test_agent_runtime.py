"""6.1 (AG-01): цикл агента — шаги, итог, лимиты, остановка."""

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.approval import ApprovalGate, ApprovalRequest, Decision
from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.runtime import (
    SYSTEM_PROMPT,
    AgentRuntime,
    LLMStrategy,
    ModelReply,
    RunLimits,
    ToolCall,
)
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import ToolAccess
from ramvi_service.agent.workspace import Permissions, Workspace
from ramvi_storage import open_core
from ramvi_storage.database import Database
from ramvi_storage.repositories import AgentRuns


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(f"call-{name}-{len(json.dumps(args))}", name, json.dumps(args))


def reply(content: str = "", *calls: ToolCall, tokens: int = 10, finish: str = "") -> ModelReply:
    return ModelReply(
        content,
        tuple(calls),
        finish or ("tool_calls" if calls else "stop"),
        {"prompt_tokens": tokens, "completion_tokens": 1},
    )


class Script:
    """Модель по сценарию; запоминает, что ей показали."""

    def __init__(self, *replies: ModelReply | Callable[[threading.Event], ModelReply]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []
        self.tools: list[list[str]] = []

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
        max_tokens: int,
    ) -> ModelReply:
        self.seen.append([dict(m) for m in messages])
        self.tools.append([t["function"]["name"] for t in tools])
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return item(cancel) if callable(item) else item


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_core(tmp_path / "data")
    yield database
    database.close()


class Env:
    def __init__(
        self,
        tmp: Path,
        db: Database,
        strategy: Any,
        decisions: list[Decision] | None = None,
        limits: RunLimits | None = None,
    ) -> None:
        self.root = tmp / "project"
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
        self.ws = Workspace(self.root, Permissions(write=True, terminal=True))
        self.decisions = list(decisions or [])
        self.statuses: list[str] = []
        self.runs = AgentRuns(db)
        self.gate = ApprovalGate(self.ask, ChangeStore(tmp / "changes", self.ws), ToolAccess())
        self.tools = build_tools(self.ws, ToolAccess(), gate=self.gate)
        self.events: list[dict[str, Any]] = []
        self.runtime = AgentRuntime(
            strategy, self.tools, self.runs, limits=limits, emit=self.events.append
        )

    def ask(self, request: ApprovalRequest) -> Decision:
        self.statuses.append(self.runs.recent()[0].status)
        return self.decisions.pop(0)

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.events]


FIX = call("edit_file", path="src/calc.py", old_text="a - b", new_text="a + b")


def test_multi_step_task_completes(tmp_path: Path, db: Database) -> None:
    script = Script(
        reply("1. Find the file\n2. Fix add\n3. Verify", call("find_files", pattern="*.py")),
        reply("", call("read_file", path="src/calc.py")),
        reply("", FIX),
        reply("", call("read_file", path="src/calc.py")),
        reply("Fixed add(): it now returns a + b."),
    )
    env = Env(tmp_path, db, script, [Decision("approve")])
    result = env.runtime.run("Fix add() in calc.py", workspace=str(env.root), model_id=None)
    assert (result.status, result.stop_reason, result.steps) == ("done", "done", 5)
    assert result.result == "Fixed add(): it now returns a + b."
    assert (env.root / "src" / "calc.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    # Наблюдение возвращается модели: на втором шаге она видит найденный файл.
    assert "src/calc.py" in script.seen[1][-1]["content"]
    assert script.seen[0][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "    2\t    return a + b" in script.seen[4][-1]["content"]
    # Сохранено: итог, причина, шаги, токены, журнал инструментов.
    run = env.runs.get(result.run_id)
    assert (run.status, run.stop_reason, run.steps, run.result) == (
        "done",
        "done",
        5,
        result.result,
    )
    assert run.token_usage == {"prompt_tokens": 50, "completion_tokens": 5}
    calls = env.runs.tool_calls(result.run_id)
    assert [c.tool_name for c in calls] == ["find_files", "read_file", "edit_file", "read_file"]
    assert [c.permission for c in calls] == ["auto", "auto", "approved", "auto"]
    assert env.statuses == ["waiting_approval"]
    kinds = env.kinds()
    assert kinds[:3] == ["agent.step", "agent.plan", "agent.tool_call"]
    assert "agent.tool_approval_required" in kinds and kinds[-1] == "agent.done"
    approval = next(e for e in env.events if e["type"] == "agent.tool_approval_required")
    assert "+    return a + b" in approval["request"]["diff"]
    assert all(e["version"] == 1 and e["run_id"] == result.run_id for e in env.events)
    assert [e["seq"] for e in env.events] == list(range(1, len(env.events) + 1))


def test_simple_answer_without_tools(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(reply("Hello!")))
    result = env.runtime.run("Say hello")
    assert (result.status, result.steps, result.result) == ("done", 1, "Hello!")


def test_stops_at_step_limit(tmp_path: Path, db: Database) -> None:
    paths = iter(["src", ".", "src", "."] * 5)
    script = Script(lambda cancel: reply("", call("list_dir", path=next(paths))))
    env = Env(tmp_path, db, script, limits=RunLimits(max_steps=3))
    result = env.runtime.run("loop forever")
    assert (result.status, result.stop_reason, result.steps) == ("failed", "max_steps", 3)
    assert env.runs.get(result.run_id).ended_at is not None


def test_stops_at_token_limit(tmp_path: Path, db: Database) -> None:
    script = Script(
        reply("", call("list_dir"), tokens=600), reply("", call("list_dir", path="src"), tokens=600)
    )
    env = Env(tmp_path, db, script, limits=RunLimits(max_tokens=1000))
    result = env.runtime.run("expensive")
    assert (result.status, result.stop_reason, result.steps) == ("failed", "max_tokens", 2)


def test_time_limit_does_not_interrupt_a_generation(tmp_path: Path, db: Database) -> None:
    def slow(cancel: threading.Event) -> ModelReply:
        time.sleep(0.8)
        assert not cancel.is_set()  # время не прерывает идущий ответ модели
        return reply("", call("list_dir"))

    env = Env(tmp_path, db, Script(slow), limits=RunLimits(max_duration=0.5))
    result = env.runtime.run("slow")
    assert (result.status, result.stop_reason, result.steps) == ("failed", "max_duration", 1)
    assert len(env.runs.tool_calls(result.run_id)) == 1  # шаг доделан


def test_time_limit_lets_a_running_command_finish(tmp_path: Path, db: Database) -> None:
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "slow.py").write_bytes(b"import time\ntime.sleep(1.5)\nprint('ok')\n")
    command = call("run_command", command=f'"{sys.executable}" slow.py')
    env = Env(
        tmp_path, db, Script(reply("", command)), [Decision("approve")], RunLimits(max_duration=0.5)
    )
    result = env.runtime.run("run slow")
    assert result.stop_reason == "max_duration"
    (finished,) = env.runs.tool_calls(result.run_id)
    assert json.loads(finished.result or "")["content"].startswith("exit code 0")


def test_user_continues_past_a_checkpoint(tmp_path: Path, db: Database) -> None:
    paths = iter(["src", "."] * 10)
    script = Script(lambda cancel: reply("", call("list_dir", path=next(paths))))
    answers = [True, False]
    asked: list[tuple[str, int, str]] = []

    def on_limit(request: Any) -> bool:
        asked.append((request.reason, request.steps, env.runs.recent()[0].status))
        return answers.pop(0)

    env = Env(tmp_path, db, script, limits=RunLimits(max_steps=2, max_no_progress=None))
    env.runtime.on_limit = on_limit
    result = env.runtime.run("long work")
    assert asked == [("max_steps", 2, "waiting_approval"), ("max_steps", 4, "waiting_approval")]
    assert (result.status, result.stop_reason, result.steps) == ("cancelled", "max_steps", 4)
    assert env.kinds().count("agent.limit_reached") == 2


def test_limits_can_be_switched_off(tmp_path: Path, db: Database) -> None:
    paths = iter(["src", "."] * 20)
    replies: list[Any] = [lambda c: reply("", call("list_dir", path=next(paths)))] * 30
    script = Script(*replies, reply("finished"))
    off = RunLimits(max_steps=None, max_tokens=None, max_duration=None, max_no_progress=None)
    env = Env(tmp_path, db, script, limits=off)
    result = env.runtime.run("very long work")
    assert (result.status, result.steps) == ("done", 31)


def test_stuck_asks_and_can_continue(tmp_path: Path, db: Database) -> None:
    script = Script(*[reply("", call("list_dir"))] * 3, reply("done"))
    env = Env(tmp_path, db, script)
    env.runtime.on_limit = lambda request: request.reason == "stuck"
    result = env.runtime.run("same call")
    assert result.status == "done" and len(env.runs.tool_calls(result.run_id)) == 3


def test_user_stop_cancels_the_run(tmp_path: Path, db: Database) -> None:
    cancel = threading.Event()

    def generating(event: threading.Event) -> ModelReply:
        threading.Timer(0.2, cancel.set).start()
        event.wait(30)
        raise InterruptedError

    env = Env(tmp_path, db, Script(generating))
    result = env.runtime.run("stop me", cancel=cancel)
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
    assert env.runs.get(result.run_id).status == "cancelled"


def test_stop_in_approval_cancels_and_changes_nothing(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(reply("", FIX)), [Decision("stop")])
    result = env.runtime.run("fix")
    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
    assert b"a - b" in (env.root / "src" / "calc.py").read_bytes()
    assert env.gate.cancel is None  # gate возвращён в исходное состояние


def test_denied_action_reason_goes_back_and_run_continues(tmp_path: Path, db: Database) -> None:
    script = Script(reply("", FIX), reply("OK, I will not change the file."))
    env = Env(tmp_path, db, script, [Decision("deny", reason="do not touch calc.py")])
    result = env.runtime.run("fix")
    assert result.status == "done" and result.steps == 2
    assert "do not touch calc.py" in script.seen[1][-1]["content"]


def test_repeated_call_without_new_information_is_a_loop(tmp_path: Path, db: Database) -> None:
    env = Env(tmp_path, db, Script(reply("", call("list_dir"))))
    result = env.runtime.run("loop")
    assert (result.status, result.stop_reason) == ("failed", "stuck")
    assert len(env.runs.tool_calls(result.run_id)) == 3
    limit = next(e for e in env.events if e["type"] == "agent.limit_reached")
    assert limit["request"]["detail"] == "no_progress: list_dir"


def edit(old: str, new: str) -> ToolCall:
    return ToolCall(
        f"e-{old}-{new}",
        "edit_file",
        json.dumps({"path": "src/calc.py", "old_text": old, "new_text": new}),
    )


def test_rereading_after_own_edit_is_not_a_loop(tmp_path: Path, db: Database) -> None:
    read = call("read_file", path="src/calc.py")
    script = Script(
        reply("", read),
        reply("", edit("a - b", "a + b")),
        reply("", read),
        reply("", edit("return", "return  ")),
        reply("", read),
        reply("done"),
    )
    env = Env(tmp_path, db, script, [Decision("approve", series=5)])
    assert env.runtime.run("fix").status == "done"


def test_undoing_own_change_is_a_loop(tmp_path: Path, db: Database) -> None:
    script = Script(reply("", edit("a - b", "a + b")), reply("", edit("a + b", "a - b")))
    env = Env(tmp_path, db, script, [Decision("approve", series=5)])
    result = env.runtime.run("flip")
    assert result.stop_reason == "stuck"
    limit = next(e for e in env.events if e["type"] == "agent.limit_reached")
    assert limit["request"]["detail"] == "oscillation: src/calc.py"


def test_editing_the_same_line_again_and_again_is_a_loop(tmp_path: Path, db: Database) -> None:
    script = Script(
        reply("", edit("a - b", "a * b")),
        reply("", edit("a * b", "a / b")),
        reply("", edit("a / b", "a ** b")),
        reply("never reached"),
    )
    env = Env(tmp_path, db, script, [Decision("approve", series=5)])
    result = env.runtime.run("same line")
    assert result.stop_reason == "stuck"
    limit = next(e for e in env.events if e["type"] == "agent.limit_reached")
    assert limit["request"]["detail"] == "same_lines: src/calc.py:2"
    assert len(env.runs.tool_calls(result.run_id)) == 3


def test_invalid_tool_call_is_reported_not_executed(tmp_path: Path, db: Database) -> None:
    broken = ToolCall("c1", "read_file", "{not json")
    unknown = ToolCall("c2", "delete_everything", "{}")
    script = Script(reply("", broken, unknown), reply("I could not read the file."))
    env = Env(tmp_path, db, script)
    result = env.runtime.run("read")
    assert result.status == "done"
    tool_messages = [m for m in script.seen[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert tool_messages[0]["content"].startswith("ERROR agent.tool.invalid_args")
    assert tool_messages[1]["content"].startswith("ERROR agent.tool.unknown")
    assert [c.permission for c in env.runs.tool_calls(result.run_id)] == ["denied", "denied"]


def test_too_many_calls_in_one_step(tmp_path: Path, db: Database) -> None:
    many = [call("list_dir", path=p) for p in ("src", ".", "src/", "./src")]
    script = Script(reply("", *many), reply("done"))
    env = Env(tmp_path, db, script, limits=RunLimits(max_calls_per_step=2))
    env.runtime.run("many")
    contents = [m["content"] for m in script.seen[1] if m["role"] == "tool"]
    assert len(contents) == 4 and contents[2] == contents[3] == "ERROR agent.run.too_many_calls"
    assert len(env.runs.tool_calls(env.runs.recent()[0].id)) == 2


@pytest.mark.parametrize(
    ("final", "reason"),
    [(reply("cut", finish="length"), "length"), (reply("   "), "empty_answer")],
)
def test_bad_final_answers_fail(
    tmp_path: Path, db: Database, final: ModelReply, reason: str
) -> None:
    env = Env(tmp_path, db, Script(final))
    result = env.runtime.run("x")
    assert (result.status, result.stop_reason) == ("failed", reason)


def test_model_error_fails_the_run(tmp_path: Path, db: Database) -> None:
    def broken(cancel: threading.Event) -> ModelReply:
        raise ConnectionError("engine is down")

    env = Env(tmp_path, db, Script(broken))
    result = env.runtime.run("x")
    assert (result.status, result.stop_reason, result.error) == (
        "failed",
        "model_error",
        "ConnectionError",
    )


def test_only_available_tools_are_offered(tmp_path: Path, db: Database) -> None:
    script = Script(reply("ok"))
    env = Env(tmp_path, db, script)
    env.ws.permissions = Permissions()  # только чтение
    env.runtime.run("x")
    assert "edit_file" not in script.tools[0] and "read_file" in script.tools[0]


def test_runs_without_storage(tmp_path: Path) -> None:
    root = tmp_path / "p"
    root.mkdir()
    runtime = AgentRuntime(
        Script(reply("", call("list_dir")), reply("empty")), build_tools(Workspace(root))
    )
    result = runtime.run("look")
    assert result.status == "done" and result.run_id.startswith("run-")


def test_llm_strategy_builds_request_and_parses_tool_calls() -> None:
    sent: list[dict[str, Any]] = []

    def complete(request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        sent.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                            },
                            {"function": {"name": "list_dir", "arguments": {"path": "."}}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }

    strategy = LLMStrategy(complete, {"ramvi_model": "m"})
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    result = strategy.decide([{"role": "user", "content": "x"}], tools, threading.Event(), 99)
    assert sent[0]["tools"] == tools and sent[0]["max_tokens"] == 99
    assert sent[0]["temperature"] == 0 and sent[0]["ramvi_model"] == "m"
    assert result.content == "" and result.finish_reason == "tool_calls"
    assert result.tool_calls[0] == ToolCall("call_1", "read_file", '{"path": "a"}')
    assert result.tool_calls[1] == ToolCall("call_1", "list_dir", '{"path": "."}')
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    strategy.decide([], [], threading.Event(), 5)
    assert "tools" not in sent[1]
