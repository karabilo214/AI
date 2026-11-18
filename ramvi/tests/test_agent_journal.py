"""6.7 (AG-07): журнал — хронология, причина изменения файла, очистка."""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.approval import ApprovalGate, ApprovalRequest, Decision
from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.journal import DAY, AgentJournal, JournalRetention
from ramvi_service.agent.runtime import AgentRuntime
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import ToolAccess
from ramvi_service.agent.workspace import Permissions, Workspace
from ramvi_storage import open_core
from ramvi_storage.database import Database
from ramvi_storage.repositories import AgentRuns
from test_agent_runtime import FIX, Script, call, done, reply  # noqa: E402

PLAN = "1. Read calc.py\n2. add() subtracts instead of adding: fix it"


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_core(tmp_path / "data")
    yield database
    database.close()


class Setup:
    def __init__(self, tmp: Path, db: Database) -> None:
        self.root = tmp / "project"
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
        self.changes = tmp / "changes"
        self.ws = Workspace(self.root, Permissions(write=True, terminal=True))
        self.runs = AgentRuns(db)
        self.journal = AgentJournal(self.runs, self.changes)

    def run(self, script: Script, decisions: list[Decision], goal: str = "Fix add()") -> Any:
        answers = list(decisions)

        def ask(request: ApprovalRequest) -> Decision:
            return answers.pop(0)

        gate = ApprovalGate(ask, ChangeStore(self.changes, self.ws), ToolAccess())
        tools = build_tools(self.ws, ToolAccess(), gate=gate)
        return AgentRuntime(script, tools, self.runs).run(goal, workspace=str(self.root))


def fix_script() -> Script:
    return Script(
        reply(PLAN, call("read_file", path="src/calc.py"), tokens=100),
        reply("", FIX, tokens=200),
        done("Fixed add()."),
    )


def test_why_was_the_file_changed(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    result = setup.run(fix_script(), [Decision("approve")])
    (answer,) = setup.journal.why_changed("src/calc.py")
    assert answer["run_id"] == result.run_id and answer["objective"] == "Fix add()"
    assert answer["plan"] == PLAN and answer["step"] == 2 and answer["tool"] == "edit_file"
    assert answer["args"] == {"path": "src/calc.py", "old_text": "a - b", "new_text": "a + b"}
    assert answer["approval"] == {"action": "approve", "series": 0, "reason": ""}
    assert "-    return a - b\n+    return a + b\n" in answer["diff"]
    assert answer["rollback"] == "available"
    ChangeStore(setup.changes, setup.ws).rollback(answer["change_id"])
    assert setup.journal.why_changed("./src\\calc.py")[0]["rollback"] == "rolled_back"
    assert setup.journal.why_changed("src/other.py") == []


def test_denied_change_is_in_the_timeline_but_not_a_change(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    script = Script(reply("", FIX), done("Left as is."))
    result = setup.run(script, [Decision("deny", reason="keep it")])
    assert setup.journal.why_changed("src/calc.py") == []
    kinds = [e["kind"] for e in setup.journal.timeline(result.run_id)]
    assert "agent.tool_approval_required" in kinds and "agent.tool_approval_decided" in kinds
    decided = next(
        e
        for e in setup.journal.timeline(result.run_id)
        if e["kind"] == "agent.tool_approval_decided"
    )
    assert decided["payload"]["decision"]["reason"] == "keep it"
    errors = setup.journal.errors(result.run_id)
    assert [e["payload"]["error"] for e in errors] == ["agent.approval.denied"]


def test_summary_shows_time_tokens_context_and_errors(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    result = setup.run(fix_script(), [Decision("approve")])
    summary = setup.journal.summary(result.run_id)
    assert summary["status"] == "done" and summary["steps"] == 3
    assert summary["tokens"] == {"prompt_tokens": 310, "completion_tokens": 3}
    assert summary["context_peak"] == 200 and summary["tool_calls"] == 2
    assert len(summary["changes"]) == 1 and summary["errors"] == 0
    assert summary["duration"] >= 0 and summary["result"] == "Fixed add()."
    usage = [e for e in setup.journal.timeline(result.run_id) if e["kind"] == "agent.usage"]
    assert [u["payload"]["context"] for u in usage] == [100, 200, 10]


def test_long_texts_are_cut_in_the_journal(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    huge = "x" * 100_000
    result = setup.run(Script(reply(huge, call("list_dir")), done("ok")), [])
    plan = next(e for e in setup.journal.timeline(result.run_id) if e["kind"] == "agent.plan")
    assert len(plan["payload"]["text"]) < 33_000 and "chars cut" in plan["payload"]["text"]


def _age(db: Database, run_id: str, days: float) -> None:
    db.execute(
        "UPDATE agent_runs SET ended_at = ? WHERE id = ?", (time.time() - days * DAY, run_id)
    )


def test_clear_removes_runs_and_snapshots(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    old = setup.run(fix_script(), [Decision("approve")])
    (setup.root / "src" / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    new = setup.run(fix_script(), [Decision("approve")])
    _age(db, old.run_id, 10)
    old_change = setup.journal.summary(old.run_id)["changes"][0]
    assert setup.journal.clear(before=time.time() - DAY) == 1
    assert not (setup.changes / old_change).exists()
    assert [r.id for r in setup.runs.finished()] == [new.run_id]
    assert setup.journal.clear() == 1
    assert setup.runs.finished() == [] and setup.journal.size()["snapshots"] == 0


def test_running_runs_are_never_cleared(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    running = setup.runs.create("in progress", str(setup.root))
    setup.runs.update(running.id, status="running")
    assert setup.journal.clear() == 0
    assert setup.runs.get(running.id).status == "running"


def test_auto_cleanup_by_age_and_size(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    ids = []
    for _ in range(4):
        (setup.root / "src" / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
        ids.append(setup.run(fix_script(), [Decision("approve")]).run_id)
    _age(db, ids[0], 40)
    for index, run_id in enumerate(ids[1:], 1):
        _age(db, run_id, 5 - index)  # от старых к новым
    assert setup.journal.maintain(JournalRetention(max_age_days=30, max_bytes=None)) == 1
    left = [r.id for r in setup.runs.finished()]
    assert left == ids[1:]
    one_run = setup.journal.size()
    per_run = (one_run["database"] + one_run["snapshots"]) // 3
    removed = setup.journal.maintain(JournalRetention(max_age_days=None, max_bytes=per_run + 10))
    assert removed == 2 and [r.id for r in setup.runs.finished()] == [ids[-1]]
    unlimited = JournalRetention(max_age_days=None, max_bytes=None)
    assert setup.journal.maintain(unlimited) == 0


def test_compaction_returns_disk_space(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    big = "y" * 30_000
    for _ in range(30):
        setup.run(Script(reply(big, call("list_dir")), done("ok")), [])
    db_file = tmp_path / "data" / "ramvi.sqlite3"
    wal = Path(str(db_file) + "-wal")
    before = db_file.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    setup.journal.clear()
    after = db_file.stat().st_size + (wal.stat().st_size if wal.exists() else 0)
    assert after < before / 2, (before, after)


def test_storage_journal_rows_are_json(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    result = setup.run(fix_script(), [Decision("approve")], goal="Исправь add() — проверка")
    for event in setup.journal.timeline(result.run_id):
        json.dumps(event["payload"], ensure_ascii=False)
    assert setup.journal.summary(result.run_id)["objective"] == "Исправь add() — проверка"
