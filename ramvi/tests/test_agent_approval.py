"""6.4 (AG-04): diff перед записью, подтверждение, серии, откат через снимки."""

import json
import os
import shutil
import stat
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import ramvi_service.agent.changes as changes_module
from ramvi_service.agent.approval import ApprovalGate, ApprovalRequest, Decision
from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import StopRequested, ToolAccess, ToolError, ToolResult, ToolRunner
from ramvi_service.agent.workspace import Permissions, Workspace

ORIGINAL = b"def answer():\n    return 42\n"


class Journal:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        self.entries.append({"tool": tool, "permission": permission})
        return str(len(self.entries) - 1)

    def finish(
        self, call_id: str, result: str, duration: float, permission: str | None = None
    ) -> None:
        entry = self.entries[int(call_id)]
        entry["result"] = json.loads(result)
        if permission is not None:
            entry["permission"] = permission


class Asker:
    """Пользователь по сценарию: решения по очереди, запросы запоминаются."""

    def __init__(self, *decisions: Decision, on_ask: Callable[[], None] | None = None) -> None:
        self.decisions, self.requests, self.on_ask = list(decisions), [], on_ask

    requests: list[ApprovalRequest]

    def __call__(self, request: ApprovalRequest) -> Decision:
        self.requests.append(request)
        if self.on_ask is not None:
            self.on_ask()
        return self.decisions.pop(0)


class Env:
    def __init__(self, tmp: Path, asker: Asker, access: ToolAccess | None = None) -> None:
        self.root = tmp / "project"
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        self.file = self.root / "src" / "app.py"
        if not self.file.exists():
            self.file.write_bytes(ORIGINAL)
        self.access = access or ToolAccess()
        self.ws = Workspace(self.root, Permissions(write=True, terminal=True))
        self.store = ChangeStore(tmp / "changes", self.ws)
        self.gate = ApprovalGate(asker, self.store, self.access)
        self.runner: ToolRunner = build_tools(self.ws, self.access, gate=self.gate)
        self.journal = Journal()

    def call(self, name: str, **args: Any) -> ToolResult:
        return self.runner.execute(name, args, self.journal)

    def edit(self, old: str = "return 42", new: str = "return 7") -> ToolResult:
        return self.call("edit_file", path="src/app.py", old_text=old, new_text=new)


APPROVE = Decision("approve")


def test_denied_change_leaves_files_untouched(tmp_path: Path) -> None:
    asker = Asker(Decision("deny", reason="keep 42, change the docstring instead"))
    env = Env(tmp_path, asker)
    result = env.edit()
    assert env.file.read_bytes() == ORIGINAL
    assert result.error == "agent.approval.denied"
    assert "keep 42" in result.for_model()  # поправка задачи уходит модели
    request = asker.requests[0]
    assert request.kind == "write" and request.files == ("src/app.py",)
    assert "-    return 42\n+    return 7\n" in request.diff  # diff показан до записи
    assert env.journal.entries[0]["permission"] == "denied"
    assert env.store.history() == []


def test_approved_change_is_applied_and_can_be_rolled_back(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    result = env.edit()
    assert result.ok, result.content
    assert env.file.read_bytes() == b"def answer():\n    return 7\n"
    change_id = result.data["change_id"]
    entry = env.journal.entries[0]
    assert entry["permission"] == "approved" and entry["result"]["change_id"] == change_id
    record = env.store.rollback(change_id)
    assert record.rolled_back and env.file.read_bytes() == ORIGINAL
    with pytest.raises(ToolError) as again:
        env.store.rollback(change_id)
    assert again.value.code == "agent.change.already_rolled_back"


def test_stop_raises_and_changes_nothing(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(Decision("stop")))
    with pytest.raises(StopRequested):
        env.edit()
    assert env.file.read_bytes() == ORIGINAL
    assert env.journal.entries[0]["permission"] == "denied"
    assert env.journal.entries[0]["result"]["error"] == "agent.approval.stopped"


def test_cancelled_run_does_not_ask_or_apply(tmp_path: Path) -> None:
    asker = Asker(APPROVE)
    env = Env(tmp_path, asker)
    env.gate.cancel = threading.Event()
    env.gate.cancel.set()
    with pytest.raises(StopRequested):
        env.edit()
    assert asker.requests == [] and env.file.read_bytes() == ORIGINAL


def test_user_can_turn_confirmation_off(tmp_path: Path) -> None:
    def never(request: ApprovalRequest) -> Decision:
        raise AssertionError("must not ask")

    env = Env(tmp_path, Asker())
    env.gate = ApprovalGate(never, env.store, ToolAccess(confirm_writes=False))
    env.runner.gate = env.gate
    assert env.edit().ok
    assert env.journal.entries[0]["permission"] == "auto"
    assert b"return 7" in env.file.read_bytes()


def test_series_approves_a_limited_number_of_actions(tmp_path: Path) -> None:
    asker = Asker(Decision("approve", series=2), Decision("deny"))
    env = Env(tmp_path, asker)
    assert env.edit("return 42", "return 1").ok
    assert env.edit("return 1", "return 2").ok
    assert env.edit("return 2", "return 3").ok
    assert env.edit("return 3", "return 4").error == "agent.approval.denied"
    assert len(asker.requests) == 2  # первый вопрос + после окончания серии
    assert [e["permission"] for e in env.journal.entries] == [
        "approved",
        "approved",
        "approved",
        "denied",
    ]
    assert b"return 3" in env.file.read_bytes()
    capped = Env(
        tmp_path / "capped", Asker(Decision("approve", series=1000)), ToolAccess(max_series=1)
    )
    capped.edit("return 42", "return 1")
    capped.edit("return 1", "return 2")
    assert capped.gate._series.remaining["write"] == 0


def test_commands_run_only_after_approval(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    asker = Asker(Decision("deny"), APPROVE)
    env = Env(tmp_path, asker)
    (env.root / "mark.py").write_bytes(f"open(r'{marker}', 'w').close()\n".encode())
    command = f'"{sys.executable}" mark.py'
    assert env.call("run_command", command=command).error == "agent.approval.denied"
    assert not marker.exists()
    done = env.call("run_command", command=command)
    assert done.ok and done.data["exit_code"] == 0 and marker.exists()
    assert asker.requests[0].kind == "command" and asker.requests[0].command == command


def test_destructive_command_always_asks_even_in_series(tmp_path: Path) -> None:
    access = ToolAccess(allow_destructive_commands=True)
    victim = tmp_path / "project" / "victim.txt"
    asker = Asker(Decision("approve", series=5), Decision("deny"))
    env = Env(tmp_path, asker, access)
    victim.write_bytes(b"x")
    assert env.call("run_command", command="git status").ok
    assert env.call("run_command", command="rm victim.txt").error == "agent.approval.denied"
    assert victim.exists() and len(asker.requests) == 2
    assert asker.requests[1].destructive == "deletes_files"


def test_file_changed_before_approval_is_not_overwritten(tmp_path: Path) -> None:
    holder: dict[str, Env] = {}

    def user_edits() -> None:
        holder["env"].file.write_bytes(b"user edit\n")

    env = Env(tmp_path, Asker(APPROVE, on_ask=user_edits))
    holder["env"] = env
    result = env.edit()
    assert result.error == "agent.change.conflict"
    assert env.file.read_bytes() == b"user edit\n"


def test_patch_is_atomic_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    real = changes_module._atomic_write

    def flaky(target: Path, data: bytes, mode: int | None) -> None:
        if target.name == "second.py":
            raise OSError("disk full")
        real(target, data, mode)

    monkeypatch.setattr(changes_module, "_atomic_write", flaky)
    edits = [
        {"path": "src/app.py", "old_text": "return 42", "new_text": "return 0"},
        {"path": "new/deep/second.py", "old_text": "", "new_text": "x = 1\n"},
    ]
    result = env.call("apply_patch", edits=edits)
    assert result.error == "agent.tool.failed"
    assert env.file.read_bytes() == ORIGINAL
    assert not (env.root / "new").exists()
    assert env.store.history() == []


def test_new_files_and_folders_are_removed_on_rollback(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    edits = [
        {"path": "src/app.py", "old_text": "return 42", "new_text": "return 0"},
        {"path": "docs/guide/intro.md", "old_text": "", "new_text": "# Intro\n"},
    ]
    result = env.call("apply_patch", edits=edits)
    assert result.ok and (env.root / "docs" / "guide" / "intro.md").exists()
    env.store.rollback(result.data["change_id"])
    assert env.file.read_bytes() == ORIGINAL and not (env.root / "docs").exists()


def test_rollback_does_not_overwrite_later_user_edits(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    change_id = env.edit().data["change_id"]
    env.file.write_bytes(b"user kept working\n")
    with pytest.raises(ToolError) as conflict:
        env.store.rollback(change_id)
    assert conflict.value.code == "agent.change.conflict"
    assert env.file.read_bytes() == b"user kept working\n"
    env.store.rollback(change_id, force=True)
    assert env.file.read_bytes() == ORIGINAL


def test_rollback_survives_restart_and_revoked_grant(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    change_id = env.edit().data["change_id"]
    env.ws.revoke()
    reopened = ChangeStore(
        tmp_path / "changes", Workspace(env.root)
    )  # новый процесс, только чтение
    assert [r.id for r in reopened.history()] == [change_id]
    reopened.rollback(change_id)
    assert env.file.read_bytes() == ORIGINAL


def test_revoked_grant_blocks_apply(tmp_path: Path) -> None:
    holder: dict[str, Env] = {}
    env = Env(tmp_path, Asker(APPROVE, on_ask=lambda: holder["env"].ws.revoke()))
    holder["env"] = env
    assert env.edit().error == "agent.access.revoked"
    assert env.file.read_bytes() == ORIGINAL


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_file_mode_is_preserved(tmp_path: Path) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    os.chmod(env.file, 0o755)
    change_id = env.edit().data["change_id"]
    assert stat.S_IMODE(os.stat(env.file).st_mode) == 0o755
    env.store.rollback(change_id)
    assert stat.S_IMODE(os.stat(env.file).st_mode) == 0o755


def test_snapshot_is_written_before_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = Env(tmp_path, Asker(APPROVE))
    order: list[str] = []
    real = changes_module._atomic_write

    def spy(target: Path, data: bytes, mode: int | None) -> None:
        order.append(target.name)
        real(target, data, mode)

    monkeypatch.setattr(changes_module, "_atomic_write", spy)
    env.edit()
    assert order == ["0.orig", "manifest.json", "app.py", "manifest.json"]
    shutil.rmtree(tmp_path / "changes")
