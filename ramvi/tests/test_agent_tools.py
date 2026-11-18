"""6.2 (AG-02): инструменты агента — схема, журнал, файловые инструменты."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.files import FileLimits, WriteProposal, register_file_tools
from ramvi_service.agent.tools import (
    SCHEMA_VERSION,
    StorageJournal,
    ToolResult,
    ToolRunner,
    ToolSpec,
    redact,
    validate,
)
from ramvi_service.agent.workspace import Permissions, Workspace
from ramvi_storage import open_core
from ramvi_storage.database import Database
from ramvi_storage.repositories import AgentRuns


class ListJournal:
    """Журнал в памяти: проверяет порядок «запись → исполнение → результат»."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        self.entries.append({"tool": tool, "args": args, "permission": permission})
        return str(len(self.entries) - 1)

    def finish(self, call_id: str, result: str, duration: float) -> None:
        self.entries[int(call_id)]["result"] = json.loads(result)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src" / "main.py").write_bytes(b"import os\n\ndef main():\n    return 42\n")
    (root / "src" / "pkg" / "util.py").write_bytes(b"def helper():\n    return 'Main'\n")
    (root / "README.md").write_bytes(b"# Demo\nmain entry\n")
    (root / ".env").write_bytes(b"TOKEN=main-secret\n")
    (root / ".git" / "config").write_bytes(b"main = 1\n")
    (root / "blob.bin").write_bytes(b"main\0\1\2")
    (tmp_path / "outside.txt").write_bytes(b"main outside\n")
    return root


def _runner(root: Path, permissions: Permissions | None = None, **limits: Any) -> ToolRunner:
    runner = ToolRunner()
    register_file_tools(runner, Workspace(root, permissions), FileLimits(**limits))
    return runner


def _run(runner: ToolRunner, name: str, **args: Any) -> ToolResult:
    return runner.execute(name, args, ListJournal())


# --- схема и проверка аргументов --------------------------------------------------------


def test_schema_is_versioned_and_exportable(project: Path) -> None:
    runner = _runner(project)
    schema = runner.schema()
    assert schema["version"] == SCHEMA_VERSION
    names = [tool["name"] for tool in schema["tools"]]
    assert names == [
        "list_dir",
        "read_file",
        "find_files",
        "grep",
        "diff",
        "write_file",
        "edit_file",
    ]
    effects = {tool["name"]: tool["effect"] for tool in schema["tools"]}
    assert effects["write_file"] == effects["edit_file"] == "write"
    for spec in runner.specs:
        exported = spec.to_openai()
        assert exported["type"] == "function"
        assert exported["function"]["parameters"]["additionalProperties"] is False
    with pytest.raises(ValueError):
        runner.register(runner.specs[0], lambda args: ToolResult(True, ""))


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "$.path: required"),
        ({"path": 1}, "$.path: expected string"),
        ({"path": "a", "rm": True}, "$.rm: unknown argument"),
        ({"path": "a", "start_line": 0}, "$.start_line: below 1"),
        ({"path": "a", "start_line": True}, "$.start_line: expected integer"),
    ],
)
def test_invalid_calls_are_not_executed(project: Path, args: dict[str, Any], expected: str) -> None:
    called: list[dict[str, Any]] = []
    runner = ToolRunner()
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}},
        "required": ["path"],
        "additionalProperties": False,
    }
    runner.register(
        ToolSpec("probe", "", schema, "read"),
        lambda a: called.append(a) or ToolResult(True, "ran"),  # type: ignore[func-returns-value]
    )
    journal = ListJournal()
    result = runner.execute("probe", args, journal)
    assert not result.ok and result.error == "agent.tool.invalid_args"
    assert expected in result.content
    assert not called
    assert journal.entries[0]["permission"] == "denied"
    assert runner.execute("rm_rf", {}, journal).error == "agent.tool.unknown"
    assert runner.execute("probe", "not an object", journal).error == "agent.tool.invalid_args"
    assert [e["tool"] for e in journal.entries] == ["probe", "rm_rf", "probe"]
    assert not called


def test_validate_enum_lengths_and_arrays() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["a", "b"]},
            "name": {"type": "string", "minLength": 2, "maxLength": 3},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        },
    }
    assert validate(schema, {"mode": "a", "name": "ab", "tags": ["x"]}) == []
    errors = validate(schema, {"mode": "c", "name": "abcd", "tags": ["x", 1, "z"]})
    assert len(errors) == 4


# --- журнал -------------------------------------------------------------------------------


def test_journal_is_written_before_execution(project: Path) -> None:
    journal = ListJournal()
    seen: list[int] = []
    runner = ToolRunner()
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    runner.register(
        ToolSpec("probe", "", schema, "read"),
        lambda a: seen.append(len(journal.entries)) or ToolResult(True, "ok"),  # type: ignore[func-returns-value]
    )
    runner.execute("probe", {}, journal)
    assert seen == [1]  # запись уже была, когда обработчик начал работу
    assert journal.entries[0]["result"]["ok"] is True

    class Broken(ListJournal):
        def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
            raise OSError("disk full")

    with pytest.raises(OSError):
        runner.execute("probe", {}, Broken())
    assert seen == [1]  # без записи в журнал не исполняется


def test_storage_journal_keeps_args_result_and_errors(project: Path, tmp_path: Path) -> None:
    db: Database = open_core(tmp_path / "data")
    try:
        runs = AgentRuns(db)
        run = runs.create("test", str(project))
        journal = StorageJournal(runs, run.id)
        runner = _runner(project)
        assert runner.execute("read_file", {"path": "README.md"}, journal).ok
        assert not runner.execute("read_file", {"path": "../outside.txt"}, journal).ok
        assert not runner.execute("read_file", {"bad": 1}, journal).ok
        calls = runs.tool_calls(run.id)
        assert [c.permission for c in calls] == ["auto", "auto", "denied"]
        assert calls[0].args_redacted == {"path": "README.md"}
        first = json.loads(calls[0].result or "")
        assert first["ok"] and "# Demo" in first["content"]
        assert json.loads(calls[1].result or "")["error"] == "agent.access.parent"
        assert all(c.duration is not None and c.duration >= 0 for c in calls)
    finally:
        db.close()


def test_redaction_hides_secret_keys_and_long_text() -> None:
    clean = redact({"api_key": "sk-1", "nested": {"Password": "x"}, "content": "a" * 5000})
    assert clean["api_key"] == "***" and clean["nested"]["Password"] == "***"
    assert clean["content"]["chars"] == 5000 and len(clean["content"]["head"]) == 200
    assert clean["content"]["sha256"] == hashlib.sha256(b"a" * 5000).hexdigest()[:16]


def test_errors_go_back_to_the_model(project: Path) -> None:
    runner = _runner(project)
    result = _run(runner, "read_file", path="../outside.txt")
    assert result.for_model().startswith("ERROR agent.access.parent")
    assert _run(runner, "read_file", path=".env").error == "agent.access.secret"
    assert _run(runner, "read_file", path="blob.bin").error == "agent.tool.not_text"
    assert _run(runner, "read_file", path="nope.py").error == "agent.access.not_found"

    def crash(args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("/private/path leaked")

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    runner.register(ToolSpec("crash", "", schema, "read"), crash)
    failed = _run(runner, "crash")
    assert failed.error == "agent.tool.failed" and "/private" not in failed.content


def test_long_results_are_truncated_for_the_model() -> None:
    result = ToolResult(True, "x" * 20_000)
    assert len(result.for_model()) < 16_100 and result.for_model().endswith("truncated]")


# --- чтение ---------------------------------------------------------------------------------


def test_list_dir_hides_secrets_and_marks_links(project: Path, tmp_path: Path) -> None:
    (project / "link").symlink_to(tmp_path, target_is_directory=True)
    result = _run(_runner(project), "list_dir")
    lines = result.content.splitlines()
    assert "d src/" in lines and "l link" in lines and "f README.md (18 B)" in lines
    assert not any(".env" in line for line in lines)
    assert _run(_runner(project), "list_dir", path="README.md").error == "agent.tool.not_directory"
    limited = _run(_runner(project, max_list_entries=2), "list_dir")
    assert limited.truncated and limited.data["entries"] == 2


def test_read_file_numbers_and_pages_lines(project: Path) -> None:
    runner = _runner(project)
    page = _run(runner, "read_file", path="src/main.py", start_line=3, max_lines=1)
    assert page.content.splitlines()[0] == "     3\tdef main():"
    assert page.data == {"start_line": 3, "end_line": 3, "total_lines": 4}
    assert page.truncated
    whole = _run(runner, "read_file", path="src/main.py")
    assert not whole.truncated and "[lines 1-4 of 4]" in whole.content
    big = _run(_runner(project, max_text_bytes=10), "read_file", path="src/main.py")
    assert big.truncated and "of 1+]" in big.content


def test_find_files_skips_git_and_secrets(project: Path) -> None:
    runner = _runner(project)
    assert _run(runner, "find_files", pattern="*.py").data["files"] == [
        "src/main.py",
        "src/pkg/util.py",
    ]
    assert _run(runner, "find_files", pattern="pkg/*.PY", path="src").data["files"] == [
        "src/pkg/util.py"
    ]
    assert _run(runner, "find_files", pattern="config").data["files"] == []
    assert _run(runner, "find_files", pattern=".env*").data["files"] == []
    limited = _run(_runner(project, max_find_results=1), "find_files", pattern="*.py")
    assert limited.truncated and len(limited.data["files"]) == 1
    walk = _run(_runner(project, max_walk_entries=2), "find_files", pattern="*")
    assert walk.error == "agent.tool.too_many_files"


def test_grep_is_bounded_and_respects_access(project: Path) -> None:
    runner = _runner(project)
    result = _run(runner, "grep", pattern="main")
    lines = result.content.splitlines()
    assert "README.md:2: main entry" in lines
    assert "src/main.py:3: def main():" in lines
    # Секреты, .git, бинарные и внешние файлы не просматриваются.
    assert not any(x in result.content for x in ("main-secret", "main = 1", "outside"))
    assert "skipped 1" in result.content  # blob.bin
    insensitive = _run(runner, "grep", pattern="MAIN", ignore_case=True, glob="util.py")
    assert insensitive.content.splitlines() == ["src/pkg/util.py:2:     return 'Main'"]
    assert _run(runner, "grep", pattern="def", path="src/main.py").data["matches"] == 1
    assert _run(runner, "grep", pattern="(").error == "agent.tool.bad_pattern"
    assert _run(runner, "grep", pattern="x", path="..").error == "agent.access.parent"
    capped = _run(runner, "grep", pattern="main", max_matches=1)
    assert capped.truncated and capped.data["matches"] == 1
    long = _run(_runner(project, max_pattern_chars=3), "grep", pattern="main")
    assert long.error == "agent.tool.pattern_too_long"


def test_diff_between_two_files(project: Path) -> None:
    (project / "src" / "main2.py").write_bytes(b"import os\n\ndef main():\n    return 43\n")
    result = _run(_runner(project), "diff", path="src/main.py", other_path="src/main2.py")
    assert "-    return 42\n+    return 43\n" in result.content
    same = _run(_runner(project), "diff", path="src/main.py", other_path="src/main.py")
    assert same.data == {"identical": True}


# --- подготовка записи ------------------------------------------------------------------


def test_write_proposal_does_not_touch_files(project: Path) -> None:
    runner = _runner(project, Permissions(write=True))
    original = (project / "src" / "main.py").read_bytes()
    result = _run(runner, "write_file", path="src/main.py", content="print('hi')\n")
    proposal = result.data["proposal"]
    assert isinstance(proposal, WriteProposal)
    assert (project / "src" / "main.py").read_bytes() == original
    assert proposal.before_sha256 == hashlib.sha256(original).hexdigest()
    assert proposal.after == b"print('hi')\n" and proposal.display == "src/main.py"
    assert "+print('hi')" in proposal.diff and "awaiting user approval" in result.content
    new = _run(runner, "write_file", path="docs/new.md", content="x")
    assert new.data["proposal"].before_sha256 is None
    assert new.data["proposal"].diff.startswith("--- /dev/null\n+++ b/docs/new.md\n")
    assert "\\ No newline at end of file" in new.data["proposal"].diff
    assert not (project / "docs").exists()
    same = _run(runner, "write_file", path="README.md", content="# Demo\nmain entry\n")
    assert same.error == "agent.tool.no_change"


def test_write_needs_permission_and_respects_bounds(project: Path) -> None:
    reader = _runner(project)
    result = _run(reader, "write_file", path="a.txt", content="x")
    assert result.error == "agent.access.permission"
    writer = _runner(project, Permissions(write=True))
    assert _run(writer, "write_file", path="../x", content="x").error == "agent.access.parent"
    hook = _run(writer, "write_file", path=".git/hooks/pre-commit", content="x")
    assert hook.error == "agent.access.git_write"
    assert _run(writer, "write_file", path=".env", content="x").error == "agent.access.secret"
    assert _run(writer, "write_file", path="blob.bin", content="x").error == "agent.tool.not_text"


def test_edit_replaces_one_unique_fragment(project: Path) -> None:
    runner = _runner(project, Permissions(write=True))
    result = _run(
        runner, "edit_file", path="src/main.py", old_text="return 42", new_text="return 7"
    )
    assert result.data["proposal"].after.decode().endswith("    return 7\n")
    assert "return 42" in (project / "src" / "main.py").read_text()
    missing = _run(runner, "edit_file", path="src/main.py", old_text="zzz", new_text="y")
    assert missing.error == "agent.tool.edit_not_found"
    (project / "dup.txt").write_bytes(b"a\na\n")
    dup = _run(runner, "edit_file", path="dup.txt", old_text="a", new_text="b")
    assert dup.error == "agent.tool.edit_ambiguous" and dup.data["params"]["occurrences"] == 2
    nofile = _run(runner, "edit_file", path="nope.txt", old_text="a", new_text="b")
    assert nofile.error == "agent.access.not_found"


def test_crlf_and_bom_are_preserved(project: Path) -> None:
    (project / "win.txt").write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
    runner = _runner(project, Permissions(write=True))
    result = _run(runner, "edit_file", path="win.txt", old_text="two", new_text="TWO")
    assert result.data["proposal"].after == b"\xef\xbb\xbfone\r\nTWO\r\n"
