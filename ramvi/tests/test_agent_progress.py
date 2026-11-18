"""6.1: распознавание зацикливания по журналу запуска."""

from ramvi_service.agent.files import unified_diff
from ramvi_service.agent.progress import ProgressMonitor, changed_lines
from ramvi_service.agent.tools import ToolResult


def test_changed_lines_from_diff() -> None:
    before = "".join(f"line {i}\n" for i in range(1, 21))
    after = before.replace("line 5\n", "LINE 5\n").replace("line 15\nline 16\n", "x\n")
    diff = unified_diff(before, after, "a.txt")
    assert changed_lines(diff) == {"a.txt": [5, 15]}
    new = unified_diff("", "hello\n", "n.txt", new=True)
    assert changed_lines(new) == {"n.txt": [0]}


def _write(path: str, before: str | None, after: str, lines: list[int]) -> ToolResult:
    files = [{"path": path, "before_sha256": before, "after_sha256": after, "lines": lines}]
    return ToolResult(True, "applied", {"change_id": "c", "applied": files})


def test_same_result_without_changes_counts_only_in_a_row() -> None:
    monitor = ProgressMonitor()
    same = ToolResult(True, "a.py")
    assert monitor.observe("find", "{}", same) is None
    assert monitor.observe("find", "{}", same) is None
    assert monitor.observe("find", "{}", same) == ("no_progress", "find")
    monitor.forgive("no_progress", "find")
    assert monitor.observe("find", "{}", same) is None


def test_a_change_in_between_means_progress() -> None:
    monitor = ProgressMonitor()
    same = ToolResult(True, "content")
    monitor.observe("read", "{}", same)
    monitor.observe("read", "{}", same)
    assert monitor.observe("edit", "{}", _write("a.py", "h0", "h1", [3])) is None
    assert monitor.observe("read", "{}", same) is None  # после записи — новое состояние
    command = ToolResult(True, "exit code 1", {"exit_code": 1})
    monitor.observe("run", "{}", command)
    assert monitor.observe("run", "{}", command) is None  # команда могла изменить файлы


def test_oscillation_and_same_lines() -> None:
    monitor = ProgressMonitor()
    assert monitor.observe("edit", "1", _write("a.py", "h0", "h1", [2])) is None
    assert monitor.observe("edit", "2", _write("a.py", "h1", "h0", [2])) == (
        "oscillation",
        "a.py",
    )
    lines = ProgressMonitor(max_same_lines=2)
    assert lines.observe("edit", "1", _write("b.py", "x0", "x1", [7])) is None
    assert lines.observe("edit", "2", _write("b.py", "x1", "x2", [7])) == ("same_lines", "b.py:7")
    lines.forgive("same_lines", "b.py:7")
    assert lines.observe("edit", "3", _write("b.py", "x2", "x3", [7])) is None


def test_checks_can_be_switched_off() -> None:
    monitor = ProgressMonitor(max_no_progress=None, max_same_lines=None)
    for _ in range(10):
        assert monitor.observe("find", "{}", ToolResult(True, "same")) is None
