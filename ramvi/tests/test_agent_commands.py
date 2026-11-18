"""6.2/6.5: run_command — предложение, блокировки, исполнение после подтверждения."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.commands import CommandProposal, classify, execute_command
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import ToolAccess, ToolResult
from ramvi_service.agent.workspace import Permissions, Workspace

TERMINAL = Permissions(terminal=True)


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        self.calls.append((tool, permission))
        return str(len(self.calls) - 1)

    def finish(self, call_id: str, result: str, duration: float) -> None:
        json.loads(result)


def _run(
    root: Path,
    command: str,
    permissions: Permissions = TERMINAL,
    access: ToolAccess | None = None,
    **extra: Any,
) -> ToolResult:
    runner = build_tools(Workspace(root, permissions), access)
    return runner.execute("run_command", {"command": command, **extra}, Journal())


@pytest.mark.parametrize(
    ("argv", "reason", "network"),
    [
        (["pytest", "-q"], None, False),
        (["python", "-m", "pytest"], None, False),
        (["git", "status"], None, False),
        (["git", "diff", "--stat"], None, False),
        (["rm", "-rf", "build"], "deletes_files", False),
        (["Remove-Item", "x"], "deletes_files", False),
        (["sudo", "make"], "system", False),
        (["mkfs.ext4", "/dev/sda"], "system", False),
        (["bash", "-c", "echo"], "shell", False),
        (["cmd.exe", "/c", "dir"], "shell", False),
        (["python3", "-c", "print(1)"], "inline_code", False),
        (["node", "-e", "1"], "inline_code", False),
        (["git", "push", "origin"], "git_history", True),
        (["git", "reset", "--hard"], "git_history", False),
        (["git", "reset", "HEAD~1"], None, False),
        (["git", "clean", "-fd"], "git_history", False),
        (["git", "branch", "-D", "x"], "git_history", False),
        (["git", "checkout", "--", "."], "git_history", False),
        (["git", "fetch"], None, True),
        (["curl", "https://example.invalid"], None, True),
        (["pip", "install", "x"], None, True),
        (["python", "-m", "pip", "install", "x"], None, True),
        (["npm", "test"], None, False),
        (["npm", "install"], None, True),
    ],
)
def test_classify(argv: list[str], reason: str | None, network: bool) -> None:
    assert classify(argv) == (reason, network)


def test_proposal_does_not_run_anything(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    script = tmp_path / "make_marker.py"
    script.write_bytes(f"open(r'{marker}', 'w').close()\n".encode())
    result = _run(tmp_path, f'"{sys.executable}" make_marker.py', timeout=30)
    assert result.ok, result.content
    proposal = result.data["proposal"]
    assert isinstance(proposal, CommandProposal)
    assert "awaiting user approval" in result.content
    assert proposal.timeout == 30 and proposal.destructive is None and not proposal.network
    assert not marker.exists()
    # Исполнение — только явным вызовом после подтверждения.
    done = execute_command(proposal)
    assert done.ok and done.data["exit_code"] == 0 and marker.exists()


def test_terminal_permission_and_tool_toggle(tmp_path: Path) -> None:
    denied = _run(tmp_path, "git status", Permissions())
    assert denied.error == "agent.access.permission"
    off = ToolAccess(disabled=frozenset({"run_command"}))
    assert _run(tmp_path, "git status", access=off).error == "agent.tool.disabled"
    runner = build_tools(Workspace(tmp_path, TERMINAL), off)
    offered = [t["function"]["name"] for t in runner.openai_tools()]
    assert "run_command" not in offered and "git_status" in offered


def test_shell_syntax_and_bad_commands_are_rejected(tmp_path: Path) -> None:
    assert _run(tmp_path, "pytest && rm -rf /").error == "agent.tool.shell_syntax"
    assert _run(tmp_path, "pytest > out.txt").error == "agent.tool.shell_syntax"
    assert _run(tmp_path, "echo 'unterminated").error == "agent.tool.bad_command"
    missing = _run(tmp_path, "definitely-not-a-program-xyz")
    assert missing.error == "agent.tool.command_not_found"
    assert _run(tmp_path, "git status", cwd="..").error == "agent.access.parent"


def test_destructive_commands_need_separate_permission(tmp_path: Path) -> None:
    blocked = _run(tmp_path, "rm -rf src")
    assert blocked.error == "agent.tool.destructive_blocked"
    assert blocked.data["params"]["reason"] == "deletes_files"
    allowed = _run(tmp_path, "rm -rf src", access=ToolAccess(allow_destructive_commands=True))
    assert allowed.ok and allowed.data["proposal"].destructive == "deletes_files"


def test_network_commands_need_network_permission(tmp_path: Path) -> None:
    assert _run(tmp_path, "git fetch").error == "agent.access.permission"
    both = Permissions(terminal=True, network=True)
    assert _run(tmp_path, "git fetch", both).data["proposal"].network


def test_program_path_must_stay_inside_workspace(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "tools").mkdir(parents=True)
    script = root / "tools" / "check.py"
    script.write_bytes(b"print('ok')\n")
    assert _run(root, "../outside.sh").error == "agent.access.parent"
    assert _run(root, "./missing.sh").error == "agent.access.not_found"
    result = _run(root, "tools/check.py")
    assert result.ok and result.data["proposal"].argv[0] == str(script.resolve())


def _proposal(tmp_path: Path, code: str, timeout: float = 30) -> CommandProposal:
    argv = (sys.executable, "-c", code)
    return CommandProposal(" ".join(argv), argv, tmp_path, ".", timeout, None, False)


def test_execute_returns_output_errors_and_exit_code(tmp_path: Path) -> None:
    code = "import sys; print('hello'); print('oops', file=sys.stderr); sys.exit(2)"
    result = execute_command(_proposal(tmp_path, code))
    assert result.ok and result.data["exit_code"] == 2
    assert result.content.splitlines() == [
        "exit code 2",
        "--- stdout ---",
        "hello",
        "--- stderr ---",
        "oops",
    ]
    slow = execute_command(_proposal(tmp_path, "import time; time.sleep(30)", timeout=1))
    assert not slow.ok and slow.error == "agent.tool.timeout" and slow.data["timed_out"]
    big = execute_command(_proposal(tmp_path, "print('x' * 5000)"), max_output_bytes=100)
    assert big.truncated and "output truncated" in big.content


def test_secret_environment_is_not_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "sk-secret")
    monkeypatch.setenv("SERVICE_TOKEN", "t-secret")
    monkeypatch.setenv("PLAIN_VALUE", "visible")
    code = "import os; print(sorted(k for k in os.environ if k in ('MY_API_KEY','SERVICE_TOKEN','PLAIN_VALUE')))"
    result = execute_command(_proposal(tmp_path, code))
    assert "['PLAIN_VALUE']" in result.content
    assert os.environ["MY_API_KEY"] == "sk-secret"  # в самом сервисе не тронуто
