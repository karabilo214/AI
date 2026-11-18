"""6.2: git-инструменты только читают и не дают репозиторию запустить программу."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.git import register_git_tools
from ramvi_service.agent.tools import ToolResult, ToolRunner
from ramvi_service.agent.workspace import Restrictions, Workspace

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True
    ).stdout


class Journal:
    def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
        return "0"

    def finish(
        self, call_id: str, result: str, duration: float, permission: str | None = None
    ) -> None:
        json.loads(result)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    (root / "src" / "app.py").write_bytes(b"x = 1\n")
    (root / ".env").write_bytes(b"TOKEN=old\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "first commit")
    (root / "src" / "app.py").write_bytes(b"x = 2\n")
    (root / ".env").write_bytes(b"TOKEN=new-secret\n")
    (root / "notes.txt").write_bytes(b"new\n")
    return root


def _run(
    root: Path, name: str, restrictions: Restrictions | None = None, **args: Any
) -> ToolResult:
    runner = ToolRunner()
    register_git_tools(runner, Workspace(root, restrictions=restrictions))
    return runner.execute(name, args, Journal())


def test_status_diff_and_log(repo: Path) -> None:
    status = _run(repo, "git_status")
    assert status.ok, status.content
    lines = status.content.splitlines()
    assert lines[0].startswith("## main")
    assert " M src/app.py" in lines and "?? notes.txt" in lines
    diff = _run(repo, "git_diff")
    assert "-x = 1\n+x = 2" in diff.content
    assert "src/app.py" in _run(repo, "git_diff", stat_only=True).content
    assert _run(repo, "git_diff", staged=True).content == "(no changes)"
    log = _run(repo, "git_log", max_count=5)
    assert log.content.strip().endswith("Test: first commit")
    one_file = _run(repo, "git_status", path="src/app.py")
    assert "notes.txt" not in one_file.content and "src/app.py" in one_file.content


def test_secrets_are_not_shown(repo: Path) -> None:
    status = _run(repo, "git_status")
    diff = _run(repo, "git_diff")
    assert ".env" not in status.content and "new-secret" not in diff.content
    # Пользователь снял скрытие секретов — тогда видно.
    opened = _run(repo, "git_diff", Restrictions(secret_patterns=()))
    assert "new-secret" in opened.content


def test_not_a_repository_and_bounds(tmp_path: Path, repo: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _run(plain, "git_status").error == "agent.tool.not_git_repo"
    assert _run(repo, "git_log", path="..").error == "agent.access.parent"
    assert _run(repo, "git_log", max_count=0).error == "agent.tool.invalid_args"


def test_subfolder_grant_does_not_show_parent_repo(repo: Path) -> None:
    status = _run(repo / "src", "git_status")
    assert "app.py" in status.content and "notes.txt" not in status.content


def test_repo_config_cannot_run_programs(repo: Path) -> None:
    marker = repo.parent / "fsmonitor-ran"
    hook = f"{sys.executable} -c \"open(r'{marker}', 'w').close()\""
    if os.name == "nt":
        hook = hook.replace("\\", "/")
    git(repo, "config", "core.fsmonitor", hook)
    # Контроль: обычный git status действительно запускает эту программу.
    subprocess.run(["git", "status"], cwd=repo, env=ENV, capture_output=True)
    if not marker.exists():
        pytest.skip("this git does not run core.fsmonitor commands")
    marker.unlink()
    assert _run(repo, "git_status").ok
    assert not marker.exists()


def test_unsafe_repo_config_is_refused_unless_user_allows(repo: Path) -> None:
    git(repo, "config", "filter.evil.clean", "cat")
    refused = _run(repo, "git_status")
    assert refused.error == "agent.tool.git_unsafe_config"
    assert refused.data["params"]["key"] == "filter.evil.clean"
    assert _run(repo, "git_status", Restrictions(safe_git=False)).ok
    git(repo, "config", "--unset", "filter.evil.clean")
    git(repo, "config", "core.worktree", str(repo.parent))
    assert _run(repo, "git_log").error == "agent.tool.git_unsafe_config"
