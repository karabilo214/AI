"""6.2 (AG-02): git-инструменты только для чтения — `git_status`, `git_diff`, `git_log`.

Git сам умеет запускать программы из настроек репозитория (fsmonitor, фильтры
clean/smudge, внешний diff, textconv, хуки). Чужой репозиторий не должен получить
так исполнение кода, поэтому:

* системные и пользовательские настройки git отключены (`GIT_CONFIG_NOSYSTEM`,
  `GIT_CONFIG_GLOBAL` = пусто), fsmonitor, хуки, внешний diff и textconv выключены
  в командной строке, пейджера и запроса пароля нет, индекс не перезаписывается;
* если в настройках самого репозитория есть фильтры, include или `core.worktree`,
  инструмент отказывает (`agent.tool.git_unsafe_config`), пока пользователь не снял
  ограничение `Restrictions.safe_git`.

Вывод ограничен выбранной папкой (pathspec), секреты исключены pathspec-ами; путь
проходит `Workspace.resolve`. Это эффект `read`: подтверждение не нужно.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from ramvi_service.agent.process import ProcessLimits, run_process
from ramvi_service.agent.tools import ToolError, ToolResult, ToolRunner, ToolSpec
from ramvi_service.agent.workspace import Workspace

GIT_LIMITS = ProcessLimits(timeout=30.0, max_output_bytes=512 * 1024)
_HARDENING = [
    "--no-pager",
    "-c",
    "core.fsmonitor=false",
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "diff.external=",
    "-c",
    "color.ui=false",
    "-c",
    "core.quotePath=false",
    "-c",
    "log.showSignature=false",
]
_UNSAFE_PREFIXES = ("filter.", "include.", "includeif.")
_UNSAFE_KEYS = ("core.worktree",)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    for key in [k for k in env if k.startswith("GIT_")]:
        del env[key]  # GIT_DIR, GIT_WORK_TREE и т. п. не должны уводить из папки
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",
        GIT_PAGER="cat",
        LC_ALL="C",
    )
    return env


class GitTools:
    """Обработчики git-инструментов одной рабочей папки."""

    def __init__(self, workspace: Workspace, limits: ProcessLimits = GIT_LIMITS) -> None:
        self.ws, self.limits = workspace, limits

    def _git(self, cwd: Path, *args: str) -> str:
        executable = shutil.which("git")
        if executable is None:
            raise ToolError("agent.tool.git_missing")
        try:
            result = run_process([executable, *_HARDENING, *args], cwd, self.limits, env=_env())
        except OSError:
            raise ToolError("agent.tool.git_missing") from None
        if result.timed_out:
            raise ToolError("agent.tool.timeout", seconds=self.limits.timeout)
        if result.exit_code != 0:
            stderr = result.stderr.lower()
            if "not a git repository" in stderr or "inside a git repository" in stderr:
                raise ToolError("agent.tool.not_git_repo")
            raise ToolError("agent.tool.git_failed", stderr=result.stderr[-500:])
        return result.stdout + ("\n[output truncated]" if result.truncated else "")

    def _check_config(self, cwd: Path) -> None:
        if not self.ws.restrictions.safe_git:
            return
        listing = self._git(cwd, "config", "--local", "--no-includes", "--name-only", "--list")
        for key in (line.strip().lower() for line in listing.splitlines()):
            if key.startswith(_UNSAFE_PREFIXES) or key in _UNSAFE_KEYS:
                raise ToolError("agent.tool.git_unsafe_config", key=key)

    def _target(self, path: str) -> tuple[Path, list[str]]:
        """Папка запуска и pathspec: выбранный путь без секретов."""
        target = self.ws.resolve(path)
        rel = self.ws.relative(target)
        if not os.path.isabs(rel):  # внутри папки: пути в выводе — от её корня
            cwd, spec = self.ws.root, rel
        elif target.is_dir():  # полный доступ, путь вне папки
            cwd, spec = target, "."
        else:
            cwd, spec = target.parent, target.name
        excludes: list[str] = []
        for pattern in self.ws.restrictions.secret_patterns:
            excludes += [
                f":(exclude,glob,icase)**/{pattern}",
                f":(exclude,glob,icase)**/{pattern}/**",
            ]
        self._check_config(cwd)
        return cwd, ["--", spec, *excludes]

    def git_status(self, args: dict[str, Any]) -> ToolResult:
        cwd, spec = self._target(args.get("path", "."))
        out = self._git(cwd, "status", "--short", "--branch", "--untracked-files=normal", *spec)
        return ToolResult(True, out or "(clean)", {"clean": len(out.splitlines()) <= 1})

    def git_diff(self, args: dict[str, Any]) -> ToolResult:
        cwd, spec = self._target(args.get("path", "."))
        flags = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
        if args.get("staged"):
            flags.append("--cached")
        if args.get("stat_only"):
            flags.append("--stat")
        out = self._git(cwd, *flags, *spec)
        return ToolResult(True, out or "(no changes)", {"empty": not out})

    def git_log(self, args: dict[str, Any]) -> ToolResult:
        cwd, spec = self._target(args.get("path", "."))
        count = str(args.get("max_count", 20))
        fmt = "--format=%h %ad %an: %s"
        flags = ["log", "--no-color", "--date=short", fmt, "-n", count]
        if args.get("stat"):
            flags.append("--stat")
        out = self._git(cwd, *flags, *spec)
        return ToolResult(True, out or "(no commits)", {"empty": not out})


def register_git_tools(runner: ToolRunner, workspace: Workspace) -> GitTools:
    """Зарегистрировать git-инструменты одной рабочей папки."""
    tools = GitTools(workspace)
    path = {"type": "string", "maxLength": 4096, "description": "Folder or file; default all"}

    def schema(extra: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": path, **extra},
            "additionalProperties": False,
        }

    runner.register(
        ToolSpec(
            "git_status", "Changed, staged and untracked files (git status).", schema({}), "read"
        ),
        tools.git_status,
    )
    runner.register(
        ToolSpec(
            "git_diff",
            "Current uncommitted changes as a diff; staged=true for the index.",
            schema({"staged": {"type": "boolean"}, "stat_only": {"type": "boolean"}}),
            "read",
        ),
        tools.git_diff,
    )
    runner.register(
        ToolSpec(
            "git_log",
            "Recent commits: short hash, date, author, subject.",
            schema(
                {
                    "max_count": {"type": "integer", "minimum": 1, "maximum": 200},
                    "stat": {"type": "boolean"},
                }
            ),
            "read",
        ),
        tools.git_log,
    )
    return tools
