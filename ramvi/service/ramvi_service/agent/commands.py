"""6.2/6.5: `run_command` — предложение команды и её исполнение после подтверждения.

Обработчик инструмента команду **не запускает**: он разбирает её, проверяет и
возвращает `CommandProposal`. Запуск — `execute_command` после подтверждения
пользователя (6.4), в дочернем процессе с лимитами времени и вывода (`process.py`).

Команда идёт без оболочки: `&&`, `|`, `>` и т. п. не работают и отклоняются, чтобы
модель не думала, что они работают. Разбор назначения команды — подсказка для
пользователя и блокировка очевидного, а не изоляция: команда работает с правами
пользователя, поэтому право `terminal` по умолчанию выключено, а каждая команда
требует подтверждения.

* разрушительные (удаление, `git push`/`reset --hard`/`clean`, повышение прав,
  форматирование, оболочки и `python -c` — произвольный код) заблокированы, пока
  пользователь не включил `ToolAccess.allow_destructive_commands`;
* сетевые (`curl`, `git fetch`, `pip install` …) требуют права `network`;
* переменные окружения с секретами (`*TOKEN*`, `*API_KEY*` …) в процесс не передаются.
"""

import os
import shlex
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramvi_service.agent.process import ProcessLimits, run_process
from ramvi_service.agent.tools import (
    SECRET_KEY,
    ToolAccess,
    ToolError,
    ToolResult,
    ToolRunner,
    ToolSpec,
)
from ramvi_service.agent.workspace import Workspace

MAX_COMMAND_CHARS = 4000
MAX_TIMEOUT = 600
SHELL_OPERATORS = frozenset({"&&", "||", "|", ";", ">", ">>", "<", "&", "2>", "2>&1", "|&"})
CMD_METACHARS = set("&|<>^%!")

SHELLS = frozenset(
    {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "cmd", "powershell", "pwsh"}
)
INLINE_CODE = {
    "python": {"-c"},
    "python3": {"-c"},
    "py": {"-c"},
    "node": {"-e", "--eval", "-p", "--print"},
    "ruby": {"-e"},
    "perl": {"-e", "-E"},
    "php": {"-r"},
    "deno": {"eval"},
    "bun": {"-e", "--eval"},
}
DELETING = frozenset(
    {"rm", "rmdir", "del", "erase", "rd", "unlink", "shred", "remove-item", "rimraf", "mv", "move"}
)
SYSTEM = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "runas",
        "mkfs",
        "dd",
        "format",
        "diskpart",
        "fdisk",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "chmod",
        "chown",
        "icacls",
        "takeown",
        "reg",
        "bcdedit",
        "kill",
        "killall",
        "pkill",
        "taskkill",
        "launchctl",
        "systemctl",
        "sc",
    }
)
GIT_ALWAYS = frozenset(
    {"push", "clean", "rm", "restore", "rebase", "filter-branch", "filter-repo", "update-ref"}
)
NETWORK = frozenset(
    {"curl", "wget", "ssh", "scp", "sftp", "rsync", "ftp", "nc", "ncat", "telnet", "npx"}
)
GIT_NETWORK = frozenset({"clone", "fetch", "pull", "push", "ls-remote", "submodule"})
PACKAGE_MANAGERS = frozenset(
    {"pip", "pip3", "uv", "poetry", "npm", "yarn", "pnpm", "cargo", "go", "gem", "composer"}
)
PACKAGE_NETWORK = frozenset(
    {"install", "add", "update", "upgrade", "download", "publish", "get", "fetch", "sync"}
)


@dataclass(frozen=True)
class CommandProposal:
    """Проверенная команда, ожидающая подтверждения."""

    command: str
    argv: tuple[str, ...]
    cwd: Path
    display_cwd: str
    timeout: float
    destructive: str | None  # причина, если команда разрушительная
    network: bool


def _name(program: str) -> str:
    name = Path(program.strip("\"'")).name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def classify(argv: list[str]) -> tuple[str | None, bool]:
    """Причина «разрушительная» (или None) и признак сети для разобранной команды."""
    program = _name(argv[0])
    args = [a.lower() for a in argv[1:]]
    if program in ("python", "python3", "py") and args[:2] in (["-m", "pip"], ["-m", "pip3"]):
        program, args = "pip", args[2:]
    first = args[0] if args else ""
    network = (
        program in NETWORK
        or (program == "git" and first in GIT_NETWORK)
        or (program in PACKAGE_MANAGERS and first in PACKAGE_NETWORK)
    )
    if program in SHELLS:
        return "shell", network
    if program in INLINE_CODE and INLINE_CODE[program] & set(args):
        return "inline_code", network
    if program in DELETING:
        return "deletes_files", network
    if program in SYSTEM or program.startswith("mkfs"):
        return "system", network
    if program == "git":
        destructive = (
            first in GIT_ALWAYS
            or (first == "reset" and "--hard" in args)
            or (first == "checkout" and ("--" in args or "." in args or "-f" in args))
            or (first == "stash" and ("drop" in args or "clear" in args))
            or (first in ("branch", "tag") and {"-d", "--delete"} & set(args))
            or (first == "reflog" and ("expire" in args or "delete" in args))
            or (first == "gc" and any(a.startswith("--prune") for a in args))
        )
        if destructive:
            return "git_history", network
    return None, network


def clean_env() -> dict[str, str]:
    """Окружение процесса без переменных с секретами."""
    return {k: v for k, v in os.environ.items() if not SECRET_KEY.search(k)}


def _split(command: str) -> list[str]:
    windows = os.name == "nt"
    try:
        parts = shlex.split(command, posix=not windows)
    except ValueError:
        raise ToolError("agent.tool.bad_command") from None
    if windows:
        parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] == '"' else p for p in parts]
    return parts


class CommandTools:
    """Обработчик `run_command` одной рабочей папки."""

    def __init__(self, workspace: Workspace, access: ToolAccess) -> None:
        self.ws, self.access = workspace, access

    def _program(self, argv: list[str], cwd: Path) -> list[str]:
        program = argv[0]
        if os.path.isabs(program):  # как поиск в PATH: запуск программы, а не доступ к файлу
            if not os.path.isfile(program):
                raise ToolError("agent.tool.command_not_found", program=program)
            resolved = program
        elif "/" in program or "\\" in program:
            relative = os.path.relpath(cwd / program, self.ws.root)
            target = self.ws.resolve(relative)
            if not target.is_file():
                raise ToolError("agent.tool.command_not_found", program=program)
            resolved = str(target)
        else:
            found = shutil.which(program, path=clean_env().get("PATH"))
            if found is None:
                raise ToolError("agent.tool.command_not_found", program=program)
            resolved = found
        if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
            # .cmd/.bat исполняет только cmd.exe — тогда спецсимволы cmd запрещены.
            if any(CMD_METACHARS & set(arg) for arg in argv):
                raise ToolError("agent.tool.shell_syntax")
            return ["cmd.exe", "/d", "/s", "/c", resolved, *argv[1:]]
        return [resolved, *argv[1:]]

    def run_command(self, args: dict[str, Any]) -> ToolResult:
        command = args["command"]
        argv = _split(command)
        if not argv:
            raise ToolError("agent.tool.bad_command")
        if any(part in SHELL_OPERATORS for part in argv):
            raise ToolError("agent.tool.shell_syntax")
        cwd = self.ws.resolve(args.get("cwd", "."))
        if not cwd.is_dir():
            raise ToolError("agent.tool.not_directory", path=args.get("cwd", "."))
        destructive, network = classify(argv)
        if destructive and not self.access.allow_destructive_commands:
            raise ToolError("agent.tool.destructive_blocked", reason=destructive)
        if network:
            self.ws.require("network")
        launch = self._program(argv, cwd)
        proposal = CommandProposal(
            command=command,
            argv=tuple(launch),
            cwd=cwd,
            display_cwd=self.ws.relative(cwd),
            timeout=float(args.get("timeout", 120)),
            destructive=destructive,
            network=network,
        )
        content = f"Proposed command in {proposal.display_cwd}: {command} (awaiting user approval)"
        return ToolResult(True, content, {"proposal": proposal})


def execute_command(
    proposal: CommandProposal,
    *,
    max_output_bytes: int = 64 * 1024,
    cancel: threading.Event | None = None,
) -> ToolResult:
    """Запустить подтверждённую команду; вывод и ошибки — в результат для модели."""
    limits = ProcessLimits(timeout=proposal.timeout, max_output_bytes=max_output_bytes)
    try:
        result = run_process(
            list(proposal.argv), proposal.cwd, limits, env=clean_env(), cancel=cancel
        )
    except FileNotFoundError:
        return ToolResult.failure("agent.tool.command_not_found", program=proposal.argv[0])
    except OSError as error:
        return ToolResult.failure("agent.tool.command_failed", error=type(error).__name__)
    status = (
        f"timed out after {proposal.timeout:g} s"
        if result.timed_out
        else "cancelled"
        if result.cancelled
        else f"exit code {result.exit_code}"
    )
    parts = [status]
    if result.stdout:
        parts += ["--- stdout ---", result.stdout.rstrip("\n")]
    if result.stderr:
        parts += ["--- stderr ---", result.stderr.rstrip("\n")]
    if result.truncated:
        parts.append(f"[output truncated to {max_output_bytes} bytes per stream]")
    data = {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "duration": result.duration,
    }
    code = "agent.tool.timeout" if result.timed_out else None
    if result.cancelled:
        code = "agent.tool.cancelled"
    return ToolResult(code is None, "\n".join(parts), data, code, result.truncated)


def register_command_tools(
    runner: ToolRunner, workspace: Workspace, access: ToolAccess
) -> CommandTools:
    """Зарегистрировать `run_command` (право `terminal`, риск critical)."""
    tools = CommandTools(workspace, access)
    schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_COMMAND_CHARS,
                "description": "Program and arguments, e.g. 'pytest -q tests'. No shell: "
                "&&, |, > and redirects are not supported.",
            },
            "cwd": {"type": "string", "maxLength": 4096, "description": "Folder to run in"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT},
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    runner.register(
        ToolSpec(
            "run_command",
            "Run tests, linters, builds or diagnostics. The user must approve each command.",
            schema,
            "terminal",
            "command",
            "critical",
        ),
        tools.run_command,
    )
    return tools
