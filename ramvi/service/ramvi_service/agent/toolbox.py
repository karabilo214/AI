"""Полный набор инструментов агента для одной рабочей папки."""

from typing import Any

from ramvi_service.agent.commands import register_command_tools
from ramvi_service.agent.files import FileLimits, register_file_tools
from ramvi_service.agent.git import register_git_tools
from ramvi_service.agent.tools import ToolAccess, ToolRunner
from ramvi_service.agent.workspace import Workspace


def build_tools(
    workspace: Workspace,
    access: ToolAccess | None = None,
    limits: FileLimits | None = None,
    gate: Any = None,
) -> ToolRunner:
    """Файловые, git- и командные инструменты с общими правами и настройками доступа.

    `gate` — `ApprovalGate`: с ним подтверждённые записи и команды исполняются (6.4).
    """
    runner = ToolRunner(workspace, access, gate)
    register_file_tools(runner, workspace, limits)
    register_git_tools(runner, workspace)
    register_command_tools(runner, workspace, runner.access)
    return runner
