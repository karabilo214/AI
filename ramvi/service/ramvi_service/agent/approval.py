"""6.4 (AG-04): подтверждение действий агента перед записью и запуском команд.

Инструменты с эффектом `write`/`command` возвращают предложение (diff или команду).
`ApprovalGate.handle` решает, что с ним делать:

* подтверждение выключено пользователем (`ToolAccess.confirm_*`) — применить сразу;
* действует разрешённая серия — применить и уменьшить остаток;
* иначе спросить пользователя (`ask`, показывает diff/команду) и ждать решения:
  - `approve` — применить; `series=N` — ещё N следующих действий того же вида без
    вопроса (не больше `ToolAccess.max_series`);
  - `deny` — не применять; причина (поправка задачи) уходит модели;
  - `stop` — не применять и остановить запуск (`ToolRunner` пишет журнал и поднимает
    `StopRequested`).

Пока решения нет, ничего не меняется: без подтверждения исходные файлы остаются
прежними. Разрушительная команда спрашивается всегда, даже внутри серии, пока
пользователь не выключил `confirm_destructive`.
"""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.commands import CommandProposal, execute_command
from ramvi_service.agent.files import PatchProposal, WriteProposal
from ramvi_service.agent.tools import ToolAccess, ToolError, ToolResult, ToolSpec
from ramvi_service.agent.workspace import AccessDenied

Kind = Literal["write", "command"]


@dataclass(frozen=True)
class ApprovalRequest:
    """То, что видит пользователь: вид действия, файлы, diff или команда."""

    id: str
    tool: str
    kind: Kind
    title: str
    diff: str = ""
    command: str = ""
    cwd: str = ""
    files: tuple[str, ...] = ()
    destructive: str | None = None
    network: bool = False


@dataclass(frozen=True)
class Decision:
    action: Literal["approve", "deny", "stop"]
    series: int = 0  # approve: ещё столько следующих действий того же вида без вопроса
    reason: str = ""  # deny: поправка задачи для модели


Ask = Callable[[ApprovalRequest], Decision]


@dataclass
class _Series:
    remaining: dict[str, int] = field(default_factory=lambda: {"write": 0, "command": 0})


class ApprovalGate:
    """Подтверждение и применение предложений одного запуска агента."""

    def __init__(
        self,
        ask: Ask,
        changes: ChangeStore,
        access: ToolAccess | None = None,
        *,
        cancel: threading.Event | None = None,
    ) -> None:
        self.ask, self.changes = ask, changes
        self.access = access or ToolAccess()
        self.cancel = cancel
        self._series = _Series()

    @staticmethod
    def request(spec: ToolSpec, proposal: object) -> ApprovalRequest:
        rid = uuid.uuid4().hex
        if isinstance(proposal, CommandProposal):
            return ApprovalRequest(
                rid,
                spec.name,
                "command",
                proposal.command,
                command=proposal.command,
                cwd=proposal.display_cwd,
                destructive=proposal.destructive,
                network=proposal.network,
            )
        if isinstance(proposal, WriteProposal | PatchProposal):
            writes = proposal.files if isinstance(proposal, PatchProposal) else (proposal,)
            names = tuple(w.display for w in writes)
            return ApprovalRequest(
                rid, spec.name, "write", ", ".join(names), diff=proposal.diff, files=names
            )
        raise TypeError(f"unsupported proposal: {type(proposal).__name__}")

    def _needs_question(self, request: ApprovalRequest) -> bool:
        if request.kind == "command" and request.destructive and self.access.confirm_destructive:
            return True
        confirm = (
            self.access.confirm_writes if request.kind == "write" else self.access.confirm_commands
        )
        return confirm and self._series.remaining[request.kind] <= 0

    def decide(self, request: ApprovalRequest) -> tuple[Decision, str]:
        """Решение и отметка для журнала: `auto` / `approved` / `denied`."""
        if not self._needs_question(request):
            confirm = (
                self.access.confirm_writes
                if request.kind == "write"
                else self.access.confirm_commands
            )
            if confirm:  # в пределах разрешённой серии
                self._series.remaining[request.kind] -= 1
                return Decision("approve"), "approved"
            return Decision("approve"), "auto"
        if self.cancel is not None and self.cancel.is_set():
            return Decision("stop"), "denied"
        decision = self.ask(request)
        if self.cancel is not None and self.cancel.is_set():
            return Decision("stop"), "denied"
        if decision.action != "approve":
            return decision, "denied"
        series = max(0, min(decision.series, self.access.max_series))
        self._series.remaining[request.kind] = series
        return decision, "approved"

    def handle(self, spec: ToolSpec, proposal: object) -> tuple[ToolResult, str]:
        """Спросить (если нужно) и применить; вернуть результат и отметку для журнала."""
        request = self.request(spec, proposal)
        decision, permission = self.decide(request)
        if decision.action == "stop":
            return ToolResult.failure("agent.approval.stopped"), "denied"
        if decision.action == "deny":
            params = {"reason": decision.reason} if decision.reason else {}
            return ToolResult.failure("agent.approval.denied", **params), permission
        if isinstance(proposal, CommandProposal):
            return execute_command(proposal, cancel=self.cancel), permission
        assert isinstance(proposal, WriteProposal | PatchProposal)
        try:
            record = self.changes.apply(proposal)
        except (ToolError, AccessDenied) as error:
            return ToolResult.failure(error.code, **error.params), permission
        names = ", ".join(f.display for f in record.files)
        content = f"Applied change to {names} (change {record.id})."
        return ToolResult(True, content, {"change_id": record.id}), permission
