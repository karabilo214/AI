"""4.5 AUTO-02: профиль задачи — режим работы и приоритет пользователя.

Профиль меняет только рекомендации и контекст, с которым грузится модель;
установленные файлы не трогаются. Хранится в настройках (`settings.json`:
`task_mode`, `task_priority`, `task_context`).

Режим задаёт контекст по умолчанию (от него зависит KV-кэш, а значит план и
скорость) и возможности, без которых модель задаче не подходит. Приоритет
используется ранжированием (4.6).
"""

from dataclasses import dataclass
from typing import Any

PRIORITIES = ("balanced", "quality", "speed", "ram", "ttft")
MIN_CONTEXT, MAX_CONTEXT = 512, 131072


@dataclass(frozen=True)
class Mode:
    context: int  # токенов контекста по умолчанию
    needs: tuple[str, ...] = ()  # возможности записи каталога (`capabilities`)


MODES: dict[str, Mode] = {
    "chat": Mode(2048),  # как прежний контекст чата (ModelConfig.maxt)
    "code": Mode(8192),
    "agent": Mode(8192, ("tools",)),
    "long_context": Mode(32768),
    "json": Mode(4096, ("json",)),
    "economy": Mode(2048),
}


@dataclass(frozen=True)
class TaskProfile:
    mode: str = "chat"
    priority: str = "balanced"
    context: int = 0  # 0 — по режиму

    @property
    def effective_context(self) -> int:
        return self.context or MODES[self.mode].context

    @property
    def needs(self) -> tuple[str, ...]:
        return MODES[self.mode].needs

    def problem(self) -> tuple[str, dict[str, Any]] | None:
        """Ключ ошибки и параметры; None — профиль верный."""
        if self.mode not in MODES:
            return "settings.bad_task_mode", {"allowed": ", ".join(MODES)}
        if self.priority not in PRIORITIES:
            return "settings.bad_task_priority", {"allowed": ", ".join(PRIORITIES)}
        bad_type = isinstance(self.context, bool) or not isinstance(self.context, int)
        if bad_type or (self.context and not MIN_CONTEXT <= self.context <= MAX_CONTEXT):
            return "settings.bad_task_context", {"low": MIN_CONTEXT, "high": MAX_CONTEXT}
        return None

    @property
    def ranking(self) -> str:
        """Приоритет ранжирования (4.6): режим «экономия» при «балансе» — экономия RAM."""
        if self.mode == "economy" and self.priority == "balanced":
            return "ram"
        return self.priority

    def json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "priority": self.priority,
            "context": self.effective_context,
            "needs": list(self.needs),
        }
