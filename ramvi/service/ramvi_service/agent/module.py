"""Lazy lifecycle marker for the installed Agent section; execution stays in Core."""

from typing import Any


class AgentModule:
    def initialize(self, context: dict[str, Any]) -> None:
        pass

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def create() -> AgentModule:
    return AgentModule()
