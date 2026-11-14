"""Только тестовый модуль контракта; не обязательная функция продукта."""

import os
import time
from typing import Any


class FixtureModule:
    def initialize(self, context: dict[str, Any]) -> None:
        self.services = context["services"]
        self.state = context["checkpoint"].get("value", 0)
        self.module_id = context["module_id"]

    def activate(self) -> None:
        self.state += 1

    def deactivate(self) -> None:
        pass

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        self.state += 1

    def dispose(self) -> None:
        pass

    def checkpoint(self) -> dict[str, Any]:
        return {"value": self.state}

    def value(self) -> dict[str, Any]:
        return {"value": self.state}

    def settings(self, value: str) -> dict[str, Any]:
        self.services.call("settings.set", {"value": {"module": self.module_id, "value": value}})
        return dict(self.services.call("settings.get", {}))

    def event(self, response: str) -> dict[str, Any]:
        return dict(
            self.services.call(
                "events.emit", {"response_id": response, "name": "tick", "payload": {"value": 7}}
            )
        )

    def crash(self) -> dict[str, Any]:
        os._exit(12)

    def hang(self) -> dict[str, Any]:
        time.sleep(60)
        return {}


def create() -> FixtureModule:
    return FixtureModule()
