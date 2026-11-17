"""4.8 AUTO-06: защита от неудачного выбора модели.

- **Проверка до запуска движка.** Если по плану модель не помещается даже во всю
  память компьютера («Не хватает VRAM/RAM» — тот же расчёт, что у движка: движок
  считает по свободной памяти, её не больше всей), загрузка сразу заканчивается
  понятной ошибкой `engine.no_memory` — без запуска процесса и ожидания.
- **Подсказки при нехватке памяти** — каждая проверена расчётом и несёт действие для
  кнопки интерфейса:
  `reduce_context` (наибольший контекст, при котором модель помещается в свободную
  сейчас память), `economy_mode`, `smaller_model` (установленная модель, которая
  помещается сейчас), `close_programs` (компьютер способен, но память занята).

Активная модель посреди ответа не меняется: загрузка и смена модели ждут слот общей
модели (`SharedLLM.exclusive`, 3.17) — это проверяют тесты 3.17 и 4.8.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_storage.repositories import Model, Models, NotFound

from ramvi_service import compat
from ramvi_service.engines import resolve_model
from ramvi_service.i18n import msg
from ramvi_service.model_fit import ModelFit
from ramvi_service.task_profile import TaskProfile

CONTEXTS = (16384, 8192, 4096, 2048)  # куда уменьшать контекст, от большего
BLOCKING = (compat.NO_VRAM, compat.NO_RAM)


def _fits_now(verdict: dict[str, Any]) -> bool:
    return verdict.get("status") in compat.RUNNABLE and bool((verdict.get("now") or {}).get("fits"))


class Guard:
    def __init__(
        self,
        fit: ModelFit,
        models: Models,
        app_dir: Path | None,
        profile: Callable[[], TaskProfile],
    ) -> None:
        self.fit, self.models, self.app_dir, self.profile = fit, models, app_dir, profile

    def _verdict(self, model: Model, task: TaskProfile | None = None) -> dict[str, Any]:
        path = resolve_model(model, self.app_dir).path
        return self.fit.for_file(path, model.id, model.id, task=task)

    def preflight(self, model: Model) -> tuple[str, str] | None:
        """(ключ, подробность) — не загружать: план не помещается во всю память компьютера."""
        verdict = self._verdict(model)
        if verdict.get("status") in BLOCKING:
            reasons = verdict.get("reasons") or [{}]
            return "engine.no_memory", str(reasons[0].get("code", ""))
        return None

    def suggestions(self, model_id: str) -> list[dict[str, Any]]:
        """Что можно сделать, если модели не хватает памяти; пусто — предложить нечего."""
        try:
            model = self.models.get(model_id)
        except NotFound:
            return []
        task = self.profile()
        current = self._verdict(model, task)
        found: list[dict[str, Any]] = []
        for context in CONTEXTS:
            if context < task.effective_context:
                smaller = replace(task, context=context)
                if _fits_now(self._verdict(model, smaller)):
                    found.append(
                        {
                            "action": "reduce_context",
                            "params": {"context": context},
                            "message": msg("guard.reduce_context", context=context).json(),
                        }
                    )
                    break
        if task.mode != "economy":
            economy = TaskProfile("economy", task.priority)
            if economy.effective_context < task.effective_context and _fits_now(
                self._verdict(model, economy)
            ):
                found.append(
                    {
                        "action": "economy_mode",
                        "params": {"mode": "economy"},
                        "message": msg("guard.economy_mode").json(),
                    }
                )
        others = []
        for other in self.models.all():
            if other.id == model.id:
                continue
            verdict = self._verdict(other, task)
            if _fits_now(verdict):
                size = (verdict.get("model") or {}).get("size_b") or 0.0
                others.append((size, other.id))
        for _, other_id in sorted(others, reverse=True)[:2]:  # самые крупные из помещающихся
            found.append(
                {
                    "action": "smaller_model",
                    "params": {"model_id": other_id},
                    "message": msg("guard.smaller_model", model_id=other_id).json(),
                }
            )
        if current.get("status") in compat.RUNNABLE and not _fits_now(current):
            found.append(
                {
                    "action": "close_programs",
                    "params": {},
                    "message": msg("guard.close_programs").json(),
                }
            )
        return found
