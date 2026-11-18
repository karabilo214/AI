"""6.6 (AG-06): готова ли модель к работе агентом — проверка до первого запроса.

* **инструменты** — движок/провайдер умеет вызовы инструментов для этой модели
  (карта возможностей движка, `catalog.support.capabilities`); нет — агент не
  запускается (`agent.ready.no_tools`);
* **контекст** — хватает на правила, описания инструментов и работу. Требование
  считается по настоящему размеру правил и схем (≈ 3 символа на токен, с запасом) плюс
  место под ответ и содержимое файлов. На проверке 6.1 при контексте 2048 ответ
  обрезался на втором шаге, при 8192 — задачи решались. Мало — `agent.ready.context`
  (контекст поднимается в профиле задачи) или `agent.ready.model_context` (предел
  самой модели);
* **размер** — модель меньше `small_model_b` млрд параметров может плохо планировать
  действия: предупреждение (`agent.ready.small_model`), не запрет.

Ошибки не дают запустить агента — с ними он всё равно не сможет работать;
предупреждения показываются пользователю и не мешают.
"""

import json
from dataclasses import dataclass
from typing import Any

CHARS_PER_TOKEN = 3  # с запасом: у Qwen3 на английском ≈ 4, на русском и коде меньше
WORK_TOKENS = 4096  # ответ модели и содержимое прочитанных файлов
SMALL_MODEL_B = 7.0


@dataclass(frozen=True)
class ModelFacts:
    """Что известно о модели, с которой будет работать агент."""

    tools: bool  # вызовы инструментов поддержаны для этой модели
    context: int  # контекст, с которым запущен движок (профиль задачи)
    model_context: int | None = None  # предел самой модели (заголовок GGUF)
    size_b: float | None = None  # миллиарды параметров


@dataclass(frozen=True)
class Readiness:
    errors: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_context: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "required_context": self.required_context,
        }


def required_context(system_prompt: str, tools: list[dict[str, Any]]) -> int:
    """Контекст, нужный агенту: правила и схемы инструментов плюс место для работы."""
    chars = len(system_prompt) + len(json.dumps(tools, ensure_ascii=False))
    return -(-chars // CHARS_PER_TOKEN) + WORK_TOKENS


@dataclass(frozen=True)
class ReadinessCheck:
    small_model_b: float = SMALL_MODEL_B

    def check(
        self, facts: ModelFacts, system_prompt: str, tools: list[dict[str, Any]]
    ) -> Readiness:
        need = required_context(system_prompt, tools)
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        if not facts.tools:
            errors.append({"code": "agent.ready.no_tools", "params": {}})
        if facts.model_context is not None and facts.model_context < need:
            errors.append(
                {
                    "code": "agent.ready.model_context",
                    "params": {"have": facts.model_context, "need": need},
                }
            )
        elif facts.context < need:
            errors.append(
                {"code": "agent.ready.context", "params": {"have": facts.context, "need": need}}
            )
        if facts.size_b is not None and facts.size_b < self.small_model_b:
            warnings.append(
                {
                    "code": "agent.ready.small_model",
                    "params": {"size_b": facts.size_b, "recommended_b": self.small_model_b},
                }
            )
        return Readiness(tuple(errors), tuple(warnings), need)


def facts_from_fit(
    verdict: dict[str, Any], capabilities: tuple[str, ...] | list[str]
) -> ModelFacts:
    """Факты из оценки модели (`ModelFit.for_file`) и карты возможностей движка."""
    model = verdict.get("model") or {}
    limit = None
    for reason in (verdict.get("task") or {}).get("reasons", []):
        if reason.get("code") == "task.reason.context":
            limit = reason.get("params", {}).get("model")
    return ModelFacts(
        tools="tools" in capabilities,
        context=int(verdict.get("context") or 0),
        model_context=limit,
        size_b=model.get("size_b"),
    )
