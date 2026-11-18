"""AG-08: сменная стратегия. Только предложения; исполнение и права — в runtime.

Память/контроллер могут обернуть Strategy.decide и читать AgentJournal.experience.
Прошлый опыт — данные, не разрешения. Веса модели автоматически не меняются.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON-строка, как её прислала модель


@dataclass(frozen=True)
class ModelReply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class Strategy(Protocol):
    """Выбор следующего действия (6.8): план, вызовы инструментов или итог."""

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
        max_tokens: int,
    ) -> ModelReply: ...


Complete = Callable[[dict[str, Any], threading.Event], dict[str, Any]]


class LLMStrategy:
    """MVP: решения принимает основная LLM (OpenAI-совместимый ответ с tool_calls)."""

    def __init__(self, complete: Complete, extra: dict[str, Any] | None = None) -> None:
        self.complete, self.extra = complete, dict(extra or {})

    def decide(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
        max_tokens: int,
    ) -> ModelReply:
        request: dict[str, Any] = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            **self.extra,
        }
        if tools:
            request["tools"] = tools
        response = self.complete(request, cancel)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            arguments = function.get("arguments", "")
            calls.append(
                ToolCall(
                    str(raw.get("id") or f"call_{index}"),
                    str(function.get("name", "")),
                    arguments if isinstance(arguments, str) else json.dumps(arguments),
                )
            )
        usage = response.get("usage") or {}
        return ModelReply(
            content=message.get("content") or "",
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        )
