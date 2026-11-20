"""Module-owned speech policy and contextual phrase rotation. No providers or devices."""

import copy
from typing import Any

DEFAULTS = {"mode": "sentences", "buffer": 2, "reactions": True}
# Categories are selected by the contextual LLM start record, never by keyword guessing.
PHRASES = {
    "consider": (
        ["Дай подумать.", "Сейчас разберёмся.", "Подумаю над этим."],
        ["Let me think.", "Let's work through this.", "I'll give this some thought."],
    ),
    "explain": (
        ["Попробую объяснить понятно.", "Давай разберём по порядку.", "Сейчас подберу объяснение."],
        [
            "Let me find a clear explanation.",
            "Let's take this step by step.",
            "I'll break it down.",
        ],
    ),
    "calculate": (
        ["Сейчас посчитаю.", "Дай разобраться с числами.", "Прикину по шагам."],
        [
            "Let me work out the numbers.",
            "I'll calculate that.",
            "Let's work through the calculation.",
        ],
    ),
    "write": (
        ["Подберу формулировку.", "Сейчас набросаю вариант.", "Дай собраться с мыслями."],
        ["Let me find the right words.", "I'll draft a version.", "Let me gather my thoughts."],
    ),
    "code": (
        ["Подумаю над решением.", "Разберу задачу по шагам.", "Сейчас вникну в задачу."],
        [
            "Let me think through the solution.",
            "I'll work through the problem.",
            "Let me look at the approach.",
        ],
    ),
    "compare": (
        ["Сопоставлю варианты.", "Дай взвесить различия.", "Сейчас разберу, что к чему."],
        [
            "Let me compare the options.",
            "I'll weigh the differences.",
            "Let's look at the tradeoffs.",
        ],
    ),
    "plan": (
        ["Прикину порядок действий.", "Давай наметим шаги.", "Сейчас соберу план."],
        ["Let me map out the steps.", "I'll put a plan together.", "Let's work out the order."],
    ),
    "support": (
        ["Я тебя слышу.", "Давай не будем спешить.", "Я рядом."],
        ["I hear you.", "Let's take our time.", "I'm here."],
    ),
}


def settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("invalid delivery settings")
    result = {**DEFAULTS, **value}
    if (
        not isinstance(result["mode"], str)
        or result["mode"] not in {"sentences", "complete"}
        or type(result["buffer"]) is not int
        or result["buffer"] not in {1, 2, 3}
        or type(result["reactions"]) is not bool
    ):
        raise ValueError("invalid delivery policy")
    return result


class ReactionBank:
    def __init__(self, recent: Any = None) -> None:
        self.recent: list[str] = []
        if recent is not None:
            if (
                not isinstance(recent, list)
                or len(recent) > 24
                or any(not isinstance(x, str) or len(x) > 40 for x in recent)
            ):
                raise ValueError("invalid reaction history")
            self.recent = copy.deepcopy(recent)

    def choose(self, category: str, language: str) -> tuple[str, str] | None:
        if category not in PHRASES or language not in {"ru", "en"}:
            return None
        choices = [
            (f"{category}:{language}:{i}", phrase)
            for i, phrase in enumerate(PHRASES[category][language == "en"])
        ]
        for item in choices:
            if item[0] not in self.recent:
                return item
        return min(choices, key=lambda item: self.recent.index(item[0]))

    def used(self, identifier: str) -> None:
        self.recent = [x for x in self.recent if x != identifier]
        self.recent.append(identifier)
        self.recent = self.recent[-24:]
