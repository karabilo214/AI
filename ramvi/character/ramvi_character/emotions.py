"""Emotional dynamics from structured appraisals, not phrase-to-expression rules."""

import copy
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ramvi_character.personality import number

DEFINITIONS = Path(__file__).parent / "definitions" / "emotions.json"
ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
THRESHOLD = 0.025


def definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"id", "expression", "half_life_s", "events"}:
        raise ValueError("invalid emotional state definition")
    for key in ("id", "expression"):
        if not isinstance(value[key], str) or not ID.fullmatch(value[key]):
            raise ValueError("invalid state or expression id")
    if value["id"] == "neutral":
        raise ValueError("neutral is the recovery baseline")
    events = value["events"]
    if not isinstance(events, dict) or len(events) > 8:
        raise ValueError("invalid emotional event affinities")
    for event in events:
        if not isinstance(event, str) or not ID.fullmatch(event):
            raise ValueError("invalid event id")
    return {
        "id": value["id"],
        "expression": value["expression"],
        "half_life_s": number(value["half_life_s"], 1, 86400),
        "events": {k: number(v, -1, 1) for k, v in events.items()},
    }


class EmotionalState:
    def __init__(
        self, saved: dict[str, Any] | None = None, *, clock: Callable[[], float] = time.time
    ) -> None:
        raw = json.loads(DEFINITIONS.read_text(encoding="utf-8"))
        if raw["schema"] != "ramvi.character.emotions/1":
            raise ValueError("unsupported emotion definitions")
        self._clock = clock
        self._definitions = {item["id"]: definition(item) for item in raw["states"]}
        self._custom: list[dict[str, Any]] = []
        self._values = dict.fromkeys(self._definitions, 0.0)
        self._updated = number(clock(), 0, 1e12)
        if saved is not None:
            self._restore(saved)

    def identifiers(self) -> list[str]:
        return list(self._definitions)

    def expression(self, emotion: str) -> str:
        """Лицо набора для эмоции ответа (anger → furious); неизвестная — нейтральное."""
        item = self._definitions.get(emotion)
        return item["expression"] if item is not None else "neutral"

    def add(self, value: dict[str, Any]) -> None:
        item = definition(value)
        if item["id"] in self._definitions or len(self._custom) >= 16:
            raise ValueError("state already exists or custom state limit reached")
        self._advance()
        self._definitions[item["id"]] = item
        self._values[item["id"]] = 0.0
        self._custom.append(item)

    def _advance(self) -> None:
        now = number(self._clock(), 0, 1e12)
        elapsed = max(0.0, now - self._updated)
        for key, value in self._values.items():
            decayed = value * math.exp2(-elapsed / self._definitions[key]["half_life_s"])
            self._values[key] = decayed if decayed >= 1e-6 else 0.0
        self._updated = max(now, self._updated)

    def influence(self, changes: dict[str, float]) -> None:
        if not isinstance(changes, dict) or set(changes) - self._definitions.keys():
            raise ValueError("unknown emotional state")
        validated = {k: number(v, -1, 1) for k, v in changes.items()}
        self._advance()
        for key, delta in validated.items():
            value = self._values[key]
            self._values[key] = value + delta * (1 - value) if delta >= 0 else value * (1 + delta)

    def react(
        self,
        event: str,
        *,
        strength: float = 0.5,
        relevance: float = 1,
        novelty: float = 0.5,
        certainty: float = 1,
    ) -> None:
        if not isinstance(event, str) or not any(
            event in x["events"] for x in self._definitions.values()
        ):
            raise ValueError("unknown appraisal event")
        strength, relevance, novelty, certainty = map(
            number, (strength, relevance, novelty, certainty)
        )
        gain = strength * relevance * (0.25 + 0.75 * certainty)
        changes = {}
        for key, item in self._definitions.items():
            if event in item["events"]:
                scale = (0.2 + 0.8 * novelty) if key in {"interest", "surprise"} else 1.0
                changes[key] = item["events"][event] * gain * scale
        self.influence(changes)

    def view(self) -> dict[str, Any]:
        self._advance()
        ranked = sorted(
            ((key, value) for key, value in self._values.items() if value >= THRESHOLD),
            key=lambda item: (-item[1], item[0]),
        )
        primary = ranked[0] if ranked else ("neutral", 1.0)
        secondary = ranked[1] if len(ranked) > 1 else None
        expression = self._definitions[primary[0]]["expression"] if ranked else "neutral"
        return {
            "primary": {"id": primary[0], "intensity": primary[1]},
            "secondary": {"id": secondary[0], "intensity": secondary[1]} if secondary else None,
            "components": copy.deepcopy(self._values),
            "expression": expression,
        }

    def checkpoint(self) -> dict[str, Any]:
        self._advance()
        return copy.deepcopy(
            {
                "schema": "ramvi.character.emotional_state/1",
                "updated_at": self._updated,
                "components": self._values,
                "custom": self._custom,
            }
        )

    def _restore(self, saved: dict[str, Any]) -> None:
        if (
            not isinstance(saved, dict)
            or saved.get("schema") != "ramvi.character.emotional_state/1"
        ):
            raise ValueError("unsupported emotional state schema")
        custom = saved.get("custom", [])
        if not isinstance(custom, list) or len(custom) > 16:
            raise ValueError("invalid custom states")
        for value in custom:
            self.add(value)
        components = saved.get("components", {})
        if not isinstance(components, dict) or set(components) - self._definitions.keys():
            raise ValueError("unknown saved emotional state")
        self._values.update({k: number(v) for k, v in components.items()})
        # Future wall-clock values must not freeze decay after a clock correction.
        self._updated = min(number(saved.get("updated_at"), 0, 1e12), self._updated)
        self._advance()
