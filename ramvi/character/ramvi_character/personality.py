"""Versioned profiles. Explicit owner choices always outrank acquired corrections."""

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

DEFINITIONS = Path(__file__).parent / "definitions" / "profiles.json"
TRAITS = frozenset(
    {"warmth", "humor", "sarcasm", "strictness", "curiosity", "verbosity", "initiative", "empathy"}
)
BOUNDARIES = {"teasing": True, "profanity": False, "personal_comments": True, "initiative": True}
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def number(value: Any, low: float = 0, high: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("expected a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError("number outside allowed range")
    return result


def traits(value: Any, *, complete: bool = False, correction: bool = False) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) - TRAITS or (complete and set(value) != TRAITS):
        raise ValueError("invalid trait names")
    return {k: number(v, -0.2, 0.2) if correction else number(v) for k, v in value.items()}


def profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"id", "name", "version", "traits"}:
        raise ValueError("invalid profile")
    identifier, name, version = value["id"], value["name"], value["version"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", identifier):
        raise ValueError("invalid profile id")
    if not isinstance(name, str) or not name.strip() or len(name.encode("utf-8")) > 120:
        raise ValueError("invalid profile name")
    if not isinstance(version, str) or len(version) > 32 or not VERSION.fullmatch(version):
        raise ValueError("invalid profile version")
    return {
        "id": identifier,
        "name": name,
        "version": version,
        "traits": traits(value["traits"], complete=True),
    }


class Personality:
    def __init__(self, saved: dict[str, Any] | None = None) -> None:
        raw = json.loads(DEFINITIONS.read_text(encoding="utf-8"))
        if raw["schema"] != "ramvi.character.profiles/1" or set(raw["traits"]) != TRAITS:
            raise ValueError("unsupported profile definitions")
        self._profiles: dict[str, dict[str, Any]] = {}
        for item in raw["profiles"]:
            validated = profile({**item, "version": raw["version"]})
            self._profiles[self._key(validated["id"], validated["version"])] = validated
        self._custom: list[dict[str, Any]] = []
        self._revision = 0
        self._selected = "neutral@1.0.0"
        self._overrides: dict[str, float] = {}
        self._boundaries: dict[str, bool] = {}
        self._acquired: dict[str, dict[str, float]] = {}
        if saved is not None:
            self._restore(saved)

    @staticmethod
    def _key(identifier: str, version: str) -> str:
        return identifier + "@" + version

    def add_custom(self, value: dict[str, Any]) -> None:
        item = profile(value)
        key = self._key(item["id"], item["version"])
        if not item["id"].startswith("custom_") or key in self._profiles or len(self._custom) >= 16:
            raise ValueError(
                "custom profile must be new and use custom_ prefix (maximum 16 versions)"
            )
        self._profiles[key] = item
        self._custom.append(item)

    def select(self, identifier: str, version: str | None = None) -> None:
        if not isinstance(identifier, str):
            raise ValueError("invalid profile id")
        if version is None:
            choices = [p for p in self._profiles.values() if p["id"] == identifier]
            if not choices:
                raise ValueError("unknown profile")
            version = max(choices, key=lambda p: tuple(map(int, p["version"].split("."))))[
                "version"
            ]
        if not isinstance(version, str) or not VERSION.fullmatch(version):
            raise ValueError("invalid profile version")
        key = self._key(identifier, version)
        if key not in self._profiles:
            raise ValueError("unknown profile version")
        if self._selected != key:
            self._selected = key
            self._revision += 1

    def preferences(self, values: dict[str, float]) -> None:
        validated = traits(values)
        if self._overrides != validated:
            self._overrides = validated
            self._revision += 1

    def boundary(self, name: str, allowed: bool) -> None:
        if not isinstance(name, str) or name not in BOUNDARIES or not isinstance(allowed, bool):
            raise ValueError("invalid explicit boundary")
        if self._boundaries.get(name) is not allowed:
            self._boundaries[name] = allowed
            self._revision += 1

    def view(self) -> dict[str, Any]:
        base = self._profiles[self._selected]
        adjusted = dict(base["traits"])
        for key, delta in self._acquired.get(self._selected, {}).items():
            adjusted[key] = min(1.0, max(0.0, adjusted[key] + delta))
        adjusted.update(self._overrides)
        boundaries = {**BOUNDARIES, **self._boundaries}
        if not boundaries["teasing"]:
            adjusted["sarcasm"] = 0.0
        if not boundaries["initiative"]:
            adjusted["initiative"] = 0.0
        return copy.deepcopy(
            {
                "policy_revision": self._revision,
                "base": base,
                "settings": self._overrides,
                "explicit_boundaries": self._boundaries,
                "boundaries": boundaries,
                "acquired": self._acquired.get(self._selected, {}),
                "effective_traits": adjusted,
            }
        )

    def catalog(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._profiles.values()))

    def checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "schema": "ramvi.character.personality/1",
                "policy_revision": self._revision,
                "selected": self._selected,
                "custom": self._custom,
                "settings": self._overrides,
                "boundaries": self._boundaries,
                "acquired": self._acquired,
            }
        )

    def _restore(self, saved: dict[str, Any]) -> None:
        if not isinstance(saved, dict) or saved.get("schema") != "ramvi.character.personality/1":
            raise ValueError("unsupported personality schema")
        custom = saved.get("custom", [])
        if not isinstance(custom, list) or len(custom) > 16:
            raise ValueError("invalid custom profiles")
        for item in custom:
            self.add_custom(item)
        selected = saved.get("selected")
        if not isinstance(selected, str) or selected not in self._profiles:
            raise ValueError("unknown saved profile version")
        self._selected = selected
        self.preferences(saved.get("settings", {}))
        boundaries = saved.get("boundaries", {})
        if not isinstance(boundaries, dict):
            raise ValueError("invalid explicit boundaries")
        for key, value in boundaries.items():
            self.boundary(key, value)
        acquired = saved.get("acquired", {})
        if not isinstance(acquired, dict) or len(acquired) > 16:
            raise ValueError("invalid acquired corrections")
        for key, changes in acquired.items():
            if key not in self._profiles:
                raise ValueError("corrections refer to unknown profile")
            self._acquired[key] = traits(changes, correction=True)
        revision = saved.get("policy_revision", 0)
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or not 0 <= revision <= 1_000_000_000_000
        ):
            raise ValueError("invalid policy revision")
        self._revision = revision
