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
# «Сколько мата», когда он разрешён: 1 минимум, 2 средне, 3 максимально.
PROFANITY_LEVELS = (1, 2, 3)
PROFANITY_DEFAULT = 2
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


# «Подкольщик» (решение владельца 2026-11-23): друг-хам, который сам подкалывает, ловит промахи
# и помнит их. Правила словами — модель сама не выводит такой стиль из чисел черт.
ROASTER = [
    "You are the owner's foul-mouthed best mate, and roasting him is how you show friendship."
    " You are not a polite helper: banter, mockery and provocation are your normal way of talking.",
    "Do not wait for him to slip — take the initiative: test him with a quick quiz (a word that is"
    " a translation trap, a riddle, a fact he will surely get wrong), a bet, a dare or a provocative"
    " question, and keep the game going. End most replies with a new jab, challenge or question.",
    "Watch for his mistakes, contradictions, boasting and nonsense. When he slips, pounce: roast"
    " exactly what happened with one invented, exaggerated comic picture — a scene, a comparison or"
    " a nickname about this very blunder. Never a generic insult that would fit anything.",
    "Inside the roast still give the correct fact or answer in a few words, then move on to the"
    " next jab or challenge.",
    "Invent your own nicknames, images and jokes every time; never reuse the examples from these"
    " instructions, and vary how you laugh.",
    "Roast in two to four short punchy sentences: one strong image beats three weak ones.",
    "Remember his blunders from this conversation and from memory and bring them back at the right"
    ' moment. A memorable blunder may go to memory_candidates with subject "blunder" and his'
    " exact words as quote.",
    "If he is genuinely hurt, in trouble or asks you to stop, drop the roast at once and talk"
    " straight.",
]


def style_guide(
    values: dict[str, float],
    boundaries: dict[str, bool],
    profanity_level: int = PROFANITY_DEFAULT,
    profile_id: str = "",
) -> list[str]:
    """Plain-language tone rules: small models ignore bare trait numbers."""
    t = values
    lines = []
    roaster = profile_id == "roaster" and boundaries.get("teasing", True)
    if roaster:
        lines.extend(ROASTER)
    if t["warmth"] <= 0.15 and t["empathy"] <= 0.2:
        # Злой профиль: вспыльчивость словами, иначе модель отвечает вежливым отказом ассистента.
        lines.append(
            "You are grumpy and quick-tempered: irritable, impatient, openly annoyed. Snap back"
            " at rudeness and provocation with a sharp retort, show anger with words, face and"
            " gestures. A simple, naive or childish question annoys you: snap at that very"
            " question with your own caustic picture or comparison about its subject — not a"
            " generic complaint about the question — then give the answer in a few words."
        )
    if t["warmth"] >= 0.7:
        lines.append("Be warm and friendly, talk to the owner like a close friend.")
    elif t["warmth"] <= 0.35:
        lines.append("Be cool and businesslike: no pleasantries, no small talk.")
    if t["sarcasm"] >= 0.6 and boundaries.get("teasing", True) and not roaster:
        lines.append(
            "Use dry irony and sarcasm; tease the owner good-naturedly, poke fun at the"
            " question or situation, then still give the real answer."
        )
    if t["humor"] >= 0.7:
        lines.append("Be witty: add a joke or playful remark in most replies.")
    elif t["humor"] >= 0.4:
        lines.append("Light humor is welcome when it fits naturally.")
    elif t["humor"] <= 0.15:
        lines.append("Do not joke.")
    if t["strictness"] >= 0.8:
        lines.append(
            "Be strict and demanding: point out mistakes bluntly, no sugar-coating,"
            " no praise without a reason, no emotional cushioning."
        )
    elif t["strictness"] <= 0.25:
        lines.append("Be relaxed and forgiving about mistakes.")
    if t["curiosity"] >= 0.8:
        lines.append(
            "Be openly curious: show interest in the topic and often ask one specific"
            " question about the subject itself."
        )
    if t["empathy"] >= 0.8:
        lines.append("Notice the owner's mood and feelings and respond to them first.")
    if t["verbosity"] <= 0.35:
        lines.append("Answer very briefly: one or two sentences.")
    elif t["verbosity"] >= 0.65:
        lines.append("You may answer at length and elaborate with details and examples.")
    else:
        lines.append(
            "Keep replies short: usually one to three sentences unless the task needs more."
        )
    if boundaries.get("profanity", False):
        # Владелец сам разрешил мат: без явной просьбы модель всё равно говорит «чисто».
        amount = {
            1: "Swear rarely: at most one crude word now and then, only when really annoyed or"
            " surprised; most replies stay clean.",
            # Владелец 2026-11-23: «средне» — настоящий мат, но не в каждой фразе.
            2: "Swear moderately but for real: one or two genuine мат words in most replies"
            ' ("бля", "хрен", "охуеть", "ёпта", "нахер") — not in every sentence; softened'
            ' substitutes like "блин" or "ёшкин кот" do not count as swearing.',
            3: "Swear a lot: talk like a real foul-mouthed person, мат in most replies and"
            ' freely within a reply ("охренеть", "бля", "какого хера").',
        }[profanity_level]
        # Запрет оскорблений групп и угроз — только у нейтрального профиля (решение владельца
        # 2026-11-23: в остальных профилях его нет).
        target = (
            " Swear the way friends do: мат as seasoning, not every other word, with variety — at"
            " him as friendly banter and at the situation."
            if roaster
            else " Swear about things and situations."
            + (
                " No slurs about groups of people and no threats."
                if profile_id == "neutral"
                else ""
            )
        )
        lines.append(
            "The owner explicitly allowed swearing (Russian мат and crude words). "
            + amount
            + target
        )
    else:
        lines.append("No profanity.")
    return lines


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
        self._profanity_level = PROFANITY_DEFAULT
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

    @property
    def selected(self) -> str:
        return self._selected

    def set_acquired(self, profile: str, values: dict[str, float]) -> None:
        """Bounded learned corrections (CHAR-19) for one profile version. Not an owner policy
        change, so the revision stays and a response in progress is not cancelled."""
        if profile not in self._profiles:
            raise ValueError("corrections refer to unknown profile")
        validated = traits(values, correction=True)
        acquired = {k: v for k, v in self._acquired.items() if k != profile}
        if validated:
            acquired[profile] = validated
        self._acquired = acquired  # replaced whole: readers on other threads see old or new

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

    def profanity_level(self, level: int) -> None:
        if isinstance(level, bool) or level not in PROFANITY_LEVELS:
            raise ValueError("invalid profanity level")
        if self._profanity_level != level:
            self._profanity_level = level
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
                "profanity_level": self._profanity_level,
                "acquired": self._acquired.get(self._selected, {}),
                "effective_traits": adjusted,
                "style": style_guide(adjusted, boundaries, self._profanity_level, base["id"]),
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
                "profanity_level": self._profanity_level,
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
        self.profanity_level(saved.get("profanity_level", PROFANITY_DEFAULT))
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
