"""9.32 (CHAR-VIVID): злой по-настоящему злится, подкольщик по-настоящему шутит."""

import json
from typing import Any

import pytest
from ramvi_character.brain import (
    GUIDANCE,
    SIGNAL_SCHEMA,
    STRATEGIES,
    MemoryController,
    RuleController,
    parse_signal,
)
from ramvi_character.personality import Personality

pytestmark = pytest.mark.brain
EMOTIONS = ["joy", "confidence", "interest", "surprise", "sadness", "frustration", "anger", "calm"]


def signal(event: str, social: str = "friendly") -> Any:
    raw = {
        "schema": SIGNAL_SCHEMA,
        "event": event,
        "confidence": 0.9,
        "social_context": social,
        "current_task": "",
        "owner_reaction": "unknown",
        "last_reply_reaction": "none",
    }
    return parse_signal(json.dumps(raw))


def view(profile: str, **boundaries: bool) -> dict[str, Any]:
    p = Personality()
    p.select(profile)
    for name, allowed in boundaries.items():
        p.boundary(name, allowed)
    return p.view()


def decide(profile: str, event: str, recent: tuple[str, ...] = (), **kw: Any) -> Any:
    social = kw.pop("social", "friendly")
    return RuleController(recent).decide(signal(event, social), view(profile, **kw), EMOTIONS)


def test_every_strategy_has_a_plan() -> None:
    assert set(GUIDANCE) == set(STRATEGIES)


def test_angry_attacks_grumbles_and_takes_offense() -> None:
    assert decide("angry", "rudeness").strategy == "attack"
    assert decide("angry", "rudeness").emotion == "anger"
    assert decide("angry", "question").strategy == "grumble"
    assert decide("angry", "gratitude").strategy == "grumble"
    assert decide("angry", "complaint").strategy == "take_offense"
    # Грубость сразу после наезда — обида и злость сильнее.
    again = decide("angry", "rudeness", ("attack",))
    assert again.strategy == "take_offense"
    assert again.intensity > decide("angry", "rudeness").intensity
    # Без разрешения на подколы — ворчит, но не наезжает на владельца.
    assert decide("angry", "rudeness", teasing=False).strategy == "grumble"


def test_roaster_tells_jokes_and_does_not_repeat_the_same_trick() -> None:
    first = decide("roaster", "small_talk")
    second = decide("roaster", "small_talk", (first.strategy,))
    third = decide("roaster", "small_talk", (first.strategy, second.strategy))
    assert len({first.strategy, second.strategy, third.strategy}) == 3
    assert {first.strategy, second.strategy, third.strategy} <= {"tell_joke", "prank", "pun"}
    assert decide("roaster", "joke").strategy in ("pun", "tell_joke")
    assert decide("roaster", "playful_provocation", ("roast",)).strategy == "pun"


def test_safety_and_serious_context_outrank_the_character() -> None:
    for profile in ("angry", "roaster"):
        assert decide(profile, "danger", social="serious").strategy == "serious_help"
        assert decide(profile, "distress").strategy == "gentle_support"
    assert decide("angry", "rudeness", social="serious").strategy not in ("attack", "take_offense")
    assert decide("angry", "rudeness", social="tense").strategy == "attack"  # грубость — напряжённо
    assert decide("roaster", "small_talk", social="serious").strategy == "answer_plainly"


def test_memory_passes_recent_strategies_to_the_rules() -> None:
    class Nothing:
        def outcome(self, event: str, strategy: str) -> str:
            return "unknown"

    d = MemoryController(Nothing(), ("tell_joke",)).decide(
        signal("small_talk"), view("roaster"), EMOTIONS
    )
    assert d.strategy == "prank" and d.controller == "memory/1"
