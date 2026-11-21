"""Profiles and emotion dynamics: explicit priorities, persistence and varied appraisals."""

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.emotions import EmotionalState
from ramvi_character.module import CharacterModule
from ramvi_character.personality import Personality
from ramvi_service.modules.host import ModuleHost
from ramvi_service.sections import SectionRegistry


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_stop_teasing_preserves_other_traits_and_outranks_preferences_and_corrections() -> None:
    p = Personality()
    p.select("sarcastic")
    original = p.view()
    assert original["effective_traits"]["sarcasm"] == 0.85
    p.boundary("teasing", False)  # Explicit owner request: stop teasing.
    current = p.view()
    assert current["effective_traits"]["sarcasm"] == 0
    for key in original["effective_traits"]:
        if key != "sarcasm":
            assert current["effective_traits"][key] == original["effective_traits"][key]
    assert current["base"] == original["base"]
    p.preferences({"sarcasm": 1, "warmth": 0.9})
    saved = p.checkpoint()
    saved["acquired"] = {"sarcastic@1.0.0": {"sarcasm": 0.2, "warmth": -0.2}}
    p = Personality(saved)
    assert p.view()["effective_traits"]["sarcasm"] == 0
    assert p.view()["effective_traits"]["warmth"] == 0.9
    assert p.view()["base"]["traits"]["warmth"] == 0.55
    p.select("friendly")
    assert p.view()["boundaries"]["teasing"] is False
    assert p.view()["effective_traits"]["sarcasm"] == 0
    assert p.view()["acquired"] == {}  # Corrections do not leak across profiles.


def test_custom_versions_roundtrip_without_changing_builtin_profiles() -> None:
    p = Personality()
    builtins = p.catalog()
    assert {x["name"] for x in builtins} == {
        "Neutral",
        "Friendly",
        "Sarcastic",
        "Strict",
        "Curious",
        "Angry",  # злой: вспыльчивый, огрызается — без ненависти и угроз
    }
    custom = {
        "id": "custom_researcher",
        "name": "Исследователь",
        "version": "2.0.0",
        "traits": builtins[0]["traits"],
    }
    p.add_custom(custom)
    custom["traits"]["curiosity"] = 0.98  # Caller must not mutate the stored version.
    p.add_custom({**custom, "version": "10.0.0"})
    p.select("custom_researcher")
    assert p.view()["base"]["version"] == "10.0.0"
    p.select("custom_researcher", "2.0.0")
    assert p.view()["base"]["traits"]["curiosity"] == 0.5
    restored = Personality(p.checkpoint())
    assert restored.view() == p.view()
    assert restored.catalog()[:6] == Personality().catalog()  # шесть встроенных, затем свои
    modified = restored.view()
    modified["base"]["traits"]["curiosity"] = 0
    assert restored.view()["base"]["traits"]["curiosity"] == 0.5


@pytest.mark.parametrize("bad", [True, None, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_trait_updates_are_atomic(bad: Any) -> None:
    p = Personality()
    original = p.checkpoint()
    with pytest.raises(ValueError):
        p.preferences({"warmth": 0.6, "curiosity": bad})
    assert p.checkpoint() == original


def test_invalid_profile_boundary_and_correction_versions_are_rejected() -> None:
    p = Personality()
    original = p.checkpoint()
    for name, allowed in [("unknown", True), ("teasing", 1), ("teasing", "false")]:
        with pytest.raises(ValueError):
            p.boundary(name, allowed)  # type: ignore[arg-type]
        assert p.checkpoint() == original
    for identifier, version in [("neutral", "999.0.0"), ("missing", None), ("neutral", "1.0")]:
        with pytest.raises(ValueError):
            p.select(identifier, version)
        assert p.checkpoint() == original
    for changes in [{"warmth": 0.21}, {"unknown": 0.1}, {"warmth": float("nan")}]:
        saved = p.checkpoint()
        saved["acquired"] = {"neutral@1.0.0": changes}
        with pytest.raises(ValueError):
            Personality(saved)


def test_primary_secondary_half_life_and_recovery_are_time_based() -> None:
    clock = Clock()
    state = EmotionalState(clock=clock)
    state.influence({"joy": 0.8, "confidence": 0.4})
    initial = state.view()
    assert initial["primary"] == {"id": "joy", "intensity": 0.8}
    assert initial["secondary"] == {"id": "confidence", "intensity": 0.4}
    # Merely asking for the state / starting another turn does not reset it.
    assert state.view() == initial
    clock.advance(90)
    current = state.view()
    assert current["components"]["joy"] == pytest.approx(0.4)
    assert current["components"]["confidence"] == pytest.approx(0.4 * 2 ** (-90 / 150))
    clock.advance(10_000)
    assert state.view()["primary"] == {"id": "neutral", "intensity": 1.0}
    assert state.view()["secondary"] is None


def test_same_event_is_not_always_the_same_emotion_and_appraisal_context_matters() -> None:
    for seed in [17, 913, 100032]:
        rng = random.Random(seed)
        for _ in range(12):
            clock = Clock()
            fresh = EmotionalState(clock=clock)
            frustrated = EmotionalState(clock=clock)
            frustrated.influence({"anger": 0.9})
            strength = rng.uniform(0.2, 0.8)
            fresh.react("success", strength=strength)
            frustrated.react("success", strength=strength)
            assert fresh.view()["primary"]["id"] == "joy"
            assert frustrated.view()["primary"]["id"] == "anger"
            assert frustrated.view()["secondary"]["id"] == "joy"
    low, high = EmotionalState(clock=Clock()), EmotionalState(clock=Clock())
    low.react("discovery", novelty=0, relevance=0.2, certainty=0)
    high.react("discovery", novelty=1, relevance=1, certainty=1)
    assert low.view()["components"]["interest"] < high.view()["components"]["interest"]
    irrelevant = EmotionalState(clock=Clock())
    irrelevant.react("conflict", strength=1, relevance=0)
    assert irrelevant.view()["primary"]["id"] == "neutral"


def test_emotional_state_decays_across_restart_and_handles_clock_corrections() -> None:
    clock = Clock()
    state = EmotionalState(clock=clock)
    state.influence({"sadness": 0.8})
    saved = state.checkpoint()
    clock.advance(180)
    restored = EmotionalState(saved, clock=clock)
    assert restored.view()["components"]["sadness"] == pytest.approx(0.4)
    clock.advance(-300)
    assert restored.view()["components"]["sadness"] == pytest.approx(0.4)
    clock.advance(480)
    assert restored.view()["components"]["sadness"] == pytest.approx(0.2)
    future = copy.deepcopy(saved)
    future["updated_at"] = clock.now + 100_000
    restored = EmotionalState(future, clock=clock)
    clock.advance(180)
    assert restored.view()["components"]["sadness"] == pytest.approx(0.4)


def test_new_emotion_and_event_live_in_module_and_survive_checkpoint() -> None:
    clock = Clock()
    state = EmotionalState(clock=clock)
    state.add(
        {
            "id": "determination",
            "expression": "confident",
            "half_life_s": 123,
            "events": {"challenge": 0.8},
        }
    )
    state.react("challenge", strength=1)
    assert state.view()["primary"]["id"] == "determination"
    assert state.view()["expression"] == "confident"
    restored = EmotionalState(state.checkpoint(), clock=clock)
    clock.advance(123)
    assert restored.view()["components"]["determination"] == pytest.approx(0.4)
    restored.react("challenge", strength=0.5)
    assert restored.view()["components"]["determination"] > 0.4


def test_invalid_appraisals_do_not_change_emotional_components() -> None:
    clock = Clock()
    state = EmotionalState(clock=clock)
    state.influence({"calm": 0.5})
    saved = state.checkpoint()
    invalid_changes: list[dict[str, Any]] = [
        {"calm": True},
        {"calm": float("nan")},
        {"calm": 1.1},
        {"unknown": 0.2},
    ]
    for changes in invalid_changes:
        with pytest.raises(ValueError):
            state.influence(changes)
        assert state.checkpoint() == saved
    with pytest.raises(ValueError):
        state.react("conflict", strength=float("inf"))
    with pytest.raises(ValueError):
        state.react("raw phrase from a file")
    assert state.checkpoint() == saved


def test_worker_preserves_profile_boundaries_and_emotions_but_not_old_animation(
    tmp_path: Path,
) -> None:
    modules = Path(__file__).resolve().parents[1] / "modules"
    host = ModuleHost(tmp_path)
    try:
        registry = SectionRegistry(host)
        registry.discover(modules)
        registry.open("character")
        host.invoke("character", "select_profile", {"profile_id": "sarcastic"})
        host.invoke("character", "set_boundary", {"name": "teasing", "allowed": False})
        result = host.invoke("character", "react", {"event": "success", "strength": 0.8})
        assert result["emotional_state"]["primary"]["id"] == "joy"
        host.invoke("character", "present", {"emotion": "cheerful", "animation": "talk"})
        host.suspend("character")
        host.resume("character")
        result = host.invoke("character", "status")
        assert result["personality"]["base"]["id"] == "sarcastic"
        assert result["personality"]["effective_traits"]["sarcasm"] == 0
        assert result["emotional_state"]["primary"]["id"] == "joy"
        assert result["presentation"] is None
    finally:
        host.close()
    host = ModuleHost(tmp_path)
    try:
        registry = SectionRegistry(host)
        registry.discover(modules)
        registry.open("character")
        result = host.invoke("character", "status")
        assert result["personality"]["explicit_boundaries"]["teasing"] is False
        assert result["emotional_state"]["primary"]["id"] == "joy"
    finally:
        host.close()


def test_v1_state_migrates_without_inventing_emotion_from_an_old_preview_pose() -> None:
    module = CharacterModule()
    result = module.initialize(
        {
            "checkpoint": {
                "schema": "ramvi.character.state/1",
                "draft": "draft",
                "model": "model-a",
                "emotion": "cheerful",
            }
        }
    )
    assert result["draft"] == "draft" and result["emotion"] == "cheerful"
    assert result["emotional_state"]["primary"]["id"] == "neutral"
    assert result["personality"]["base"]["id"] == "neutral"
    assert module.checkpoint()["schema"] == "ramvi.character.state/3"


def test_policy_revision_tracks_owner_changes_and_checkpoint_is_rpc_bounded() -> None:
    import json

    p = Personality()
    revision = p.view()["policy_revision"]
    p.boundary("teasing", False)
    assert p.view()["policy_revision"] == revision + 1
    p.boundary("teasing", False)
    assert p.view()["policy_revision"] == revision + 1
    assert Personality(p.checkpoint()).view()["policy_revision"] == revision + 1
    base_traits = p.view()["base"]["traits"]
    corrections = {}
    for i in range(16):
        identifier = (f"custom_{i}_" + "x" * 48)[:48]
        version = "9999999999.9999999999.9999999999"
        p.add_custom(
            {
                "id": identifier,
                "name": "Я" * 60,
                "version": version,
                "traits": {k: 0.12345678901234567 for k in base_traits},
            }
        )
        corrections[identifier + "@" + version] = {k: 0.12345678901234567 for k in base_traits}
    saved = p.checkpoint()
    saved["acquired"] = corrections
    Personality(saved)
    state = EmotionalState()
    for i in range(16):
        state.add(
            {
                "id": (f"custom_{i}_" + "x" * 32)[:32],
                "expression": "neutral",
                "half_life_s": 86400,
                "events": {(f"event_{j}_" + "x" * 32)[:32]: 0.12345678901234567 for j in range(8)},
            }
        )
    module = CharacterModule()
    module.initialize(
        {
            "checkpoint": {
                "schema": "ramvi.character.state/2",
                "draft": "😀" * 8000,
                "task": "g" * 4096,
                "personality": saved,
                "emotional_state": state.checkpoint(),
            }
        }
    )
    encoded = json.dumps(
        {"id": "checkpoint", "result": module.checkpoint()}, ensure_ascii=False
    ).encode()
    assert len(encoded) < 63_000  # Actual worker envelope limit is 65536 bytes.
