"""Context-category rotation, readiness barriers and no extra paid preface."""

import copy
import time
from typing import Any

import pytest
from ramvi_character.delivery import PHRASES, ReactionBank, settings
from ramvi_character.response import ResponsePlanner
from ramvi_character.speech import SpeechOutput
from test_character_speech import SpeechServices
from test_speech import rig, wait

__all__ = ["rig"]


def segment(identifier: str = "s1") -> dict[str, Any]:
    return {
        "id": identifier,
        "text": "Свежая завершённая фраза 65539.",
        "emotion": "neutral",
        "intensity": 0.2,
        "voice_style": "neutral",
        "animation": "none",
    }


def start(category: str = "calculate") -> dict[str, Any]:
    return {
        "type": "start",
        "schema": "ramvi.character.response/1",
        "intent": "answer",
        "emotion": "neutral",
        "intensity": 0.2,
        "reaction": category,
        "reaction_language": "ru",
    }


def test_bank_rotates_within_context_and_survives_restart() -> None:
    for category in PHRASES:
        bank = ReactionBank()
        phrases = []
        for _ in range(7):
            chosen = bank.choose(category, "ru")
            assert chosen is not None
            phrases.append(chosen[1])
            bank.used(chosen[0])
            bank = ReactionBank(copy.deepcopy(bank.recent))
        assert len(set(phrases)) == 3
        assert all(a != b for a, b in zip(phrases, phrases[1:], strict=False))
        assert bank.choose(category, "en")[1] not in phrases  # type: ignore[index]
    assert ReactionBank().choose("none", "ru") is None
    assert ReactionBank().choose("exec", "ru") is None
    assert ReactionBank().choose("write", "unknown") is None


def test_context_category_must_come_from_validated_start_not_keywords() -> None:
    planner = ResponsePlanner(emotions=[])
    accepted = planner.accept(start("write"))
    assert accepted is not None and accepted["reaction"] == "write"
    with pytest.raises(ValueError):
        ResponsePlanner(emotions=[]).accept(start("delete_files"))
    # Existing providers may omit the optional fields and get no guessed reaction.
    old = start()
    del old["reaction"]
    del old["reaction_language"]
    accepted = ResponsePlanner(emotions=[]).accept(old)
    assert accepted is not None and "reaction" not in accepted


@pytest.mark.parametrize(
    "policy",
    [{"mode": "bad"}, {"buffer": True}, {"buffer": 9}, {"reactions": "yes"}, {"command": "exec"}],
)
def test_delivery_policy_rejects_unknown_values(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        settings(policy)


def test_complete_mode_holds_eight_phrases_without_blocking_text_and_cancel_discards() -> None:
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    try:
        speech.policy({"mode": "complete", "reactions": False})
        speech.configure({"enabled": True})
        generation = speech.begin("full")
        for index in range(8):
            assert speech.offer("full", segment(f"s{index}"), generation)
        time.sleep(0.05)
        assert not svc.starts and speech.view()["held"] == 8
        speech.finish("full", generation)
        wait(lambda: len(svc.starts) == 8)
        assert len({p["segment_id"] for p in svc.starts}) == 8
        assert all(not p["preface"] for p in svc.starts)
        speech.begin("cancelled")
        speech.offer("cancelled", segment())
        speech.cancel()
        speech.finish("cancelled", speech.generation)
        time.sleep(0.03)
        assert len(svc.starts) == 8 and speech.view()["held"] == 0
        restored = SpeechOutput(svc)
        assert restored.delivery["mode"] == "complete"
        restored.close()
    finally:
        speech.close()
    assert speech.thread is not None and not speech.thread.is_alive()


def test_sentence_buffer_short_answer_flush_and_fast_answer_suppresses_reaction() -> None:
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    try:
        speech.configure({"enabled": True})
        generation = speech.begin("fast")
        speech.react("fast", start(), generation)
        speech.offer("fast", segment(), generation)
        time.sleep(0.03)
        assert not svc.starts
        speech.finish("fast", generation)
        wait(lambda: len(svc.starts) == 1)
        assert not svc.starts[0]["preface"]
        generation = speech.begin("stream")
        speech.offer("stream", segment(), generation)
        speech.offer("stream", segment("s2"), generation)
        wait(lambda: len(svc.starts) == 2)
        speech.finish("stream", generation)
        wait(lambda: len(svc.starts) == 3)
    finally:
        speech.close()


def test_slow_contextual_reaction_once_and_no_replay_after_cancel() -> None:
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    try:
        speech.configure({"enabled": True})
        generation = speech.begin("slow")
        speech.began_at -= 1
        speech.react("slow", start("write"), generation)
        wait(lambda: len(svc.starts) == 1)
        assert svc.starts[0]["preface"]
        assert svc.starts[0]["request"]["text"] in PHRASES["write"][0]
        speech.react("slow", start("write"), generation)
        speech.offer("slow", segment(), generation)
        speech.finish("slow", generation)
        wait(lambda: len(svc.starts) == 2)
        assert not svc.starts[1]["preface"]
        speech.cancel()
        speech.react("slow", start("write"), generation)
        time.sleep(0.03)
        assert len(svc.starts) == 2
        restored = SpeechOutput(svc)
        assert restored.bank.choose("write", "ru")[1] != svc.starts[0]["request"]["text"]  # type: ignore[index]
        restored.close()
    finally:
        speech.close()


def test_external_voice_does_not_get_a_local_or_paid_waiting_fallback() -> None:
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    try:
        speech.configure({"enabled": True})
        speech.local_provider = False
        generation = speech.begin("external")
        speech.began_at -= 1
        speech.react("external", start(), generation)
        assert speech.reaction is None
        time.sleep(0.03)
        assert not svc.starts
    finally:
        speech.close()


def test_service_allows_only_one_local_preface_in_addition_to_eight_segments(rig: Any) -> None:
    service, data, secrets, providers, outputs = rig
    service.configure("owner", data, {"enabled": True})
    service.advance("owner", 1, "reply")

    def play(identifier: str, preface: bool = False) -> str:
        job = service.start(
            "owner",
            data,
            secrets,
            {
                "generation": 1,
                "response_id": "reply",
                "segment_id": identifier,
                "preface": preface,
                "request": {"text": "Fresh 524287."},
            },
        )["job_id"]
        wait(lambda: service.status("owner", job)["job"]["settled"])
        return str(job)

    first = play("~reaction", True)
    for index in range(8):
        play(f"s{index}")
    assert play("~reaction", True) == first
    assert len(outputs) == 9
    with pytest.raises(RuntimeError, match="segment_limit"):
        play("extra")
    with pytest.raises(RuntimeError, match="segment_limit"):
        play("second-preface", True)
    previous = len(providers)
    service.configure("owner", data, {"provider": "openai", "remote_text_enabled": True})
    with pytest.raises(PermissionError, match="preface_local_only"):
        service.start(
            "owner", data, secrets, {"request": {"text": "Do not send this."}, "preface": True}
        )
    assert len(providers) == previous


def test_presentation_preferences_survive_restart_without_mutating_profile_or_model_context() -> (
    None
):
    from test_character_dialogue import active

    svc = SpeechServices()
    module = active(svc)
    before = module.status()["personality"]
    prefs = {"pack_path": "/tmp/user-avatar/pack.json", "show_avatar": False}
    result = module.dialogue("presentation", {"settings": prefs})
    assert result["presentation_settings"] == prefs
    assert module.status()["personality"] == before
    assert module._pack is None and not svc.starts
    invalid = module.dialogue("presentation", {"settings": {**prefs, "show_avatar": "yes"}})
    assert invalid["code"] == "character.request_error"
    restored = active(svc)
    try:
        assert restored.dialogue()["presentation_settings"] == prefs
        restored.set_task("Respond to the current owner request.")
        context = restored.prepare_context("Fresh request")
        assert "/tmp/user-avatar" not in str(context)
    finally:
        module.dispose()
        restored.dispose()
