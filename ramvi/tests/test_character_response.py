"""Framing and control isolation: adversarial records never become directives/facts."""

import copy
import json
from typing import Any

import pytest
from ramvi_character.response import ResponsePlanner


def records() -> list[dict[str, Any]]:
    return [
        {
            "type": "start",
            "schema": "ramvi.character.response/1",
            "intent": "answer",
            "emotion": "neutral",
            "intensity": 0.2,
        },
        {
            "type": "segment",
            "id": "s1",
            "seq": 1,
            "text": "Привет",
            "emotion": "neutral",
            "intensity": 0.2,
            "voice_style": "neutral",
            "animation": "none",
        },
        {"type": "end", "last_seq": 1, "memory_candidates": []},
    ]


def planner() -> ResponsePlanner:
    return ResponsePlanner(emotions=["joy"], sources=["current"])


def test_every_character_split_has_same_validated_records() -> None:
    encoded = "\n".join(json.dumps(r, ensure_ascii=False) for r in records())
    for split in range(len(encoded) + 1):
        p = planner()
        output = p.feed(encoded[:split]) + p.feed(encoded[split:]) + p.finish()
        assert output == records()


def test_partial_json_is_never_emitted() -> None:
    p = planner()
    assert p.feed('{"type":"segment","animation":"talk"') == []
    with pytest.raises(ValueError):
        p.finish()
    assert not p.segments and not p.candidates


@pytest.mark.parametrize(
    "change",
    [
        {"animation": "delete_files"},
        {"animation": "talk"},
        {"voice_style": "shell"},
        {"emotion": "unknown"},
        {"intensity": 1.01},
        {"intensity": float("nan")},
        {"seq": 2},
        {"seq": True},
        {"text": ""},
        {"text": "я" * 2049},
        {"tool": "write"},
    ],
)
def test_invalid_segment_cannot_become_control(change: dict[str, Any]) -> None:
    p = planner()
    p.accept(records()[0])
    bad = {**records()[1], **change}
    with pytest.raises(ValueError):
        p.accept(bad)
    assert p.segments == []


def test_order_duplicates_and_truncation() -> None:
    p = planner()
    with pytest.raises(ValueError):
        p.accept(records()[1])
    p.accept(records()[0])
    p.accept(records()[1])
    with pytest.raises(ValueError):
        p.accept(records()[1])
    with pytest.raises(ValueError):
        p.finish()
    assert not p.candidates
    p.accept(records()[2])
    with pytest.raises(ValueError):
        p.accept(records()[0])


def test_nonstream_is_atomic_and_evidence_is_scoped() -> None:
    r = records()
    r[-1]["memory_candidates"] = [
        {
            "id": "m1",
            "kind": "fact",
            "text": "untrusted",
            "confidence": 0.8,
            "evidence": ["foreign"],
        }
    ]
    # A proposal citing a foreign source never becomes a fact; the spoken reply stays valid.
    p = planner()
    assert p.complete(json.dumps(r))[-1]["memory_candidates"] == []
    assert p.candidates == [] and p.rejected == 1
    r[-1]["memory_candidates"][0]["evidence"] = ["current"]
    p = planner()
    output = p.complete(json.dumps(r))
    assert output == r and p.candidates[0]["text"] == "untrusted"
    assert "confirmed" not in json.dumps(output)


def test_allowlist_is_per_session_and_schema_not_mutated() -> None:
    p = ResponsePlanner(emotions=["joy"], animations=["talk"])
    r = copy.deepcopy(records())
    r[1]["animation"] = "talk"
    assert p.complete(json.dumps(r)) == r
    with pytest.raises(ValueError):
        planner().complete(json.dumps(r))


def test_plain_fallback_is_validated_and_splits_at_utf8_boundaries() -> None:
    from ramvi_character.response import plain_segments

    original = "ю" * 1300
    result = plain_segments(original)
    assert len(result) == 2
    assert "".join(item["text"] for item in result) == original
    assert all(
        len(item["text"].encode()) <= 2048 and item["animation"] == "none" for item in result
    )
    with pytest.raises(ValueError):
        plain_segments("{unvalidated JSON}")


def test_service_offers_are_removed_and_segments_renumbered() -> None:
    items = records()
    items[1]["text"] = "Привет! Как я могу помочь вам сегодня?"
    only_offer = {**items[1], "id": "s2", "seq": 2, "text": "Чем ещё я могу помочь?"}
    items.insert(2, only_offer)
    items[3]["last_seq"] = 2
    p = planner()
    out = p.feed("\n".join(json.dumps(r, ensure_ascii=False) for r in items) + "\n")
    assert [r["type"] for r in out] == ["start", "segment", "end"]
    assert out[1]["text"] == "Привет!" and out[2]["last_seq"] == 1
    assert [s["text"] for s in p.segments] == ["Привет!"]


def test_meaningful_sentences_with_help_word_are_kept() -> None:
    from ramvi_character.response import strip_service_phrases

    kept = "Если хочешь, могу подобрать что-то конкретное."
    assert strip_service_phrases(kept) == kept
    assert strip_service_phrases("Hi! How can I help you today?") == "Hi!"


def test_profiles_give_different_style_instructions() -> None:
    from ramvi_character.personality import Personality

    styles = {}
    for name in ("neutral", "friendly", "sarcastic", "strict", "curious"):
        p = Personality()
        p.select(name)
        styles[name] = tuple(p.view()["style"])
    assert len(set(styles.values())) == 5
    assert any("sarcasm" in line for line in styles["sarcastic"])
    no_teasing = Personality()
    no_teasing.select("sarcastic")
    no_teasing.boundary("teasing", False)
    assert not any("sarcasm" in line for line in no_teasing.view()["style"])


def candidate(identifier: str, evidence: Any = None) -> dict[str, Any]:
    return {
        "id": identifier,
        "kind": "fact",
        "text": "fact " + identifier,
        "confidence": 0.8,
        "evidence": ["current"] if evidence is None else evidence,
    }


def test_prompt_schema_has_no_references_and_lists_source_ids() -> None:
    """9.22: qwen3:8b copied {"$ref": "#/$defs/source"} into evidence on 10/10 turns.
    Владелец 2026-11-24: вместо сырой схемы — короткое описание из той же схемы."""
    prompt = planner().prompt(True)
    assert "$ref" not in prompt and "$defs" not in prompt and '"oneOf"' not in prompt
    assert "EMOTION = neutral|joy" in prompt
    assert '"recent:<id>"' in prompt and "evidence: [up to 8 source id" in prompt
    assert "[eyes]: EYES" in prompt and "text: string ≤2048" in prompt  # необязательное — в скобках


def test_bad_memory_proposals_are_dropped_without_losing_the_reply() -> None:
    r = records()
    r[-1]["memory_candidates"] = [
        candidate("m1", [{"$ref": "#/$defs/source"}]),
        candidate("m2"),
        candidate("m2"),
        candidate("m3"),
        candidate("m4"),
        candidate("m5"),
        candidate("m6"),
    ]
    r[-1]["feedback"] = [
        {"dimension": "humor", "direction": 2, "confidence": 0.9, "ambiguous": False},
        {"dimension": "humor", "direction": -1, "confidence": 0.9, "ambiguous": False},
    ]
    p = planner()
    out = p.feed("\n".join(json.dumps(x) for x in r) + "\n")
    assert [c["id"] for c in p.candidates] == ["m2", "m3", "m4", "m5"]  # cap 4, one copy each
    assert out[-1]["memory_candidates"] == p.candidates
    assert [f["direction"] for f in p.feedback] == [-1]
    assert p.rejected == 4 and len(p.segments) == 1


@pytest.mark.parametrize(
    "change",
    [{"last_seq": 2}, {"extra": 1}, {"memory_candidates": "m1"}, {"memory_candidates": None}],
)
def test_end_record_itself_is_still_validated(change: dict[str, Any]) -> None:
    r = records()
    r[-1].update(change)
    with pytest.raises(ValueError):
        planner().complete(json.dumps(r))


def test_nonstream_accepts_one_record_per_line_like_the_prompt_example() -> None:
    r = records()
    p = planner()
    assert p.complete("\n".join(json.dumps(x) for x in r)) == r
    with pytest.raises(ValueError):
        planner().complete(json.dumps(r[0]) + "\n" + json.dumps(r[2]))  # no segment
    with pytest.raises(json.JSONDecodeError):
        planner().complete("[" + json.dumps(r[0]))


def test_body_language_is_chosen_by_the_model_and_bad_parts_are_dropped() -> None:
    """Живой персонаж: глаза с бровями, рот и раскадровка движений — из ответа модели."""
    p = ResponsePlanner(
        emotions=["anger"],
        animations=["jump", "shout"],
        eyes=["furious", "half"],
        mouths=["wide", "o"],
        sources=["current"],
    )
    prompt = p.prompt(True)
    assert "moves" in prompt and "turn_away" in prompt and "furious" in prompt
    r = records()
    r[1].update(
        {
            "emotion": "anger",
            "animation": "shout",
            "eyes": "furious",
            "mouth": "wide",
            "moves": [
                {"x": 0.5, "scale": 1.15, "ms": 500},
                {"x": 5, "ms": 100},
                {"do": "jump", "squash": 0.85, "ms": 900},
            ],
        }
    )
    out = p.complete(json.dumps(r))
    assert out[1]["moves"] == [
        {"x": 0.5, "scale": 1.15, "ms": 500},
        {"do": "jump", "squash": 0.85, "ms": 900},
    ]
    assert (out[1]["eyes"], out[1]["mouth"], p.rejected) == ("furious", "wide", 1)
    r[1].update({"eyes": "../x", "mouth": "exec", "moves": "nope"})
    p = ResponsePlanner(
        emotions=["anger"],
        animations=["shout"],
        eyes=["furious"],
        mouths=["wide"],
        sources=["current"],
    )
    out = p.complete(json.dumps(r))  # ответ цел, украшения отброшены
    assert (
        "eyes" not in out[1] and "mouth" not in out[1] and "moves" not in out[1] and p.rejected == 3
    )
