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
    with pytest.raises(ValueError):
        planner().complete(json.dumps(r))
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
