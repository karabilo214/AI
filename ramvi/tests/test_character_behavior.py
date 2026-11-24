"""9.34: таблица поведения (правило human_brain «бывает вместе») и настроение персонажа."""

import json
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.brain import (
    SIGNAL_SCHEMA,
    Mood,
    OutcomeStore,
    behavior_rows,
    behavior_text,
    parse_signal,
)
from ramvi_character.memory import MemoryStore
from test_character_memory import DataServices, open_db

pytestmark = pytest.mark.brain


def reactions(event: str, strategy: str, signs: list[int]) -> list[dict[str, Any]]:
    return [
        {"event": event, "strategy": strategy, "response_id": f"{event}{i}", "direction": s}
        for i, s in enumerate(signs)
    ]


def test_three_irritations_are_a_guess_six_are_sure_and_rare_noise_is_not_a_row() -> None:
    decisions = {("mistake", "roast"): 5, ("question", "answer_plainly"): 40}
    rows = behavior_rows(reactions("mistake", "roast", [-1, -1, -1]), decisions)
    assert [(r["event"], r["direction"], r["status"]) for r in rows] == [("mistake", -1, "guess")]
    rows = behavior_rows(
        reactions("mistake", "roast", [-1] * 6), {**decisions, ("mistake", "roast"): 8}
    )
    assert rows[0]["status"] == "sure"
    # Три смешка на 40 обычных ответов — не чаще, чем в среднем: не строка.
    noise = reactions("question", "answer_plainly", [1, 1, 1]) + reactions(
        "small_talk", "tell_joke", [1] * 3
    )
    rows = behavior_rows(noise, {**decisions, ("small_talk", "tell_joke"): 3})
    assert {(r["event"], r["strategy"]) for r in rows} == {("small_talk", "tell_joke")}
    # Противоречивые реакции — не строка.
    assert behavior_rows(reactions("mistake", "roast", [-1, 1, -1, 1]), decisions) == []


def test_prompt_text_marks_guesses_and_only_takes_rows_for_this_event() -> None:
    rows = [
        {
            "event": "mistake",
            "strategy": "roast",
            "direction": -1,
            "status": "guess",
            "reactions": 3,
        },
        {
            "event": "small_talk",
            "strategy": "tell_joke",
            "direction": 1,
            "status": "sure",
            "reactions": 7,
        },
    ]
    text = behavior_text(rows, "mistake")
    assert "probably when he makes a mistake and you roast him, it irritated him" in text
    assert "only a guess" in text and "tell a joke" not in text
    # Один запрос на ход (событие ещё неизвестно) — все строки, догадка остаётся догадкой (9.35).
    unknown = behavior_text(rows, None)
    assert "tell a joke" in unknown and "probably when he makes a mistake" in unknown
    assert behavior_text(rows, "question") == ""


def signal(event: str, reply: str = "none") -> Any:
    raw = {
        "schema": SIGNAL_SCHEMA,
        "event": event,
        "confidence": 0.9,
        "social_context": "tense",
        "current_task": "",
        "owner_reaction": "unknown",
        "last_reply_reaction": reply,
    }
    return parse_signal(json.dumps(raw))


def test_mood_builds_up_from_rudeness_cools_down_and_thanks_soften_it() -> None:
    angry, warm = {"warmth": 0.05}, {"warmth": 0.9}
    mood = Mood()
    mood.update(signal("rudeness"), angry, 1000.0)
    first = mood.irritation
    mood.update(signal("rudeness"), angry, 1010.0)
    assert mood.irritation > first and "irritated" in mood.text() and "he is rude" in mood.text()
    calm = Mood()
    calm.update(signal("rudeness"), warm, 1000.0)
    assert calm.irritation < first  # тёплый характер заводится слабее
    level = mood.irritation
    mood.update(signal("question"), angry, 1020.0)
    assert mood.irritation < level  # остывает от реплики к реплике
    mood.update(signal("question"), angry, 1020.0 + 3600)
    assert mood.irritation < 0.1 and mood.text() == ""  # и со временем
    mood.update(signal("question", reply="laughed"), angry, 5000.0)
    assert "good mood" in mood.text()
    restored = Mood(mood.state())
    assert restored.state() == mood.state()


def test_module_shows_the_table_and_the_mood_survives_a_restart(tmp_path: Path) -> None:
    from test_character_brain import Brained
    from test_character_brain import signal as raw_signal
    from test_character_dialogue import active, finish

    svc = Brained(json.dumps(raw_signal(event="rudeness", social_context="tense")))
    m = active(svc)
    m.dialogue("profile", {"id": "angry"})
    m.dialogue("brain", {"controller": "rules"})
    m.dialogue("start", {"text": "Тупой", "model": "one"})
    finish(m)
    mood = m.dialogue("behavior", {})["mood"]
    assert mood["irritation"] > 0 and mood["cause"] == "rudeness"
    assert svc.data["mood.json"]["irritation"] == mood["irritation"]
    m.dialogue("start", {"text": "Опять тупишь", "model": "one"})
    finish(m)
    system = svc.requests[-1]["messages"][0]["content"]
    assert "YOUR MOOD NOW" in system and "he is rude" in system
    m.dispose()
    again = active(svc)  # тот же хост данных — настроение вернулось
    assert again.dialogue("behavior", {})["mood"]["irritation"] > 0
    assert again.dialogue("brain_forget", {})["forgotten"] == 0
    assert again.dialogue("behavior", {})["mood"]["irritation"] == 0
    again.dispose()


def test_rows_from_the_real_module_database(tmp_path: Path) -> None:
    db = open_db(tmp_path / "data")
    try:
        outcomes = OutcomeStore(MemoryStore(DataServices(db), user_id="owner"))
        for i in range(4):
            outcomes.record(f"r{i}", "mistake", "roast", "rules/1")
            if i < 3:
                outcomes.react(f"r{i}", "irritated")
        for i in range(10):
            outcomes.record(f"q{i}", "question", "answer_plainly", "rules/1")
        rows = behavior_rows(outcomes.reactions(), outcomes.decisions())
        assert [(r["event"], r["status"]) for r in rows] == [("mistake", "guess")]
    finally:
        db.close()
