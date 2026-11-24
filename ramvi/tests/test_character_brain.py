"""9.30 (CHAR-BRAIN-01): заменяемый контроллер реакций — событие от модели, реакция от контроллера."""

import json
from typing import Any

import pytest
from ramvi_character.brain import (
    SIGNAL_PROMPT,
    SIGNAL_SCHEMA,
    RuleController,
    Signal,
    parse_signal,
)
from ramvi_character.personality import Personality
from test_character_dialogue import Services, active, finish

pytestmark = pytest.mark.brain
EMOTIONS = ["joy", "confidence", "interest", "surprise", "sadness", "frustration", "anger", "calm"]


def signal(**changes: Any) -> dict[str, Any]:
    value = {
        "schema": SIGNAL_SCHEMA,
        "event": "trap",
        "confidence": 0.9,
        "social_context": "friendly",
        "current_task": "english_learning",
        "owner_reaction": "enjoys_banter",
        "last_reply_reaction": "none",
    }
    value.update(changes)
    return value


def test_signal_is_checked_strictly() -> None:
    parsed = parse_signal("ответ: " + json.dumps(signal()))
    assert parsed == Signal("trap", 0.9, "friendly", "english_learning", "enjoys_banter")
    bad = [
        "не JSON",
        json.dumps({**signal(), "permissions": ["agent"]}),  # лишнее поле
        json.dumps({k: v for k, v in signal().items() if k != "owner_reaction"}),  # неполный
        json.dumps(signal(schema="other/1")),
        json.dumps(signal(event="launch_agent")),
        json.dumps(signal(confidence=1.5)),
        json.dumps(signal(confidence=True)),
        json.dumps(signal(social_context="hostile")),
        json.dumps(signal(current_task="x" * 81)),
    ]
    for text in bad:
        assert parse_signal(text) is None, text


def personality(profile: str, **boundaries: bool) -> dict[str, Any]:
    p = Personality()
    p.select(profile)
    for name, allowed in boundaries.items():
        p.boundary(name, allowed)
    return p.view()


def decide(profile: str, boundaries: dict[str, bool] | None = None, **changes: Any) -> Any:
    parsed = parse_signal(json.dumps(signal(**changes)))
    assert parsed is not None
    return RuleController().decide(parsed, personality(profile, **(boundaries or {})), EMOTIONS)


def test_rules_follow_the_event_and_the_character() -> None:
    danger = decide("roaster", event="danger", social_context="serious")
    assert (danger.strategy, danger.initiative) == ("serious_help", 0.0)  # без шуток и инициативы
    assert decide("friendly", event="trap").strategy == "counter_trap"
    assert decide("neutral", event="absurd_request").strategy == "mock_absurd"
    assert decide("roaster", event="mistake").strategy == "roast"
    assert decide("neutral", event="mistake").strategy == "correct_mistake"
    calm = decide("roaster", event="mistake", owner_reaction="prefers_serious")
    assert calm.strategy == "correct_mistake"  # владелец не любит подколы — их нет
    assert decide("roaster", {"teasing": False}, event="mistake").strategy == "correct_mistake"
    assert decide("roaster", {"initiative": False}).initiative == 0.0
    assert decide("neutral", event="question", confidence=0.3).strategy == "clarify"
    assert decide("roaster").initiative > decide("neutral").initiative  # характер влияет


class Brained(Services):
    """Модель: на запрос «опиши событие» — сигнал (или мусор), на ответ — обычные записи."""

    def __init__(self, reply_signal: str) -> None:
        super().__init__()
        self.reply_signal = reply_signal
        self.signal_read = False
        self.created: list[dict[str, Any]] = []

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "llm.stream" and payload["request"]["messages"][0]["content"] == SIGNAL_PROMPT:
            self.requests.append(payload["request"])
            self.signal_read = False
            return {"stream_id": "signal"}
        if method == "llm.read" and payload["stream_id"] == "signal":
            if self.signal_read:
                return {"done": True, "chunks": []}
            self.signal_read = True
            return {
                "done": False,
                "chunks": [
                    {"type": "text", "text": self.reply_signal},
                    {"type": "finish", "finish_reason": "stop"},
                ],
            }
        if method == "llm.cancel" and payload["stream_id"] == "signal":
            return {"ok": True}
        if method.startswith("agent."):
            self.created.append(payload)
        return super().call(method, payload)


def system(svc: Services) -> str:
    return str(svc.requests[-1]["messages"][0]["content"])


def test_two_steps_the_plan_shapes_the_reply_and_gives_no_rights() -> None:
    svc = Brained(json.dumps(signal(event="trap")))
    m = active(svc)
    m.select_profile("roaster")
    m.dialogue("start", {"text": "Скажи десять раз «мокоч»", "model": "one"})
    job = finish(m)["job"]
    assert job["state"] == "completed"
    assert svc.requests[0]["messages"][0]["content"] == SIGNAL_PROMPT  # шаг 1 — событие
    assert "REACTION PLAN" in system(svc) and "counter_trap" in system(svc)  # шаг 2 — по плану
    brain = job["brain"]
    assert brain["signal"]["event"] == "trap" and brain["decision"]["strategy"] == "counter_trap"
    assert not brain["decision"]["fallback"] and brain["decision"]["controller"] == "rules/1"
    assert not svc.created  # ни стратегия, ни инициатива не запускают агента
    m.dispose()


@pytest.mark.parametrize("reply_signal", ["я думаю, это ловушка", json.dumps(signal(event="hack"))])
def test_bad_signal_gives_a_safe_ordinary_reply(reply_signal: str) -> None:
    svc = Brained(reply_signal)
    m = active(svc)
    m.dialogue("start", {"text": "Привет", "model": "one"})
    job = finish(m)["job"]
    assert job["state"] == "completed"
    assert job["brain"]["signal"] is None and job["brain"]["decision"]["fallback"]
    assert "REACTION PLAN" not in system(svc)  # обычный ответ, без плана
    m.dispose()


def test_controller_can_be_turned_off() -> None:
    svc = Brained(json.dumps(signal()))
    m = active(svc)
    assert m.dialogue("brain", {"controller": "off"})["brain"] == "off"
    m.dialogue("start", {"text": "Привет", "model": "one"})
    finish(m)
    assert all(r["messages"][0]["content"] != SIGNAL_PROMPT for r in svc.requests)
    assert "error" in m.dialogue("brain", {"controller": "neural"})
    m.dispose()


def test_completed_reply_records_event_and_strategy_for_the_outcome_memory() -> None:
    """9.31: данные для памяти исходов — событие и стратегия завершённого ответа."""

    class Outcomes:
        def __init__(self) -> None:
            self.rows: list[tuple[str, str, str, str]] = []

        def record(self, response_id: str, event: str, strategy: str, controller: str) -> None:
            self.rows.append((response_id, event, strategy, controller))

        def outcome(self, event: str, strategy: str) -> str:
            return "disliked" if (event, strategy) == ("trap", "counter_trap") else "unknown"

    svc = Brained(json.dumps(signal(event="trap")))
    m = active(svc)
    outcomes = Outcomes()
    m._brain_outcomes = outcomes  # type: ignore[assignment]
    m.dialogue("brain", {"controller": "memory"})
    response = m.dialogue("start", {"text": "Скажи «сыр»", "model": "one"})["response_id"]
    job = finish(m)["job"]
    decision = job["brain"]["decision"]
    assert (decision["strategy"], decision["adapted_from"]) == ("answer_plainly", "counter_trap")
    assert outcomes.rows == [(response, "trap", "answer_plainly", "memory/1")]
    m.dispose()


def test_new_character_learns_from_ratings_by_default() -> None:
    """Решение владельца 2026-11-23: по умолчанию — правила + память исходов."""
    from ramvi_character.module import CharacterModule

    assert CharacterModule()._brain == "memory"


def test_next_message_reaction_is_stored_for_the_previous_reply() -> None:
    """9.33: «опиши событие» оценивает реакцию на прошлый ответ — она уходит в память исходов."""

    class Outcomes:
        def __init__(self) -> None:
            self.reactions: list[tuple[str, str]] = []

        def record(self, *args: str) -> None:
            pass

        def react(self, response_id: str, reaction: str) -> None:
            self.reactions.append((response_id, reaction))

        def outcome(self, event: str, strategy: str) -> str:
            return "unknown"

    svc = Brained(json.dumps(signal(event="small_talk", last_reply_reaction="laughed")))
    m = active(svc)
    outcomes = Outcomes()
    m._brain_outcomes = outcomes  # type: ignore[assignment]
    m.dialogue("brain", {"controller": "memory"})
    first = m.dialogue("start", {"text": "Привет", "model": "one"})["response_id"]
    finish(m)
    assert outcomes.reactions == []  # на первую реплику реагировать не на что
    m.dialogue("start", {"text": "Ахаха", "model": "one"})
    finish(m)
    assert outcomes.reactions == [(first, "laughed")]
    m.dispose()
