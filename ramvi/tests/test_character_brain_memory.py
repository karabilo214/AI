"""9.31 (CHAR-BRAIN-02): правила + память исходов — неодобряемая стратегия заменяется."""

import json
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.brain import (
    SIGNAL_SCHEMA,
    MemoryController,
    OutcomeStore,
    RuleController,
    parse_signal,
)
from ramvi_character.feedback import FeedbackStore
from ramvi_character.memory import MemoryStore
from ramvi_character.personality import Personality
from ramvi_service.module_data import ModuleDatabase
from test_character_memory import DataServices, open_db

pytestmark = pytest.mark.brain
EMOTIONS = ["joy", "confidence", "interest", "surprise", "sadness", "frustration", "anger", "calm"]


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def signal(event: str, **changes: Any) -> Any:
    value = {
        "schema": SIGNAL_SCHEMA,
        "event": event,
        "confidence": 0.9,
        "social_context": "friendly",
        "current_task": "",
        "owner_reaction": "enjoys_banter",
        "last_reply_reaction": "none",
        **changes,
    }
    parsed = parse_signal(json.dumps(value))
    assert parsed is not None
    return parsed


def roaster() -> dict[str, Any]:
    p = Personality()
    p.select("roaster")
    return p.view()


def stores(db: ModuleDatabase) -> tuple[OutcomeStore, FeedbackStore]:
    memory = MemoryStore(DataServices(db), user_id="owner")
    return OutcomeStore(memory), FeedbackStore(memory)


def rate(
    outcomes: OutcomeStore,
    feedback: FeedbackStore,
    event: str,
    strategy: str,
    direction: int,
    origin: str = "explicit_ui",
    start: int = 0,
) -> None:
    for i in range(start, start + 3):
        outcomes.record(f"r{event}{i}", event, strategy, "rules/1")
        feedback.add(f"r{event}{i}", "overall", direction, origin)


def test_disliked_roast_is_replaced_and_liked_one_stays(db: ModuleDatabase) -> None:
    outcomes, feedback = stores(db)
    memory = MemoryController(outcomes)
    assert memory.decide(signal("mistake"), roaster(), EMOTIONS).strategy == "roast"  # данных нет
    rate(outcomes, feedback, "mistake", "roast", -1)  # владелец трижды нажал «не понравилось»
    assert outcomes.outcome("mistake", "roast") == "disliked"
    adapted = memory.decide(signal("mistake"), roaster(), EMOTIONS)
    assert (adapted.strategy, adapted.adapted_from) == ("correct_mistake", "roast")
    assert adapted.controller == "memory/1" and adapted.emotion == "calm"
    # Правила (обучение выключено) по-прежнему подкалывают — память не трогает базу.
    assert RuleController().decide(signal("mistake"), roaster(), EMOTIONS).strategy == "roast"
    rate(outcomes, feedback, "trap", "counter_trap", 1)
    assert memory.decide(signal("trap"), roaster(), EMOTIONS).strategy == "counter_trap"
    assert outcomes.forget() == 6  # откат: исходы забыты
    assert memory.decide(signal("mistake"), roaster(), EMOTIONS).strategy == "roast"


def test_unreliable_or_too_little_feedback_changes_nothing(db: ModuleDatabase) -> None:
    outcomes, feedback = stores(db)
    memory = MemoryController(outcomes)
    rate(outcomes, feedback, "mistake", "roast", -1, origin="implicit")  # догадки, не оценки
    assert memory.decide(signal("mistake"), roaster(), EMOTIONS).strategy == "roast"
    outcomes.record("one", "mistake", "roast", "rules/1")
    feedback.add("one", "overall", -1, "explicit_ui")  # одна кнопка — ещё не устойчиво
    assert outcomes.outcome("mistake", "roast") == "unknown"


def test_safety_strategies_are_never_changed_by_memory(db: ModuleDatabase) -> None:
    outcomes, feedback = stores(db)
    rate(outcomes, feedback, "danger", "serious_help", -1)
    decision = MemoryController(outcomes).decide(
        signal("danger", social_context="serious"), roaster(), EMOTIONS
    )
    assert decision.strategy == "serious_help" and not decision.adapted_from


def react(
    outcomes: OutcomeStore, event: str, strategy: str, reaction: str, count: int, start: int = 0
) -> None:
    for i in range(start, start + count):
        outcomes.record(f"i{event}{i}", event, strategy, "rules/1")
        outcomes.react(f"i{event}{i}", reaction)


def test_implicit_reactions_teach_without_buttons_but_need_twice_the_turns(
    db: ModuleDatabase,
) -> None:
    """9.33: раздражение на подколы про ошибки — без единой кнопки; догадка весит половину."""
    outcomes, _ = stores(db)
    memory = MemoryController(outcomes)
    react(outcomes, "mistake", "roast", "irritated", 3)
    assert outcomes.outcome("mistake", "roast") == "unknown"  # трёх догадок мало
    react(outcomes, "mistake", "roast", "irritated", 3, start=3)
    assert outcomes.outcome("mistake", "roast") == "disliked"
    assert memory.decide(signal("mistake"), roaster(), EMOTIONS).strategy == "correct_mistake"
    react(outcomes, "small_talk", "tell_joke", "laughed", 6)
    assert outcomes.outcome("small_talk", "tell_joke") == "liked"
    react(outcomes, "question", "answer_plainly", "changed_topic", 10)  # неоднозначно — не в счёт
    assert outcomes.outcome("question", "answer_plainly") == "unknown"
    with pytest.raises(ValueError):
        outcomes.react("imistake0", "whatever")
