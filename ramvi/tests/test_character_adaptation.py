"""9.19 (CHAR-19): ограниченные по диапазону и скорости персональные поправки черт."""

from pathlib import Path
from typing import Any

import pytest
from ramvi_character.adaptation import LIMIT, STEP, Adaptation
from ramvi_character.feedback import FeedbackStore
from ramvi_character.memory import MemoryStore
from ramvi_character.personality import Personality
from ramvi_service.module_data import ModuleDatabase
from ramvi_storage.database import ConstraintError
from test_character_consolidation import Both, with_candidates
from test_character_dialogue import active, finish
from test_character_memory import DataServices, open_db

PROFILE = "sarcastic@1.0.0"


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


class Clock:
    def __init__(self) -> None:
        self.hour = 0

    def __call__(self) -> str:
        day, hour = divmod(self.hour, 24)
        return f"2026-11-{20 + day:02d}T{hour:02d}:00:00Z"


def setup(db: ModuleDatabase, user: str = "owner") -> tuple[Adaptation, FeedbackStore, Clock]:
    clock = Clock()
    memory = MemoryStore(DataServices(db), user_id=user, clock=clock)
    feedback = FeedbackStore(memory)
    return Adaptation(memory, feedback), feedback, clock


def settle(feedback: FeedbackStore, trait: str, direction: int, tag: str) -> None:
    """Три достоверные оценки из двух ответов — установленное предпочтение."""
    feedback.add(f"{tag}a", trait, direction, "explicit_ui")
    feedback.add(f"{tag}b", trait, direction, "explicit_ui")
    feedback.add(
        f"{tag}b", trait, direction, "user_said", quote="да", owner_text="да"
    )  # «да» — не в зачёт
    feedback.add(f"{tag}c", trait, direction, "explicit_ui")


def test_one_step_only_from_settled_evidence(db: ModuleDatabase) -> None:
    a, f, _ = setup(db)
    f.add("r1", "humor", -1, "explicit_ui")
    f.add("r1", "humor", -1, "user_said", quote="хватит шуток", owner_text="хватит шуток")
    assert a.update(PROFILE, {}, {}) == [] and a.corrections(PROFILE) == {}
    settle(f, "humor", -1, "x")
    (change,) = a.update(PROFILE, {}, {})
    assert change["trait"] == "humor" and change["after"] == -STEP
    assert change["reason"]["reliable"] >= 3 and change["reason"]["responses"] >= 2
    assert a.corrections(PROFILE) == {"humor": -STEP}
    assert a.update(PROFILE, {}, {}) == []  # те же сигналы второй раз не считаются


def test_speed_needs_time_and_fresh_evidence_and_range_is_bounded(db: ModuleDatabase) -> None:
    a, f, clock = setup(db)
    settle(f, "verbosity", -1, "r0")
    assert a.update(PROFILE, {}, {})
    clock.hour += 1
    settle(f, "verbosity", -1, "r1")
    assert a.update(PROFILE, {}, {}) == []  # свежие оценки есть, но прошёл только час
    clock.hour += 6
    assert len(a.update(PROFILE, {}, {})) == 1  # время прошло и оценки новые
    clock.hour += 12
    assert a.update(PROFILE, {}, {}) == []  # время прошло, новых оценок нет
    for round_ in range(20):
        clock.hour += 7
        settle(f, "verbosity", -1, f"s{round_}")
        a.update(PROFILE, {}, {})
    assert a.corrections(PROFILE) == {"verbosity": -LIMIT}
    with pytest.raises(ConstraintError):  # предел есть и в схеме
        db.execute("UPDATE adaptation_state SET value = -0.5")
    history = a.history(PROFILE)
    assert history[0]["after"] == -LIMIT and all(
        abs(h["after"] - h["before"]) <= STEP + 1e-9 for h in history
    )


def test_owner_settings_and_boundaries_outrank_adaptation(db: ModuleDatabase) -> None:
    a, f, _ = setup(db)
    settle(f, "humor", 1, "h")
    settle(f, "sarcasm", 1, "s")
    settle(f, "warmth", 1, "w")
    changes = a.update(PROFILE, {"humor": 0.2}, {"teasing": False})
    assert [c["trait"] for c in changes] == ["warmth"]  # humor задан владельцем, сарказм запрещён

    personality = Personality()
    personality.select("sarcastic")
    base = dict(personality.view()["base"]["traits"])
    personality.set_acquired(PROFILE, {"humor": 0.1, "sarcasm": 0.1, "warmth": 0.03})
    personality.preferences({"humor": 0.2})
    personality.boundary("teasing", False)
    view = personality.view()
    assert view["effective_traits"]["humor"] == 0.2  # настройка владельца выше поправки
    assert view["effective_traits"]["sarcasm"] == 0.0  # граница выше поправки
    assert view["effective_traits"]["warmth"] == pytest.approx(base["warmth"] + 0.03)
    assert view["base"]["traits"] == base  # базовый профиль не меняется


def test_users_and_profiles_have_separate_corrections(db: ModuleDatabase) -> None:
    owner, owner_feedback, _ = setup(db)
    guest, guest_feedback, _ = setup(db, "guest")
    settle(guest_feedback, "humor", -1, "g")
    assert guest.update(PROFILE, {}, {}) and owner.update(PROFILE, {}, {}) == []
    assert owner.corrections(PROFILE) == {} and guest.corrections(PROFILE) == {"humor": -STEP}
    settle(owner_feedback, "humor", 1, "o")
    owner.update("friendly@1.0.0", {}, {})
    assert owner.corrections("friendly@1.0.0") == {"humor": STEP}
    assert owner.corrections(PROFILE) == {}


def talk(m: Any, text: str) -> str:
    response = m.dialogue("start", {"text": text, "model": "one"})
    assert finish(m)["job"]["state"] == "completed"
    return str(response["response_id"])


def test_module_applies_persists_and_keeps_policy_revision(db: ModuleDatabase) -> None:
    svc = Both(db, with_candidates())
    m = active(svc)
    m.dialogue("profile", {"id": "sarcastic"})
    before = m.status()["personality"]
    replies = [talk(m, f"Реплика {n}") for n in range(3)]
    for reply in replies[:2]:
        m.dialogue("feedback", {"direction": -1, "dimension": "humor", "response_id": reply})
    assert m.status()["personality"]["acquired"] == {}  # двух оценок мало
    m.dialogue("feedback", {"direction": -1, "dimension": "humor", "response_id": replies[2]})
    after = m.status()["personality"]
    assert after["acquired"] == {"humor": -STEP}
    assert after["effective_traits"]["humor"] == pytest.approx(
        before["effective_traits"]["humor"] - STEP
    )
    assert after["policy_revision"] == before["policy_revision"]  # идущий ответ не отменяется
    m.dialogue("profile", {"id": "friendly"})
    assert m.status()["personality"]["acquired"] == {}  # у другого профиля свои поправки
    m.dispose()

    restarted = active(svc)
    restarted.dialogue("profile", {"id": "sarcastic"})
    assert restarted.status()["personality"]["acquired"] == {"humor": -STEP}
    restarted.dispose()


def test_reactions_in_conversation_adapt_in_background(db: ModuleDatabase) -> None:
    from test_character_feedback import reply_with_feedback

    less = {
        "dimension": "verbosity",
        "direction": -1,
        "confidence": 0.9,
        "quote": "слишком длинно",
        "ambiguous": False,
    }
    svc = Both(db, with_candidates())
    m = active(svc)
    talk(m, "Расскажи про кофе")
    svc.output = reply_with_feedback(less)
    for _ in range(3):  # каждая реакция относится к предыдущему ответу — три разных ответа
        talk(m, "Слишком длинно, давай короче")
    assert m._consolidator is not None and m._consolidator.wait()
    assert m.status()["personality"]["acquired"] == {"verbosity": -STEP}
    assert m.status()["memory"]["background"]["errors"] == 0
    m.dispose()
