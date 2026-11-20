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


# --- 9.20 (CHAR-20): выключение, история, точки сохранения, сброс и откат ----------------


def grow(a: Adaptation, f: FeedbackStore, clock: Clock, trait: str, direction: int, n: int) -> None:
    for step in range(n):
        clock.hour += 7
        settle(f, trait, direction, f"{trait}{direction}{clock.hour}{step}")
        assert a.update(PROFILE, {}, {}), (trait, step)


def test_restore_reproduces_the_earlier_state_exactly(db: ModuleDatabase) -> None:
    a, f, clock = setup(db)
    grow(a, f, clock, "humor", -1, 2)
    grow(a, f, clock, "warmth", 1, 3)
    saved_state = a.corrections(PROFILE)
    saved = a.save(PROFILE)
    assert saved["kind"] == "saved" and saved["traits"] == saved_state
    grow(a, f, clock, "verbosity", -1, 1)
    grow(a, f, clock, "humor", 1, 1)
    before_reset = a.corrections(PROFILE)
    assert before_reset != saved_state

    clock.hour += 1
    assert {c["trait"] for c in a.reset(PROFILE)} == set(before_reset)
    assert a.corrections(PROFILE) == {}
    assert a.reset(PROFILE) == []  # сбрасывать нечего — и в журнале ничего лишнего
    clock.hour += 1
    a.restore(PROFILE)  # «вернуть как было»: последнее изменение отменено
    assert a.corrections(PROFILE) == before_reset
    clock.hour += 1
    a.restore(PROFILE, saved["id"])
    assert a.corrections(PROFILE) == saved_state  # точь-в-точь, включая убранные черты
    a.save(PROFILE)  # точка сохранения не мешает «вернуть как было»
    clock.hour += 1
    a.restore(PROFILE)  # и откат отката
    assert a.corrections(PROFILE) == before_reset

    history = a.history(PROFILE)
    assert history[0]["action"] == "restore" and {h["action"] for h in history} == {
        "step",
        "reset",
        "restore",
    }
    # Каждая запись журнала ведёт к снимку «как было до неё».
    for entry in history:
        (snap,) = [s for s in a.snapshots(PROFILE, 100) if s["id"] == entry["snapshot_id"]]
        assert snap["traits"].get(entry["trait"], 0.0) == entry["before"]
    with pytest.raises(ValueError):
        a.restore(PROFILE, "s_missing")
    with pytest.raises(ValueError):
        a.restore("friendly@1.0.0", saved["id"])  # снимок чужого профиля не подходит
    with pytest.raises(ConstraintError):
        db.execute("UPDATE adaptation_change SET action = 'magic'")


def test_owner_actions_are_not_undone_by_old_ratings(db: ModuleDatabase) -> None:
    a, f, clock = setup(db)
    grow(a, f, clock, "humor", -1, 1)
    settle(f, "humor", -1, "fresh")  # свежие оценки, которые уже были до сброса
    clock.hour += 7
    a.reset(PROFILE)
    clock.hour += 7
    assert a.update(PROFILE, {}, {}) == [] and a.corrections(PROFILE) == {}
    settle(f, "humor", -1, "after")
    assert a.update(PROFILE, {}, {})  # новые оценки после сброса — снова можно


def test_switch_off_keeps_corrections_and_ignores_ratings_given_meanwhile(
    db: ModuleDatabase,
) -> None:
    a, f, clock = setup(db)
    grow(a, f, clock, "humor", -1, 1)
    a.set_enabled(False)
    assert not a.enabled and a.applied(PROFILE) == {}
    assert a.corrections(PROFILE) == {"humor": -STEP}  # хранится, просто не применяется
    clock.hour += 7
    settle(f, "humor", -1, "off")
    assert a.update(PROFILE, {}, {}) == []
    clock.hour += 1
    a.set_enabled(True)
    assert a.applied(PROFILE) == {"humor": -STEP}
    clock.hour += 7
    assert a.update(PROFILE, {}, {}) == []  # оценки, данные при выключенной подстройке, не в счёт
    settle(f, "humor", -1, "on")
    assert a.update(PROFILE, {}, {})
    clock.hour += 7
    settle(f, "humor", -1, "again")
    clock.hour += 1
    a.set_enabled(True)  # повторное включение не обнуляет уже собранные оценки
    assert a.update(PROFILE, {}, {})


def test_automatic_snapshots_are_bounded_saved_points_kept(db: ModuleDatabase) -> None:
    from ramvi_character.adaptation import KEEP_SNAPSHOTS

    a, f, clock = setup(db)
    grow(a, f, clock, "humor", 1, 1)
    saved = a.save(PROFILE)
    for n in range(KEEP_SNAPSHOTS + 5):
        clock.hour += 1
        a.reset(PROFILE) if n % 2 == 0 else a.restore(PROFILE)
    kinds = [s["kind"] for s in a.snapshots(PROFILE, 1000)]
    assert kinds.count("saved") == 1 and len(kinds) == KEEP_SNAPSHOTS + 1
    a.restore(PROFILE, saved["id"])
    assert a.corrections(PROFILE) == {"humor": STEP}


def test_module_switch_keeps_profile_memory_and_survives_restart(db: ModuleDatabase) -> None:
    from test_character_consolidation import candidate

    name = candidate("Владельца зовут Анна", subject="имя", quote="меня зовут Анна")
    svc = Both(db, with_candidates(name))
    m = active(svc)
    m.dialogue("profile", {"id": "sarcastic"})
    base = m.status()["personality"]["effective_traits"]
    replies = [talk(m, f"Меня зовут Анна, реплика {n}") for n in range(3)]
    for reply in replies:
        m.dialogue("feedback", {"direction": -1, "dimension": "humor", "response_id": reply})
    assert m._consolidator is not None and m._consolidator.wait()
    memory = m.dialogue("memory")["memory"]["records"]
    signals = m.dialogue("feedback_view")["feedback"]["signals"]
    assert memory and m.status()["personality"]["acquired"] == {"humor": -STEP}

    view = m.dialogue("adaptation_enable", {"enabled": False})["adaptation"]
    assert view["enabled"] is False and view["corrections"] == {"humor": -STEP}
    assert view["applied"] == {}
    status = m.status()
    assert status["personality"]["acquired"] == {}
    assert status["personality"]["effective_traits"] == base
    assert status["personality"]["base"]["id"] == "sarcastic"  # профиль на месте
    assert m.dialogue("memory")["memory"]["records"] == memory  # память на месте
    assert m.dialogue("feedback_view")["feedback"]["signals"] == signals
    assert "error" in m.dialogue("adaptation_enable", {"enabled": "no"})
    m.dispose()

    restarted = active(svc)
    restarted.dialogue("profile", {"id": "sarcastic"})
    assert restarted.status()["personality"]["acquired"] == {}  # выключение пережило перезапуск
    view = restarted.dialogue("adaptation_enable", {"enabled": True})["adaptation"]
    assert restarted.status()["personality"]["acquired"] == {"humor": -STEP}
    saved = restarted.dialogue("adaptation_save")["adaptation"]["snapshots"][0]
    restarted.dialogue("adaptation_reset")
    assert restarted.status()["personality"]["acquired"] == {}
    view = restarted.dialogue("adaptation_restore", {"snapshot_id": saved["id"]})["adaptation"]
    assert view["applied"] == {"humor": -STEP} and view["history"][0]["action"] == "restore"
    assert restarted.status()["personality"]["acquired"] == {"humor": -STEP}
    assert "error" in restarted.dialogue("adaptation_restore", {"snapshot_id": "s_x"})
    exported = restarted.dialogue("memory_export")["memory"]["adaptation"]
    assert exported["enabled"] and len(exported["snapshots"]) >= 3
    restarted.dispose()
