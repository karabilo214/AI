"""9.18 (CHAR-18): обратная связь с уверенностью и происхождением; одна реплика ничего не решает."""

import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.feedback import FeedbackStore, ambiguous_words
from ramvi_character.memory import MemoryStore
from ramvi_character.module import CharacterModule
from ramvi_service.module_data import ModuleDatabase
from ramvi_storage.database import ConstraintError
from test_character_consolidation import Both, with_candidates
from test_character_dialogue import active, finish
from test_character_memory import DataServices, open_db
from test_character_response import records


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def store(db: ModuleDatabase, user: str = "owner") -> FeedbackStore:
    return FeedbackStore(MemoryStore(DataServices(db), user_id=user))


def test_code_decides_which_signal_is_a_reliable_assessment(db: ModuleDatabase) -> None:
    f = store(db)
    assert f.add("r1", "overall", 1, "explicit_ui")["reliable"]  # кнопка владельца
    said = f.add(
        "r1",
        "humor",
        1,
        "user_said",
        quote="ахаха, смешно",
        owner_text="Ахаха,  смешно же!",
        confidence=0.9,
    )
    assert said["reliable"] and said["origin"] == "user_said"
    # Цитаты нет в словах владельца — это догадка модели.
    fake = f.add("r1", "humor", 1, "user_said", quote="обожаю шутки", owner_text="ну ладно")
    assert fake["origin"] == "interpretation" and not fake["reliable"]
    # Короткое «ок» / «ну да» после шутки — не одобрение, даже дословно.
    for words in ("ок", "Ну да.", "Ага!", "ok"):
        signal = f.add("r2", "humor", 1, "user_said", quote=words, owner_text=words, confidence=1)
        assert signal["ambiguous"] and not signal["reliable"], words
    assert not f.add(
        "r3", "humor", 1, "user_said", quote="класс", owner_text="класс", ambiguous=True
    )["reliable"]
    assert not f.add(
        "r3", "verbosity", -1, "user_said", quote="короче", owner_text="короче", confidence=0.4
    )["reliable"]
    assert not f.add("r3", "warmth", 1, "implicit", confidence=1)["reliable"]
    assert ambiguous_words("  хм... ") and not ambiguous_words("очень смешно")
    # Схема сама не примет «достоверный неоднозначный» сигнал.
    with pytest.raises(ConstraintError):
        db.execute(
            "UPDATE feedback_signal SET reliable = 1 WHERE ambiguous = 1 OR origin = 'implicit'"
        )
    for name, value in [("dimension", "rudeness"), ("direction", 0), ("origin", "model")]:
        args: dict[str, Any] = {"dimension": "humor", "direction": 1, "origin": "explicit_ui"}
        args[name] = value
        with pytest.raises(ValueError):
            f.add("r4", args["dimension"], args["direction"], args["origin"])


def preference(f: FeedbackStore, dimension: str) -> dict[str, Any]:
    return next(p for p in f.preferences() if p["dimension"] == dimension)


def test_one_remark_does_not_settle_a_preference(db: ModuleDatabase) -> None:
    f = store(db)
    f.add("r1", "humor", -1, "explicit_ui")
    for _ in range(5):  # повторные нажатия на тот же ответ заменяют сигнал
        f.add("r1", "humor", -1, "explicit_ui")
    single = preference(f, "humor")
    assert single["direction"] == -1 and single["reliable"] == 1 and not single["established"]
    f.add("r1", "humor", -1, "user_said", quote="хватит шуток", owner_text="Хватит шуток")
    f.add("r1", "humor", -1, "user_said", quote="без шуток", owner_text="Давай без шуток")
    one_reply = preference(f, "humor")
    assert one_reply["reliable"] == 3 and not one_reply["established"]  # всё ещё один ответ
    f.add("r2", "humor", -1, "explicit_ui")
    settled = preference(f, "humor")
    assert settled["established"] and settled["responses"] == 2 and settled["reliable"] == 4

    for n in range(6):  # сколько угодно неоднозначных сигналов — ноль достоверных
        f.add(f"q{n}", "warmth", 1, "user_said", quote="ок", owner_text="ок")
    warmth = preference(f, "warmth")
    assert warmth["direction"] == 0 and warmth["reliable"] == 0 and warmth["unreliable"] == 6
    for n, direction in enumerate([1, 1, -1, 1]):  # 3 против 1 = 0,75
        f.add(f"v{n}", "verbosity", direction, "explicit_ui")
    assert preference(f, "verbosity")["established"]
    f.add("v9", "verbosity", -1, "explicit_ui")  # 3 против 2 — уже не согласованно
    assert not preference(f, "verbosity")["established"]


def test_signals_are_per_user(db: ModuleDatabase) -> None:
    owner, guest = store(db), store(db, "guest")
    for n in range(3):
        owner.add(f"r{n}", "sarcasm", -1, "explicit_ui")
    assert preference(owner, "sarcasm")["established"]
    assert guest.preferences() == [] and guest.signals() == []
    assert guest.clear() == 0 and len(owner.signals()) == 3


def reply_with_feedback(*signals: dict[str, Any]) -> str:
    import json

    r = records()
    r[-1]["feedback"] = list(signals)
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in r)


def talk(m: CharacterModule, text: str) -> str:
    response = m.dialogue("start", {"text": text, "model": "one"})
    assert finish(m)["job"]["state"] == "completed"
    return str(response["response_id"])


def signals(m: CharacterModule) -> list[dict[str, Any]]:
    assert m._consolidator is not None and m._consolidator.wait()
    return list(m.dialogue("feedback_view")["feedback"]["signals"])


def test_dialogue_reaction_targets_previous_reply_and_is_checked(db: ModuleDatabase) -> None:
    laugh = {
        "dimension": "humor",
        "direction": 1,
        "confidence": 0.9,
        "quote": "ахаха, отлично",
        "ambiguous": False,
    }
    svc = Both(db, reply_with_feedback(laugh))
    m = active(svc)
    talk(m, "Ахаха, отлично")  # первой реплике не на что реагировать
    assert signals(m) == []
    joke = talk(m, "Расскажи шутку")  # модель приписала смех, которого в реплике нет
    (guess,) = signals(m)
    assert guess["origin"] == "interpretation" and not guess["reliable"]
    talk(m, "Ахаха, отлично")
    signal = next(s for s in signals(m) if s["response_id"] == joke)
    assert signal["reliable"] and signal["origin"] == "user_said"

    svc.output = reply_with_feedback({**laugh, "quote": "ок"})
    talk(m, "ок")
    bare = [s for s in signals(m) if s["quote"] == "ок"]
    assert len(bare) == 1 and not bare[0]["reliable"] and bare[0]["ambiguous"]
    traits = m.status()["personality"]["effective_traits"]
    assert (
        traits == CharacterModule().status()["personality"]["effective_traits"]
    )  # характер тот же
    m.dispose()


def test_cancelled_response_records_no_feedback(db: ModuleDatabase) -> None:
    svc = Both(db, with_candidates())
    m = active(svc)
    talk(m, "Расскажи шутку")
    svc.output = reply_with_feedback(
        {
            "dimension": "humor",
            "direction": 1,
            "confidence": 1,
            "quote": "смешно",
            "ambiguous": False,
        }
    )
    svc.block.clear()
    m.dialogue("start", {"text": "смешно", "model": "one"})
    deadline = time.monotonic() + 2
    while len(svc.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    m.dialogue("cancel")
    svc.block.set()
    assert finish(m)["job"]["state"] == "cancelled"
    assert signals(m) == []
    m.dispose()


def test_owner_rating_commands_export_and_history_deletion(db: ModuleDatabase) -> None:
    svc = Both(db, with_candidates())
    m = active(svc)
    assert "error" in m.dialogue("feedback", {"direction": 1})  # ещё нечего оценивать
    first = talk(m, "Привет")
    second = talk(m, "Как дела?")
    view = m.dialogue("feedback", {"direction": -1, "dimension": "verbosity"})["feedback"]
    assert view["signals"][0]["response_id"] == second  # по умолчанию — последний ответ
    m.dialogue("feedback", {"direction": 1, "response_id": first})
    assert "error" in m.dialogue("feedback", {"direction": 1, "response_id": "missing"})
    assert "error" in m.dialogue("feedback", {"direction": 2})
    exported = m.dialogue("memory_export")["memory"]
    assert {s["response_id"] for s in exported["feedback"]} == {first, second}

    m.dialogue("history_forget", {"ids": [second]})
    left = m.dialogue("feedback_view")["feedback"]["signals"]
    assert [s["response_id"] for s in left] == [first]  # оценка ушла вместе с репликой
    assert m.dialogue("feedback_clear")["feedback"] == {"preferences": [], "signals": []}
    m.dispose()
