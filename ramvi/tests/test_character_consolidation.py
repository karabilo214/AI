"""9.16 (CHAR-16): кандидаты памяти, происхождение, подтверждение, противоречия и фон."""

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.consolidate import QUEUE, Consolidator, origin
from ramvi_character.memory import MemoryStore
from ramvi_character.module import CharacterModule
from ramvi_service.module_data import ModuleDatabase
from test_character_dialogue import Services, active, finish
from test_character_memory import DataServices, open_db
from test_character_response import records


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def candidate(text: str, **changes: Any) -> dict[str, Any]:
    item = {"id": "c1", "kind": "fact", "text": text, "confidence": 0.9, "evidence": ["current"]}
    item.update(changes)
    return item


def consolidator(db: ModuleDatabase) -> Consolidator:
    return Consolidator(MemoryStore(DataServices(db)))


def test_code_decides_origin_not_the_model() -> None:
    owner = {"current": "Меня зовут Саша, я живу в Киеве", "recent:r1u": "Люблю кофе"}
    said = candidate("Владельца зовут Саша", quote="меня зовут  Саша")
    assert origin(said, owner) == "user_said"  # дословно, без учёта регистра и пробелов
    assert origin(candidate("Зовут Саша", quote="меня зовут Петя"), owner) == "interpretation"
    assert origin(candidate("Любит кофе", evidence=["recent:r1u"]), owner) == "interpretation"
    # Цитата из ответа персонажа или памяти — не слова владельца.
    assert origin(candidate("x", quote="кофе", evidence=["recent:r1a"]), owner) == "assumption"
    assert origin(candidate("Наверное, скучает", confidence=0.3), owner) == "assumption"


def test_new_knowledge_is_never_an_active_fact_at_once(db: ModuleDatabase) -> None:
    c = consolidator(db)
    owner = {"current": "Меня зовут Саша"}
    said = candidate("Владельца зовут Саша", subject="имя", quote="Меня зовут Саша")
    first = c.apply("r1", said, owner)
    assert first["status"] == "candidate" and first["origin"] == "user_said"
    assert c.apply("r1", said, owner)["status"] == "candidate"  # тот же ответ — не повтор
    again = c.apply(
        "r2", candidate("Его зовут Саша", subject="Имя", quote="Меня зовут Саша"), owner
    )
    assert again["id"] == first["id"] and again["status"] == "active"
    assert again["occurrences"] == 3 and "response:r2" in again["source_ids"]

    guess = candidate("Скорее всего, программист", subject="работа")
    for response in ("r3", "r4", "r5"):
        result = c.apply(response, guess, owner)
    assert result["status"] == "candidate" and result["origin"] == "interpretation"


def test_contradiction_is_disputed_and_needs_the_owner(db: ModuleDatabase) -> None:
    c = consolidator(db)
    owner = {"current": "Я живу в Киеве"}
    kyiv = c.confirm(c.apply("r1", candidate("Живёт в Киеве", subject="город"), owner)["id"])
    assert kyiv["status"] == "active" and kyiv["origin"] == "user_said"

    lviv = c.apply("r2", candidate("Живёт во Львове", subject="город"), {"current": "Львов"})
    assert lviv["status"] == "disputed"
    assert c.store.get(kyiv["id"])["status"] == "active"  # type: ignore[index]

    moved = c.confirm(lviv["id"])
    assert moved["status"] == "active"
    assert c.store.get(kyiv["id"])["status"] == "superseded"  # type: ignore[index]


def test_two_unconfirmed_versions_are_both_disputed(db: ModuleDatabase) -> None:
    c = consolidator(db)
    a = c.apply("r1", candidate("Любимый цвет синий", subject="цвет"), {"current": "синий"})
    b = c.apply("r2", candidate("Любимый цвет красный", subject="цвет"), {"current": "красный"})
    assert b["status"] == "disputed"
    assert c.store.get(a["id"])["status"] == "disputed"  # type: ignore[index]


class Both(Services):
    def __init__(self, db: ModuleDatabase, output: str | None = None, **kw: Any) -> None:
        super().__init__(output, **kw)
        self.db = db

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method in ("data.query", "data.execute"):
            return DataServices(self.db).call(method, payload)
        return super().call(method, payload)


def with_candidates(*items: dict[str, Any]) -> str:
    r = records()
    r[-1]["memory_candidates"] = list(items)
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in r)


def stored(m: CharacterModule) -> list[dict[str, Any]]:
    assert m._consolidator is not None and m._consolidator.wait()
    return list(m.dialogue("memory")["memory"]["records"])


def test_completed_response_stores_candidate_with_checked_origin(db: ModuleDatabase) -> None:
    proposal = candidate("Собаку зовут Бублик", subject="собака", quote="собаку зовут Бублик")
    m = active(Both(db, with_candidates(proposal)))
    m.dialogue("start", {"text": "Мою собаку зовут Бублик", "model": "one"})
    assert finish(m)["job"]["state"] == "completed"
    (record,) = stored(m)
    assert record["origin"] == "user_said" and record["status"] == "candidate"
    assert record["source_ids"][0].startswith("response:")
    # Владелец подтверждает и удаляет через команды модуля.
    confirmed = m.dialogue("memory_confirm", {"id": record["id"]})["memory"]["records"]
    assert confirmed[0]["status"] == "active"
    assert m.dialogue("memory_forget", {"id": record["id"]})["memory"]["records"] == []
    assert "error" in m.dialogue("memory_forget", {"id": record["id"]})
    m.dispose()


def test_cancelled_invalid_and_fallback_responses_write_no_memory(db: ModuleDatabase) -> None:
    proposal = candidate("Факт", subject="факт", quote="Факт")
    # Прерванный ответ.
    svc = Both(db, with_candidates(proposal))
    svc.block.clear()
    m = active(svc)
    m.dialogue("start", {"text": "Факт", "model": "one"})
    deadline = time.monotonic() + 2
    while not svc.requests and time.monotonic() < deadline:
        time.sleep(0.005)
    m.dialogue("cancel")
    svc.block.set()
    assert finish(m)["job"]["state"] == "cancelled"
    assert stored(m) == []
    m.dispose()
    # Невалидный хвост после принятых сегментов.
    lines = with_candidates(proposal).splitlines()
    m = active(Both(db, "\n".join(lines[:2]) + '\n{"type":"bad"}\n'))
    m.dialogue("start", {"text": "Факт", "model": "one"})
    assert finish(m)["job"]["state"] == "error"
    assert stored(m) == []
    m.dispose()
    # Запасной ответ простым текстом память не пишет.
    svc = Both(db, "not JSON")
    original = svc.call

    def call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "llm.stream" and svc.requests:
            svc.output = "Обычный ответ."
        return original(method, payload)

    svc.call = call  # type: ignore[method-assign]
    m = active(svc)
    m.dialogue("start", {"text": "Факт", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed" and result["job"]["fallback"]
    assert stored(m) == []
    m.dispose()


class SlowStore(MemoryStore):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gate = threading.Event()
        self.calls = 0

    def by_subject(self, subject: str, kind: str | None = None) -> list[dict[str, Any]]:
        self.calls += 1
        self.gate.wait(timeout=5)
        return super().by_subject(subject, kind)


def test_background_is_bounded_and_stops_with_module(db: ModuleDatabase) -> None:
    slow = SlowStore(DataServices(db))
    c = Consolidator(slow)
    owner = {"current": "x"}
    accepted = [
        c.submit(f"r{n}", [candidate(f"Факт {n}", subject=f"s{n}")], owner) for n in range(40)
    ]
    assert accepted.count(False) >= 40 - QUEUE - 1  # очередь не растёт, речь не ждёт
    assert c.stats["dropped"] == accepted.count(False)
    c.stop(wait=False)
    slow.gate.set()
    assert c.wait(5)
    assert slow.calls <= 1  # после остановки очередь не дорабатывается
    assert not c.submit("late", [candidate("Поздно")], owner)

    m = active(Both(db))
    assert m._consolidator is not None
    background = m._consolidator
    m.suspend()
    assert m._consolidator is None and background.wait(1)
    m.resume()
    assert m._consolidator is not None and m._consolidator is not background
    m.deactivate()
    assert m._consolidator is None


def test_stop_halts_inside_one_response_with_many_candidates(db: ModuleDatabase) -> None:
    slow = SlowStore(DataServices(db))
    c = Consolidator(slow)
    many = [candidate(f"Факт {n}", id=f"c{n}", subject=f"s{n}") for n in range(4)]
    assert c.submit("r1", many, {"current": "x"})
    deadline = time.monotonic() + 2
    while slow.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    c.stop(wait=False)
    slow.gate.set()
    assert c.wait(5) and slow.calls == 1  # остальные кандидаты этого ответа не обработаны


def test_cancel_at_the_last_moment_writes_no_memory(db: ModuleDatabase) -> None:
    proposal = candidate("Факт", subject="факт", quote="Факт")
    m = active(Both(db, with_candidates(proposal)))
    finish_speech = m._speech.finish

    def late_cancel(*args: Any) -> Any:
        m._dialogue.cancelled.set()  # отмена пришла после разбора всего потока
        return finish_speech(*args)

    m._speech.finish = late_cancel  # type: ignore[method-assign,assignment]
    m.dialogue("start", {"text": "Факт", "model": "one"})
    assert finish(m)["job"]["state"] == "cancelled"
    assert stored(m) == []
    m.dispose()
