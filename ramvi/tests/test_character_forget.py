"""9.17 (CHAR-17): просмотр, исправление, экспорт и удаление памяти и истории без возврата."""

import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.consolidate import Consolidator
from ramvi_character.memory import MemoryStore
from ramvi_character.module import CharacterModule
from ramvi_character.recall import Recall
from ramvi_service.module_data import ModuleDatabase, load_migrations
from ramvi_service.scoped import ScopedData
from test_character_consolidation import Both, candidate, stored, with_candidates
from test_character_dialogue import active, finish
from test_character_memory import MIGRATIONS, DataServices, open_db

SECRET = "Квазиморда"


def fact(identifier: str, value: str, **changes: Any) -> dict[str, Any]:
    item = {
        "id": identifier,
        "kind": "semantic",
        "type": "fact",
        "subject": changes.pop("subject", "пароль"),
        "value": value,
        "confidence": 0.9,
        "status": "active",
        "origin": "user_said",
        "source_ids": changes.pop("source_ids", ["response:r1", "recent:r1u"]),
    }
    item.update(changes)
    return item


def files_contain(folder: Path, text: str) -> list[str]:
    needles = [text.encode(), text.casefold().encode()]
    return [
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and any(n in p.read_bytes() for n in needles)
    ]


def test_forgotten_text_leaves_no_bytes_in_db_journal_index_or_backups(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first, second, *_ = load_migrations(MIGRATIONS)
    old = ModuleDatabase(ScopedData(root, "character"), [first, second]).open()
    MemoryStore(DataServices(old)).put(fact("secret", f"Кодовое слово {SECRET}"))
    old.close()
    db = open_db(root)  # следующая миграция оставляет копию базы с секретом в backups/
    try:
        store = MemoryStore(DataServices(db))
        store.put(fact("keep", "Любит чай", subject="напиток"))
        assert files_contain(root, SECRET)
        assert store.forget("secret") and store.get("secret") is None
        assert "secret" not in [r["id"] for r in Recall(store).search(SECRET)]
        assert store.compact()["backups_removed"] >= 1
        assert files_contain(root, SECRET) == []  # ни в базе, ни в WAL, ни в FTS, ни в копиях
        assert store.get("keep") is not None
    finally:
        db.close()
    assert files_contain(root, SECRET) == []


def test_forgotten_knowledge_is_not_restored_from_the_same_sources(db: ModuleDatabase) -> None:
    store = MemoryStore(DataServices(db))
    c = Consolidator(store)
    store.put(fact("city", "Живёт в Киеве", subject="город"))
    assert store.forget("city")
    owner = {"current": "Скажи что-нибудь", "recent:r1u": "Я живу в Киеве"}
    # Модель снова предлагает то же из той же старой реплики — не восстанавливается.
    again = candidate("Живёт в Киеве", subject="город", evidence=["recent:r1u"])
    assert c.apply("r9", again, owner) is None
    # И перефраз без новой реплики владельца — тоже.
    paraphrase = candidate("Он живёт в городе Киеве", subject="Город", evidence=["recent:r7u"])
    assert c.apply("r10", paraphrase, {"recent:r7u": "ну да"}) is None
    assert store.records() == [] and c.stats["forgotten"] == 2
    # Новое явное заявление владельца в текущей реплике — его право.
    fresh = candidate("Живёт в Киеве", subject="город", quote="я живу в Киеве")
    restored = c.apply("r11", fresh, {"current": "Я живу в Киеве, запомни"})
    assert restored is not None and restored["source_ids"][:2] == ["response:r11", "recent:r11u"]
    # Другая тема не задета следом.
    other = c.apply("r12", candidate("Любит кофе", subject="напиток"), {"current": "кофе"})
    assert other is not None


def test_deletion_survives_restart_and_other_users_are_untouched(tmp_path: Path) -> None:
    root = tmp_path / "data"
    db = open_db(root)
    owner, guest = MemoryStore(DataServices(db)), MemoryStore(DataServices(db), user_id="guest")
    for n in range(3):
        owner.put(fact(f"o{n}", f"Факт владельца {n}", subject=f"тема {n}"))
    guest.put(fact("o0", "Факт гостя", subject="тема 0"))
    assert owner.forget_all() == 3
    owner.compact()
    db.close()

    db = open_db(root)  # перезапуск
    try:
        owner, guest = MemoryStore(DataServices(db)), MemoryStore(DataServices(db), user_id="guest")
        assert owner.records() == [] and Recall(owner).search("Факт владельца") == []
        assert [r["value"] for r in guest.records()] == ["Факт гостя"]
        blocked = candidate("Факт владельца 1", subject="тема 1", evidence=["recent:r1u"])
        assert Consolidator(owner).apply("r5", blocked, {"recent:r1u": "x"}) is None
        assert Consolidator(guest).apply("r5", blocked, {"recent:r1u": "x"}) is not None
    finally:
        db.close()


def test_owner_correction_replaces_and_forgets_wrong_version(db: ModuleDatabase) -> None:
    store = MemoryStore(DataServices(db))
    c = Consolidator(store)
    store.put(
        fact(
            "dog",
            "Собаку зовут Шарик",
            subject="собака",
            status="candidate",
            origin="interpretation",
            topics=["питомцы"],
        )
    )
    fixed = c.correct("dog", "Собаку зовут Бублик")
    assert fixed["value"] == "Собаку зовут Бублик" and fixed["status"] == "active"
    assert fixed["origin"] == "user_said" and fixed["topics"] == ["питомцы"]
    assert store.get("dog") is None and [r["id"] for r in store.records()] == [fixed["id"]]
    wrong = candidate("Собаку зовут Шарик", subject="собака", evidence=["recent:r1u"])
    assert c.apply("r3", wrong, {"recent:r1u": "Шарик"}) is None
    with pytest.raises(KeyError):
        c.correct("dog", "x")


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def talk(m: CharacterModule, text: str) -> str:
    response = m.dialogue("start", {"text": text, "model": "one"})
    assert finish(m)["job"]["state"] == "completed"
    return str(response["response_id"])


def test_module_commands_view_correct_export_and_delete_history(db: ModuleDatabase) -> None:
    proposal = candidate("Собаку зовут Бублик", subject="собака", quote="Собаку зовут Бублик")
    svc = Both(db, with_candidates(proposal))
    m = active(svc)
    first = talk(m, "Собаку зовут Бублик")
    (record,) = stored(m)
    svc.output = with_candidates()
    second = talk(m, "Как дела?")

    exported = m.dialogue("memory_export")["memory"]
    assert exported["schema"] == "ramvi.character.memory_export/1"
    assert [r["id"] for r in exported["records"]] == [record["id"]]
    assert [t["id"] for t in exported["history"]] == [first, second]
    assert exported["character_state"] is None or "emotional_state" in exported["character_state"]

    corrected = m.dialogue("memory_correct", {"id": record["id"], "value": "Собаку зовут Тузик"})
    (fixed,) = corrected["memory"]["records"]
    assert fixed["value"] == "Собаку зовут Тузик" and fixed["source_ids"] == ["owner:correction"]
    assert "error" in m.dialogue("memory_correct", {"id": fixed["id"], "value": " "})

    # Удаление реплики: запись, выученная только из неё, тоже забывается.
    svc.output = with_candidates(
        candidate("Любит гулять", subject="прогулки", quote="люблю гулять")
    )
    third = talk(m, "Я люблю гулять")
    assert {r["subject"] for r in stored(m)} == {"собака", "прогулки"}
    result = m.dialogue("history_forget", {"ids": [third]})["history"]
    assert result == {"removed": [third], "memory_forgotten": 1}
    assert {r["subject"] for r in stored(m)} == {"собака"}  # исправление владельца осталось
    assert [t["id"] for t in m.dialogue()["history"]] == [first, second]
    assert "error" in m.dialogue("history_forget", {"ids": ["missing"]})
    m.dispose()

    restarted = active(svc)  # те же данные модуля: история и память переживают перезапуск
    assert [t["id"] for t in restarted.dialogue()["history"]] == [first, second]
    assert restarted.dialogue("history_clear")["history"]["removed"] == [first, second]
    assert restarted.dialogue("memory_clear")["memory"]["records"] == []
    restarted.dispose()
    again = active(svc)
    assert again.dialogue()["history"] == [] and stored(again) == []
    again.dispose()


def test_history_delete_stops_running_response(db: ModuleDatabase) -> None:
    svc = Both(db, with_candidates(candidate("Факт", subject="ф", quote="Факт")))
    m = active(svc)
    first = talk(m, "Факт")
    svc.block.clear()
    m.dialogue("start", {"text": "Ещё", "model": "one"})
    deadline = time.monotonic() + 2
    while len(svc.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    removed = m.dialogue("history_clear")["history"]["removed"]
    svc.block.set()
    assert first in removed
    assert m._dialogue.thread is None or not m._dialogue.thread.is_alive()
    assert stored(m) == [] and m.dialogue()["history"] == []
    m.dispose()


def test_deleted_turn_forgets_only_what_was_learned_from_it_alone(db: ModuleDatabase) -> None:
    store = MemoryStore(DataServices(db))
    store.put(
        fact(
            "both",
            "Любит чай",
            subject="напиток",
            source_ids=["response:r1", "recent:r1u", "response:r2", "recent:r2u"],
        )
    )
    store.put(fact("only", "Любит кофе", subject="кофе", source_ids=["response:r2", "recent:r2u"]))
    gone = {"response:r2", "recent:r2u", "recent:r2a"}
    assert store.forget_sources(gone) == 1
    assert store.get("only") is None
    assert store.get("both")["source_ids"] == ["response:r1", "recent:r1u"]  # type: ignore[index]
