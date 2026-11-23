"""9.15 (CHAR-15): комбинированный поиск памяти, ограниченное чтение, бюджет контекста."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.context import Budget, ContextOrchestrator
from ramvi_character.memory import MemoryStore
from ramvi_character.recall import Recall
from ramvi_character.words import terms
from ramvi_service.module_data import ModuleDatabase
from test_character_dialogue import Services, active, finish
from test_character_memory import DataServices, open_db

NOW = "2026-11-20T10:00:00Z"


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def store(db: ModuleDatabase, user: str = "owner") -> MemoryStore:
    return MemoryStore(DataServices(db), user_id=user, clock=lambda: NOW)


def put(s: MemoryStore, identifier: str, value: str, **changes: Any) -> None:
    record = {
        "id": identifier,
        "kind": "semantic",
        "type": "fact",
        "subject": changes.pop("subject", identifier),
        "value": value,
        "confidence": 0.8,
        "status": "active",
        "origin": "user_said",
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-11-19T10:00:00Z",
        "importance": 0.3,
    }
    record.update(changes)
    s.put(record)


def ids(found: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in found]


def test_each_signal_can_find_a_record(db: ModuleDatabase) -> None:
    s = store(db)
    put(s, "files", "Опять забыл сохранить файл и потерял работу")
    put(s, "rust", "Хочет попробовать системный язык", entities=["Rust"])
    put(s, "english", "Путает he и she", topics=["английский язык"])
    put(s, "exam", "Экзамен назначен на пятницу")
    put(
        s,
        "promise",
        "Обещал рассказать, как прошло собеседование",
        kind="episodic",
        unfinished=True,
    )
    put(s, "noise", "Любит зелёный чай")
    recall = Recall(s)

    assert ids(recall.search("Мои файлы снова пропали"))[0] == "files"  # текст + основа слова
    assert ids(recall.search("Стоит учить rust?"))[0] == "rust"  # сущность
    assert ids(recall.search("Проверь мой английский язык"))[0] == "english"  # тема из двух слов
    by_task = recall.search("Что мне повторить?", task="Готовлюсь к экзамену")
    assert ids(by_task)[0] == "exam" and by_task[0]["signals"]["task"] > 0  # задача
    unrelated = recall.search("Какая сегодня погода?")
    assert ids(unrelated)[0] == "promise"  # незавершённое событие всплывает само
    assert unrelated[0]["signals"]["unfinished"] == 1.0


def test_recency_importance_and_frequency_break_ties(db: ModuleDatabase) -> None:
    s = store(db)
    put(s, "old", "Любит кофе", last_seen="2026-05-01T00:00:00Z")
    put(s, "new", "Любит кофе")
    assert ids(Recall(s).search("кофе"))[:2] == ["new", "old"]
    put(s, "important", "Любит кофе", last_seen="2026-05-01T00:00:00Z", importance=1.0)
    found = ids(Recall(s).search("кофе"))
    assert found.index("important") < found.index("old")  # та же давность, выше значимость
    put(s, "frequent", "Любит кофе", last_seen="2026-05-01T00:00:00Z", occurrences=20)
    found = ids(Recall(s).search("кофе"))
    assert found.index("frequent") < found.index("old")  # та же давность, чаще встречалась


def test_embeddings_are_not_the_only_signal(db: ModuleDatabase) -> None:
    s = store(db)
    put(s, "lexical", "Пишет программы на Python", entities=["python"])
    put(s, "semantic_only", "Увлекается змеями", importance=0.0)
    for n in range(40):  # более значимые записи вытесняют semantic_only из «значимых»
        put(s, f"filler{n}", f"Запись номер {n}", importance=0.6)

    def similarity(query: str, texts: Sequence[str]) -> list[float]:
        return [1.0 if "змеями" in t else 0.0 for t in texts]

    found = Recall(s, similarity).search("Помоги с python")
    assert ids(found)[0] == "lexical"
    # Похожесть только пересортирует найденное индексами и не добавляет новых записей.
    assert "semantic_only" not in ids(found)
    # Запись попала в кандидаты по значимости и получила полную похожесть 1.0,
    # но без текста, тем и задачи не обгоняет совпадение по ним.
    put(s, "snakes_salient", "Увлекается змеями", importance=1.0)
    found = Recall(s, similarity).search("Помоги с python")
    salient = next(r for r in found if r["id"] == "snakes_salient")
    assert salient["signals"]["semantic"] == 1.0 and ids(found)[0] == "lexical"

    with pytest.raises(ValueError):
        Recall(s, lambda q, t: [0.5]).search("python")  # по значению на каждый текст


def test_search_reads_bounded_rows_not_whole_history(db: ModuleDatabase) -> None:
    s = store(db)
    for n in range(3000):
        put(s, f"m{n}", f"Обычное сообщение номер {n} про погоду и дела")
    put(s, "needle", "Аллергия на арахис", topics=["здоровье"], importance=0.9)
    recall = Recall(s)
    found = recall.search("Можно мне арахис?", limit=8)
    assert ids(found)[0] == "needle" and len(found) == 8
    assert recall.rows_read < 400, recall.rows_read  # из 3001 записи


def test_superseded_expired_and_other_users_are_not_found(db: ModuleDatabase) -> None:
    owner, guest = store(db), store(db, "guest")
    put(owner, "old_city", "Живёт в Киеве", status="superseded", topics=["город"])
    put(owner, "tmp", "Сейчас пишет отчёт", kind="working", expires_at="2026-11-20T09:00:00Z")
    put(guest, "guest_city", "Живёт во Львове", topics=["город"])
    assert Recall(owner).search("В каком городе я живу? Пишет отчёт") == []
    assert ids(Recall(guest).search("город")) == ["guest_city"]


def test_context_budget_drops_least_relevant_memory_first(db: ModuleDatabase) -> None:
    s = store(db)
    put(s, "best", "Готовится к экзамену по английскому", topics=["экзамен"], importance=0.9)
    for n in range(10):
        put(s, f"weak{n}", f"Мелкая заметка {n} " + "x" * 200, importance=0.1)
    recall = Recall(s)
    orchestrator = ContextOrchestrator(
        "character:local",
        memory_source=lambda _text, limit: recall.context_items(
            "Как подготовиться к экзамену?", "", "character:local", limit
        ),
    )
    budget = Budget()
    assert budget.sections is not None
    budget.sections["memory"] = 700
    prepared = orchestrator.build(
        task="Поддержать разговор",
        user_input="Как подготовиться к экзамену?",
        personality={
            "policy_revision": 0,
            "base": {"id": "neutral", "version": "1.0.0"},
            "effective_traits": {
                k: 0.5
                for k in (
                    "warmth",
                    "humor",
                    "sarcasm",
                    "strictness",
                    "curiosity",
                    "verbosity",
                    "initiative",
                    "empathy",
                )
            },
            "boundaries": {"teasing": True, "profanity": False},
        },
        state={"primary": {"id": "neutral", "intensity": 1}, "secondary": None},
        budget=budget,
    )
    memory = [m["content"] for m in prepared.messages if "retrieved_memory" in m["content"]]
    assert memory and "best" in memory[0]
    dropped = [d for d in prepared.accounting["dropped"] if d["section"] == "memory"]
    assert dropped and all(d["id"].startswith("weak") for d in dropped)
    assert "Поддержать разговор" in str(prepared.messages)  # задача не выброшена


def test_dialogue_puts_found_memory_into_the_request(db: ModuleDatabase) -> None:
    class Both(Services):
        def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
            if method.startswith("data.") and method not in ("data.get", "data.set"):
                return DataServices(db).call(method, payload)
            return super().call(method, payload)

    services = Both()
    m = active(services)
    assert m.status()["memory"]["available"]
    assert m._memory is not None
    put(m._memory, "dog", "Собаку зовут Бублик", entities=["бублик"], topics=["собака"])
    m.dialogue("start", {"text": "Как зовут мою собаку?", "model": "one"})
    finish(m)
    sent = str(services.requests[0]["messages"])
    assert "Собаку зовут Бублик" in sent and "memory:dog" in sent


def test_terms_drop_stop_words() -> None:
    assert terms("Как мне выучить английский язык?") == ["выучить", "английский", "язык"]


def test_records_from_before_the_search_migration_are_indexed(tmp_path: Path) -> None:
    from ramvi_service.module_data import load_migrations
    from ramvi_service.scoped import ScopedData
    from test_character_memory import MIGRATIONS

    first, *_ = load_migrations(MIGRATIONS)
    old = ModuleDatabase(ScopedData(tmp_path / "data", "character"), [first]).open()
    old.execute(
        "INSERT INTO memory_namespace VALUES ('u:owner/c:default', 'owner', 'default', ?)", [NOW]
    )
    old.execute(
        "INSERT INTO memory_record (namespace_id, id, kind, type, subject, value, confidence,"
        " occurrences, first_seen, last_seen, status, origin, source_ids) VALUES"
        " ('u:owner/c:default', 'cat', 'semantic', 'fact', 'кошка', 'Кошку зовут Мурка',"
        " 0.9, 1, ?, ?, 'active', 'user_said', '[]')",
        [NOW, NOW],
    )
    old.close()
    upgraded = open_db(tmp_path / "data")
    try:
        assert upgraded.schema_version == len(load_migrations(MIGRATIONS))
        found = Recall(store(upgraded)).search("Как зовут мою кошку?")
        assert ids(found) == ["cat"] and found[0]["importance"] == 0.5
    finally:
        upgraded.close()
