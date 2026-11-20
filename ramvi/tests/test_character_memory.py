"""9.14 (CHAR-14): схемы памяти и Character State в своей базе модуля, пространства, перезапуск."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.emotions import EmotionalState
from ramvi_character.memory import MemoryStore
from ramvi_service.contracts.modules import Manifest
from ramvi_service.module_data import ModuleDatabase, load_migrations
from ramvi_service.modules.host import ModuleHost
from ramvi_service.scoped import ScopedData
from ramvi_storage.database import ConstraintError

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "modules" / "character" / "migrations"


class DataServices:
    """Те же вызовы data.query/data.execute, что у модуля, поверх настоящей базы модуля."""

    def __init__(self, database: ModuleDatabase) -> None:
        self.database = database

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "data.query":
            return self.database.query(payload["sql"], payload.get("params"))
        if method == "data.execute":
            return self.database.execute(payload["sql"], payload.get("params"))
        raise AssertionError(method)


def open_db(root: Path) -> ModuleDatabase:
    return ModuleDatabase(ScopedData(root, "character"), load_migrations(MIGRATIONS)).open()


def clock(values: list[str]) -> Any:
    return lambda: values[0]


def fact(identifier: str = "memory_001", **changes: Any) -> dict[str, Any]:
    item = {
        "id": identifier,
        "kind": "semantic",
        "type": "learning_difficulty",
        "subject": "english_pronouns",
        "value": "Часто путает he и she",
        "confidence": 0.94,
        "status": "active",
        "origin": "user_said",
        "source_ids": ["event_14"],
    }
    item.update(changes)
    return item


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    database = open_db(tmp_path / "data")
    try:
        yield database
    finally:
        database.close()


def test_record_has_every_spec_field_and_survives_reopen(tmp_path: Path) -> None:
    now = ["2026-10-20T08:00:00Z"]
    first = open_db(tmp_path / "data")
    store = MemoryStore(DataServices(first), clock=clock(now))
    store.put(fact())
    now[0] = "2026-11-13T09:30:00Z"
    store.observe("memory_001", ["event_39", "event_14"])
    first.close()

    second = open_db(tmp_path / "data")  # перезапуск: новое соединение, миграция не повторяется
    try:
        saved = MemoryStore(DataServices(second)).get("memory_001")
        assert second.schema_version == 2
    finally:
        second.close()
    assert saved == {
        "id": "memory_001",
        "kind": "semantic",
        "type": "learning_difficulty",
        "subject": "english_pronouns",
        "value": "Часто путает he и she",
        "confidence": 0.94,
        "occurrences": 2,
        "first_seen": "2026-10-20T08:00:00Z",
        "last_seen": "2026-11-13T09:30:00Z",
        "status": "active",
        "origin": "user_said",
        "source_ids": ["event_14", "event_39"],
        "expires_at": None,
        "importance": 0.5,
        "unfinished": False,
        "topics": [],
        "entities": [],
    }


def test_all_memory_kinds_are_stored_separately_from_character_state(db: ModuleDatabase) -> None:
    store = MemoryStore(DataServices(db))
    for kind in ("working", "profile", "episodic", "semantic", "behavioral"):
        store.put(fact(kind, kind=kind))
    store.save_state({"emotional_state": {"components": {"joy": 0.5}}})
    assert {r["kind"] for r in store.records()} == {
        "working",
        "profile",
        "episodic",
        "semantic",
        "behavioral",
    }
    assert [r["id"] for r in store.records(kind="profile")] == ["profile"]
    assert store.load_state() == {"emotional_state": {"components": {"joy": 0.5}}}
    assert store.save_state({"emotional_state": {}}) == 2  # ревизия растёт
    with sqlite3.connect(db.path) as raw:
        assert raw.execute("select count(*) from character_state").fetchone() == (1,)


def test_users_and_characters_do_not_see_each_other(db: ModuleDatabase) -> None:
    services = DataServices(db)
    owner = MemoryStore(services, user_id="owner")
    guest = MemoryStore(services, user_id="guest")
    other_character = MemoryStore(services, user_id="owner", character_id="teacher")
    owner.put(fact("same_id", value="владелец"))
    guest.put(fact("same_id", value="гость"))
    owner.save_state({"who": "owner"})

    assert owner.get("same_id")["value"] == "владелец"  # type: ignore[index]
    assert guest.get("same_id")["value"] == "гость"  # type: ignore[index]
    assert other_character.get("same_id") is None
    assert other_character.records() == [] and other_character.load_state() is None
    assert guest.load_state() is None
    assert guest.forget("same_id") and owner.get("same_id") is not None
    with pytest.raises(KeyError):
        other_character.observe("same_id", ["x"])


def test_unconfirmed_knowledge_cannot_become_an_active_fact(db: ModuleDatabase) -> None:
    store = MemoryStore(DataServices(db))
    for origin in ("interpretation", "assumption"):
        with pytest.raises(ValueError, match="cannot be active"):
            store.put(fact(origin, origin=origin))
        store.put(fact(origin, origin=origin, status="candidate"))
    # Ограничение есть и в самой схеме, а не только в коде модуля.
    with pytest.raises(ConstraintError, match="CHECK constraint"):
        db.execute(
            "UPDATE memory_record SET status = 'active' WHERE origin = 'assumption'",
        )
    assert store.get("assumption")["status"] == "candidate"  # type: ignore[index]


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "dream"},
        {"confidence": 1.5},
        {"confidence": float("nan")},
        {"occurrences": 0},
        {"status": "confirmed"},
        {"source_ids": ["bad id"]},
        {"first_seen": "2026-11-13T00:00:00Z", "last_seen": "2026-11-01T00:00:00Z"},
        {"expires_at": "2026-12-01T00:00:00Z"},  # истекает только рабочая память
        {"secret": "лишнее поле"},
        {"value": ""},
    ],
)
def test_invalid_records_are_rejected(db: ModuleDatabase, changes: dict[str, Any]) -> None:
    store = MemoryStore(DataServices(db))
    with pytest.raises(ValueError):
        store.put(fact(**changes))
    assert store.records() == []


def test_expired_working_memory_is_not_returned(db: ModuleDatabase) -> None:
    now = ["2026-11-20T10:00:00Z"]
    store = MemoryStore(DataServices(db), clock=clock(now))
    store.put(fact("task", kind="working", expires_at="2026-11-20T11:00:00Z"))
    assert [r["id"] for r in store.records()] == ["task"]
    now[0] = "2026-11-20T12:00:00Z"
    assert store.records() == [] and store.get("task") is not None


def test_real_module_applies_migration_and_restores_state_after_restart(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "modules" / "character" / "manifest.json").read_text())

    def host() -> ModuleHost:
        h = ModuleHost(tmp_path / "data")
        h.register(
            Manifest(
                "character",
                manifest["version"],
                manifest["entrypoint"],
                permissions=("data",),
            ),
            approved_permissions=("data",),
            sql_migrations=load_migrations(MIGRATIONS),
        )
        return h

    first = host()
    try:
        first.activate("character")
        status = first.status("character")["character"]
        assert status["state"] == "active"
    finally:
        first.close()  # checkpoint пишет Character State в базу модуля
    path = tmp_path / "data" / "modules" / "character" / "data.sqlite3"
    with sqlite3.connect(path) as raw:
        (schema, state), *_ = raw.execute("select schema, state from character_state").fetchall()
        assert schema == "ramvi.character.live_state/1"
        assert json.loads(state)["emotional_state"]["schema"] == (
            "ramvi.character.emotional_state/1"
        )
        felt = EmotionalState().identifiers()[1]
        saved = {
            "emotional_state": {
                "schema": "ramvi.character.emotional_state/1",
                "updated_at": time.time(),
                "components": {felt: 0.9},
                "custom": [],
            }
        }
        raw.execute("update character_state set state = ?", [json.dumps(saved)])

    second = host()
    try:
        second.activate("character")
        result = second.invoke("character", "status")
    finally:
        second.close()
    assert result["memory"]["available"] and result["memory"]["namespace"] == "u:owner/c:default"
    assert result["emotional_state"]["primary"]["id"] == felt  # база важнее JSON-checkpoint
