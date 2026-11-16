"""3.2: хранилище SQLite (WAL), миграции по владельцам, репозитории сущностей ТЗ §10."""

import shutil
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_storage import (
    CORE_MIGRATIONS,
    ConstraintError,
    Database,
    Migration,
    MigrationError,
    applied_version,
    migrate,
    open_core,
)
from ramvi_storage.database import _split_sql
from ramvi_storage.repositories import (
    AgentRuns,
    Chats,
    Downloads,
    Model,
    ModelProfile,
    Models,
    NotFound,
)

CORE_TABLES = {
    "models",
    "model_profiles",
    "chats",
    "messages",
    "agent_runs",
    "tool_calls",
    "client_keys",
    "downloads",
}
SHA = "a" * 64


def one(db: Database, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row:
    row = db.one(sql, params)
    assert row is not None
    return row


def tables(db: Database) -> set[str]:
    rows = db.all("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {r[0] for r in rows}


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = open_core(tmp_path)
    yield database
    database.close()


# --- база и схема ---------------------------------------------------------------------


def test_open_core_creates_schema_wal_and_foreign_keys(tmp_path: Path) -> None:
    with open_core(tmp_path) as db:
        assert (tmp_path / "ramvi.sqlite3").is_file()
        assert one(db, "PRAGMA journal_mode")[0] == "wal"
        assert one(db, "PRAGMA foreign_keys")[0] == 1
        assert CORE_TABLES | {"schema_migrations"} <= tables(db)
        assert applied_version(db, "core") == len(CORE_MIGRATIONS)
        assert db.integrity_ok()


def test_reopen_is_idempotent_and_keeps_data(tmp_path: Path) -> None:
    with open_core(tmp_path) as db:
        chat = Chats(db).create("первый", now=1.0)
        Chats(db).add_message(chat.id, "user", "привет", now=2.0)
    with open_core(tmp_path) as db:
        assert applied_version(db, "core") == len(CORE_MIGRATIONS)
        assert [m.content for m in Chats(db).messages(chat.id)] == ["привет"]
        assert len(db.all("SELECT * FROM schema_migrations")) == len(CORE_MIGRATIONS)


def test_moved_database_file_still_opens(tmp_path: Path) -> None:
    """PORT-01: в базе нет абсолютных путей к ней самой; закрытую базу можно перенести."""
    first = tmp_path / "a"
    with open_core(first) as db:
        Chats(db).create("перенос")
    moved = tmp_path / "b"
    moved.mkdir()
    shutil.copy2(first / "ramvi.sqlite3", moved / "ramvi.sqlite3")
    assert (
        not (first / "ramvi.sqlite3-wal").exists()
        or (first / "ramvi.sqlite3-wal").stat().st_size == 0
    )
    with open_core(moved) as db:
        assert [c.title for c in Chats(db).recent()] == ["перенос"]


def test_transaction_rolls_back_and_nests(db: Database) -> None:
    with pytest.raises(RuntimeError), db.transaction():
        db.execute("INSERT INTO chats VALUES ('x', 't', NULL, 1, 1)")
        with db.transaction():
            db.execute("INSERT INTO chats VALUES ('y', 't', NULL, 1, 1)")
        raise RuntimeError
    assert db.all("SELECT id FROM chats") == []


def test_constraint_errors_are_storage_errors(db: Database) -> None:
    with pytest.raises(ConstraintError):
        Chats(db).add_message("нет-такого-чата", "user", "x")
    with pytest.raises(ConstraintError):
        Downloads(db).create("m", "https://x", "m.part", "короткий")
    with pytest.raises(ConstraintError):
        db.execute(
            "INSERT INTO models (id, provider, architecture, path, installed_at,"
            " capabilities) VALUES ('m', 'p', 'a', 'f', 1, 'не json')"
        )


def test_two_connections_write_concurrently_without_losses(tmp_path: Path) -> None:
    """Два соединения (как два процесса) пишут в одну базу: WAL + busy_timeout, без потерь."""
    open_core(tmp_path).close()
    errors: list[BaseException] = []

    def writer(prefix: str) -> None:
        try:
            with Database(tmp_path / "ramvi.sqlite3") as conn:
                chats = Chats(conn)
                chat = chats.create(prefix)
                for i in range(150):
                    chats.add_message(chat.id, "user", f"{prefix}{i}")
        except BaseException as error:  # pragma: no cover - печатается в assert
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    with open_core(tmp_path) as db:
        assert one(db, "SELECT COUNT(*) FROM messages")[0] == 300
        assert db.integrity_ok()


def test_split_sql_handles_one_line_comments_and_strings() -> None:
    script = "CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT DEFAULT ';');\n-- конец\n"
    assert _split_sql(script) == [
        "CREATE TABLE a (x TEXT);",
        "CREATE TABLE b (y TEXT DEFAULT ';');",
    ]


# --- миграции -------------------------------------------------------------------------

V1 = Migration(1, "notes", "CREATE TABLE m_notes (id INTEGER PRIMARY KEY, text TEXT) STRICT;")
V2 = Migration(2, "notes tag", "ALTER TABLE m_notes ADD COLUMN tag TEXT;")


def test_owners_have_independent_chains(db: Database) -> None:
    assert migrate(db, "test.module", [V1]) == 1
    assert applied_version(db, "test.module") == 1
    assert applied_version(db, "core") == len(CORE_MIGRATIONS)
    assert migrate(db, "test.module", [V1, V2]) == 2
    assert applied_version(db, "core") == len(CORE_MIGRATIONS)
    assert applied_version(db, "other.module") == 0


def test_chain_must_be_contiguous_and_owner_valid(db: Database) -> None:
    with pytest.raises(MigrationError):
        migrate(db, "test.module", [V2])
    with pytest.raises(MigrationError):
        migrate(db, "../core", [V1])


def test_changed_applied_migration_is_rejected(db: Database) -> None:
    migrate(db, "test.module", [V1])
    edited = Migration(1, "notes", V1.sql.replace("text TEXT", "body TEXT"))
    with pytest.raises(MigrationError, match="was changed"):
        migrate(db, "test.module", [edited])


def test_newer_database_is_not_opened_by_older_code(db: Database) -> None:
    migrate(db, "test.module", [V1, V2])
    with pytest.raises(MigrationError, match="downgrade"):
        migrate(db, "test.module", [V1])


def test_failed_migration_leaves_no_trace(db: Database) -> None:
    migrate(db, "test.module", [V1])
    broken = Migration(
        2, "broken", "CREATE TABLE m_half (id INTEGER) STRICT; ALTER TABLE нет ADD COLUMN x;"
    )
    with pytest.raises(MigrationError, match="was not applied"):
        migrate(db, "test.module", [V1, broken])
    assert "m_half" not in tables(db)
    assert applied_version(db, "test.module") == 1


def test_python_migration_runs_in_same_transaction(db: Database) -> None:
    def fill(conn: Database) -> None:
        conn.execute("INSERT INTO m_notes (text) VALUES ('из миграции')")

    migrate(db, "test.module", [V1, Migration(2, "fill", run=fill)])
    assert one(db, "SELECT text FROM m_notes")[0] == "из миграции"


def test_backup_is_made_before_upgrade(tmp_path: Path) -> None:
    with open_core(tmp_path) as db:
        migrate(db, "test.module", [V1])
        db.execute("INSERT INTO m_notes (text) VALUES ('старое')")
        migrate(db, "test.module", [V1, V2], backup_dir=tmp_path / "backups")
    backups = list((tmp_path / "backups").glob("*.test.module.v1.*.bak"))
    assert len(backups) == 1
    with Database(backups[0]) as old:
        assert applied_version(old, "test.module") == 1
        assert one(old, "SELECT text FROM m_notes")[0] == "старое"
        assert old.integrity_ok()


def test_first_migration_of_fresh_database_makes_no_backup(tmp_path: Path) -> None:
    open_core(tmp_path).close()
    assert not (tmp_path / "backups").exists()


# --- чаты -----------------------------------------------------------------------------


def test_chats_order_messages_and_regenerate(db: Database) -> None:
    chats = Chats(db)
    old = chats.create("старый", now=10.0)
    new = chats.create("новый", now=20.0)
    assert [c.title for c in chats.recent()] == ["новый", "старый"]
    chats.add_message(old.id, "user", "вопрос", now=30.0)
    assert [c.title for c in chats.recent()] == ["старый", "новый"]
    # часы пошли назад — порядок сообщений всё равно по добавлению
    answer = chats.add_message(old.id, "assistant", "ответ", now=5.0, token_usage={"out": 3})
    chats.add_message(old.id, "user", "ещё", now=6.0)
    assert [m.content for m in chats.messages(old.id)] == ["вопрос", "ответ", "ещё"]
    assert chats.messages(old.id)[1].token_usage == {"out": 3}
    assert chats.delete_messages_from(old.id, answer.id) == 2
    assert [m.content for m in chats.messages(old.id)] == ["вопрос"]
    with pytest.raises(ValueError):
        chats.add_message(new.id, "robot", "x")
    assert chats.update(new.id, title="переименован").title == "переименован"
    assert chats.clear(old.id) == 1
    chats.add_message(new.id, "user", "x")
    chats.delete(new.id)
    assert one(db, "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (new.id,))[0] == 0
    with pytest.raises(NotFound):
        chats.get(new.id)


def test_message_reasoning_finish_attachments_roundtrip(db: Database) -> None:
    chats = Chats(db)
    chat = chats.create("")
    files = [{"name": "a.txt", "bytes": 3, "text": "abc"}]
    chats.add_message(chat.id, "user", "вопрос", attachments=files)
    chats.add_message(chat.id, "assistant", "полуответ", reasoning="думаю", finish="cancelled")
    user, answer = chats.messages(chat.id)
    assert user.attachments == files and user.reasoning == "" and user.finish is None
    assert (answer.reasoning, answer.finish, answer.attachments) == ("думаю", "cancelled", None)
    with pytest.raises(ValueError):
        chats.add_message(chat.id, "assistant", "x", finish="exploded")
    with pytest.raises(ConstraintError):  # CHECK в базе, мимо проверки репозитория
        db.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at, finish)"
            " VALUES ('m', ?, 'assistant', '', 0, 'exploded')",
            (chat.id,),
        )


def test_v1_database_upgrades_to_v2_keeping_messages(tmp_path: Path) -> None:
    with Database(tmp_path / "ramvi.sqlite3") as old:
        migrate(old, "core", CORE_MIGRATIONS[:1])
        old.execute("INSERT INTO chats VALUES ('c', 'старый', NULL, 1, 1)")
        old.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at)"
            " VALUES ('m', 'c', 'user', 'до обновления', 1)"
        )
    with open_core(tmp_path) as db:
        assert applied_version(db, "core") == 2
        (message,) = Chats(db).messages("c")
        assert (message.content, message.reasoning, message.finish) == ("до обновления", "", None)
    assert len(list((tmp_path / "backups").glob("*.core.v1.*.bak"))) == 1


# --- модели ---------------------------------------------------------------------------


def test_models_profiles_and_cascades(db: Database) -> None:
    models = Models(db)
    model = Model(
        "qwen3-14b-q4",
        "local",
        "qwen3",
        "models/q.gguf",
        1.0,
        quant="Q4_K_M",
        sha256=SHA,
        bytes=9,
        capabilities={"tools": True},
    )
    assert models.upsert(model).capabilities == {"tools": True}
    assert models.upsert(Model(**{**model.__dict__, "bytes": 10})).bytes == 10
    plan = {"resident": 21, "stream": 19}
    models.add_profile(ModelProfile(model.id, "hw1", 4096, plan, 1.0, tokps=8.0))
    models.add_profile(ModelProfile(model.id, "hw1", 4096, plan, 2.0, tokps=9.0))
    models.add_profile(ModelProfile(model.id, "hw2", 4096, plan, 3.0, tokps=1.0))
    latest = models.latest_profile(model.id, "hw1")
    assert latest is not None and latest.tokps == 9.0 and latest.plan == plan
    assert models.latest_profile(model.id, "hw3") is None
    chat = Chats(db).create("с моделью", model.id)
    models.delete(model.id)
    assert Chats(db).get(chat.id).model_id is None
    assert one(db, "SELECT COUNT(*) FROM model_profiles")[0] == 0
    with pytest.raises(NotFound):
        models.delete(model.id)


# --- загрузки -------------------------------------------------------------------------


def test_downloads_survive_restart(tmp_path: Path) -> None:
    with open_core(tmp_path) as db:
        downloads = Downloads(db)
        first = downloads.create("qwen3-8b", "https://example/a.gguf", "a.gguf.part", SHA.upper())
        done = downloads.create("qwen3-0.6b", "https://example/b.gguf", "b.gguf.part", SHA)
        downloads.update(first.id, state="paused", bytes_done=100, total_bytes=1000)
        downloads.update(done.id, state="done")
        with pytest.raises(ValueError):
            downloads.update(first.id, state="lost")
    with open_core(tmp_path) as db:
        unfinished = Downloads(db).unfinished()
        assert [(d.id, d.state, d.bytes_done, d.checksum) for d in unfinished] == [
            (first.id, "paused", 100, SHA)
        ]


# --- агент ----------------------------------------------------------------------------


def test_agent_runs_and_tool_calls(db: Database) -> None:
    runs = AgentRuns(db)
    run = runs.create("починить тест", "workspace", now=1.0)
    assert run.status == "planned" and run.ended_at is None
    runs.update(run.id, status="running", steps=1)
    runs.add_tool_call(run.id, "read_file", {"path": "a.py"}, "auto", result="ok", duration=0.1)
    runs.add_tool_call(run.id, "shell", {"cmd": "<redacted>"}, "approved")
    assert [c.tool_name for c in runs.tool_calls(run.id)] == ["read_file", "shell"]
    with pytest.raises(ValueError):
        runs.add_tool_call(run.id, "x", {}, "maybe")
    finished = runs.update(run.id, status="done", token_usage={"in": 5}, now=9.0)
    assert finished.ended_at == 9.0 and finished.token_usage == {"in": 5}
    with pytest.raises(ValueError, match="already finished"):
        runs.update(run.id, status="running")
    assert runs.recent()[0].id == run.id


def test_key_store_shares_core_database(db: Database) -> None:
    from ramvi_service.client_keys import KeyStore

    store = KeyStore(db)
    record, key = store.create("редактор", {"chat"})
    assert store.verify(key, "chat") is not None
    row = one(db, "SELECT client_name, secret_hash FROM client_keys WHERE id = ?", (record.id,))
    assert row[0] == "редактор" and key not in row[1]
    store.close()  # чужую базу KeyStore не закрывает
    assert db.integrity_ok()
