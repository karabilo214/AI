"""3.12 (CORE-DATA-01): своя база модуля, изоляция, откат, экспорт, сброс и удаление."""

import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.contracts.modules import Manifest
from ramvi_service.module_data import load_migrations
from ramvi_service.modules.host import ModuleHost
from ramvi_storage.repositories import Chats
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client

FIXTURE = "ramvi.tests.module_fixture:create"
NOTES = "CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL);"
CHARACTER = {
    "0001_profiles.sql": "CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL);",
    "0002_emotions_memory.sql": "CREATE TABLE emotions (id INTEGER PRIMARY KEY, state TEXT);\n"
    "CREATE TABLE memory (id INTEGER PRIMARY KEY, kind TEXT, text TEXT);\n"
    "CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL);",
}


def package(root: Path, module_id: str, files: dict[str, str]) -> Path:
    folder = root / module_id / "migrations"
    folder.mkdir(parents=True)
    for name, sql in files.items():
        (folder / name).write_text(sql)
    return folder


def register(host: ModuleHost, module_id: str, folder: Path) -> None:
    host.register(
        Manifest(module_id, "1.0.0", FIXTURE, permissions=("data",)),
        approved_permissions=("data",),
        sql_migrations=load_migrations(folder),
    )


@pytest.fixture
def host(tmp_path: Path):  # type: ignore[no-untyped-def]
    h = ModuleHost(tmp_path / "data")
    register(h, "agent", package(tmp_path / "pkg", "agent", {"0001_notes.sql": NOTES}))
    register(h, "character", package(tmp_path / "pkg", "character", CHARACTER))
    h.activate("agent")
    h.activate("character")
    try:
        yield h
    finally:
        h.close()


def tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as db:
        return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def db_file(host: ModuleHost, module_id: str) -> Path:
    return host.root / module_id / "data.sqlite3"


def test_character_registers_its_own_tables_not_core(host: ModuleHost, tmp_path: Path) -> None:
    assert {"profiles", "emotions", "memory", "notes"} <= tables(db_file(host, "character"))
    with open_core(tmp_path / "data") as core:
        core_tables = {
            r["name"] for r in core.all("select name from sqlite_master where type='table'")
        }
    assert not {"profiles", "emotions", "memory"} & core_tables  # Core таблицы персонажа не создаёт
    assert tables(db_file(host, "agent")) >= {"notes"}


def test_module_cannot_read_or_change_other_modules_records(host: ModuleHost) -> None:
    host.invoke("agent", "db_write", {"text": "секрет агента"})
    host.invoke("character", "db_write", {"text": "заметка персонажа"})

    assert host.invoke("agent", "db_read")["rows"] == [["секрет агента"]]
    assert host.invoke("character", "db_read")["rows"] == [
        ["заметка персонажа"]
    ]  # своя таблица notes
    agent_db = str(db_file(host, "agent"))
    attempts: list[tuple[str, bool]] = [
        (f"ATTACH DATABASE '{agent_db}' AS other", True),
        ("PRAGMA journal_mode=DELETE", False),  # вне транзакции прошла бы без авторизатора
        ("PRAGMA user_version = 7", True),
        ("CREATE TABLE stolen (x)", True),
        ("DROP TABLE notes", True),
        ("DELETE FROM schema_migrations", True),
        ("INSERT INTO notes(text) VALUES ('через query')", False),  # query — только чтение
        ("SELECT load_extension('x')", False),
    ]
    for sql, write in attempts:
        result = host.invoke("character", "db_raw", {"sql": sql, "write": write})
        assert "error" in result, sql
    assert host.invoke("agent", "db_read")["rows"] == [["секрет агента"]]
    assert host.status("character")["character"]["state"] == "active"  # отказ не уронил модуль


def test_failed_migration_rolls_back_whole_database(tmp_path: Path) -> None:
    folder = package(tmp_path / "pkg", "agent", {"0001_notes.sql": NOTES})
    first = ModuleHost(tmp_path / "data")
    register(first, "agent", folder)
    first.activate("agent")
    first.invoke("agent", "db_write", {"text": "было"})
    first.close()
    (folder / "0002_good.sql").write_text("CREATE TABLE extra (x);")
    (folder / "0003_bad.sql").write_text("CREATE TABLE broken (;")

    second = ModuleHost(tmp_path / "data")
    register(second, "agent", folder)
    with pytest.raises(Exception, match="0003|broken|migration"):
        second.activate("agent")
    try:
        assert second.status("agent")["agent"]["state"] == "error"
        path = second.root / "agent" / "data.sqlite3"
        assert "extra" not in tables(path)  # шаг 2 тоже откатился вместе с файлом
        with sqlite3.connect(path) as db:
            assert db.execute("select text from notes").fetchall() == [("было",)]
    finally:
        second.close()


def test_export_while_running(host: ModuleHost, tmp_path: Path) -> None:
    host.invoke("character", "db_write", {"text": "в экспорт"})

    result = host.export_data("character", tmp_path / "out" / "character.zip")

    assert result["module_id"] == "character" and result["sql_schema"] == 2
    with zipfile.ZipFile(result["path"]) as archive:
        names = set(archive.namelist())
        assert {"data.sqlite3", "EXPORT.json"} <= names
        assert not any(n.endswith(("-wal", "-shm")) or n.startswith("backups/") for n in names)
        archive.extract("data.sqlite3", tmp_path / "unzipped")
        meta = json.loads(archive.read("EXPORT.json"))
    with sqlite3.connect(tmp_path / "unzipped" / "data.sqlite3") as db:
        assert db.execute("select text from notes").fetchall() == [("в экспорт",)]
    assert meta["files"] >= 1 and meta["version"] == "1.0.0"
    assert host.status("character")["character"]["state"] == "active"


def test_reset_needs_stop_and_touches_only_this_module(host: ModuleHost) -> None:
    host.invoke("agent", "db_write", {"text": "агент"})
    host.invoke("character", "db_write", {"text": "персонаж"})
    with pytest.raises(RuntimeError, match="stop the module"):
        host.reset_data("character")

    host.dispose("character")
    host.reset_data("character")

    assert not db_file(host, "character").exists()
    assert host.invoke("agent", "db_read")["rows"] == [["агент"]]


def test_uninstall_character_keeps_chat_history_and_agent_data(
    host: ModuleHost, tmp_path: Path
) -> None:
    with open_core(tmp_path / "data") as core:
        chat = Chats(core).create("История", None)
        Chats(core).add_message(chat.id, "user", "привет")
    host.invoke("agent", "db_write", {"text": "данные агента"})
    host.invoke("character", "db_write", {"text": "память персонажа"})

    host.uninstall("character", purge_data=True)

    assert not db_file(host, "character").exists()
    assert host.invoke("agent", "db_read")["rows"] == [["данные агента"]]
    with open_core(tmp_path / "data") as core:
        assert [m.content for m in Chats(core).messages(chat.id)] == ["привет"]


def test_bad_migration_file_name_is_rejected(tmp_path: Path) -> None:
    folder = package(tmp_path / "pkg", "x", {"1_notes.sql": NOTES})
    with pytest.raises(ValueError, match="0001_name"):
        load_migrations(folder)


def manifest_dir(app: Path, module_id: str, files: dict[str, str]) -> None:
    folder = app / "modules" / module_id
    package(app / "modules", module_id, files)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "module_id": module_id,
                "version": "1.0.0",
                "entrypoint": FIXTURE,
                "permissions": ["data"],
                "ui": {
                    "section": {
                        "view": module_id,
                        "title_key": f"section.{module_id}",
                        "icon": module_id,
                    }
                },
            }
        )
    )


def test_module_data_over_http(tmp_path: Path) -> None:
    app = tmp_path / "app"
    manifest_dir(app, "agent", {"0001_notes.sql": NOTES})
    manifest_dir(app, "character", CHARACTER)
    server, engines = build(app / "data", TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        client = Client(server.port)
        client.ok("POST", "/sections/character/open")
        listed: dict[str, Any] = {
            m["module_id"]: m for m in client.ok("GET", "/modules")["modules"]
        }
        assert listed["character"]["state"] == "active" and listed["character"]["data_bytes"] > 0
        assert listed["agent"]["state"] == "registered" and listed["agent"]["data_bytes"] == 0
        exported = client.ok("POST", "/modules/character/export")["export"]
        assert (
            Path(exported["path"]).is_file()
            and Path(exported["path"]).parent == app / "data" / "exports"
        )
        assert client.ok("POST", "/modules/character/reset")["module"]["state"] == "disposed"
        client.ok("POST", "/modules/character/uninstall", {"purge": True})
        assert [s["id"] for s in client.ok("GET", "/sections")["sections"]] == ["chat", "models", "agent"]
        status, error = client.request("POST", "/modules/nope/reset")
        assert status == 400 and error["error"]["code"] == "module.not_found"
    finally:
        server.close()
        assert server.sections is not None
        server.sections.host.close()
        engines.stop()
