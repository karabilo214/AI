"""3.12 (CORE-DATA-01): своя база SQLite у каждого модуля поверх хранилища 3.2.

- Файл `data.sqlite3` лежит в пространстве модуля (`data/modules/<id>/`): чужие записи
  недоступны, потому что это другой файл; сброс, удаление и миграция пространства
  захватывают базу вместе с остальными данными модуля.
- Таблицы объявляет сам модуль: SQL-миграции из его пакета
  (`modules/<id>/migrations/0001_<имя>.sql`). Core таблиц модулей не определяет.
- Миграции применяет Host при запуске модуля, с резервной копией; если шаг не прошёл,
  база восстанавливается из копии целиком, модуль переходит в ошибку, данные целы.
- Модуль работает с базой через вызовы `data.query` / `data.execute`. Авторизатор SQLite
  запрещает выход за файл (ATTACH), PRAGMA, изменение схемы вне миграций и служебную
  таблицу миграций.
"""

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ramvi_storage.database import Database
from ramvi_storage.migrations import Migration, MigrationError, applied_version, migrate

from ramvi_service.scoped import ScopedData

DB_FILE = "data.sqlite3"
OWNER = "module"  # владелец миграций внутри файла модуля
MAX_SQL = 100_000
MAX_ROWS = 10_000
MAX_PARAMS = 999
_MIGRATION_FILE = re.compile(r"(\d{4})_([a-z0-9_]{1,48})\.sql")
# data_version — счётчик изменений файла; его читает FTS5 внутри своих запросов.
_READ_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "index_list",
    "index_info",
    "foreign_key_list",
    "data_version",
}
_DDL = {
    sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW, sqlite3.SQLITE_CREATE_TEMP_TABLE, sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW, sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_TEMP_TABLE, sqlite3.SQLITE_DROP_TEMP_INDEX, sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW, sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DROP_VTABLE,
}  # fmt: skip
_WRITE = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}


def load_migrations(folder: Path) -> list[Migration]:
    """SQL-миграции модуля: `0001_<имя>.sql`, `0002_…` — версии подряд с 1."""
    if not folder.is_dir():
        return []
    result = []
    for path in sorted(folder.glob("*.sql")):
        match = _MIGRATION_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"migration name must be 0001_name.sql: {path.name}")
        result.append(
            Migration(int(match.group(1)), match.group(2), path.read_text(encoding="utf-8"))
        )
    return result


def _authorizer(read_only: bool) -> Any:
    def check(
        action: int, arg1: str | None, arg2: str | None, db: str | None, source: str | None
    ) -> int:
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH) or action in _DDL:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() not in _READ_PRAGMAS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in (
            "load_extension",
            "readfile",
            "writefile",
        ):
            return sqlite3.SQLITE_DENY
        if action in _WRITE and (read_only or (arg1 or "").lower() == "schema_migrations"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return check


def _scalar_params(params: Any) -> list[Any]:
    if params is None:
        return []
    if not isinstance(params, list) or len(params) > MAX_PARAMS:
        raise ValueError("params must be a list of up to 999 values")
    for value in params:
        if isinstance(value, bool) or (
            value is not None and not isinstance(value, (str, int, float))
        ):
            raise ValueError("params may hold only strings, numbers and null")
    return params


class ModuleDatabase:
    def __init__(self, data: ScopedData, migrations: Sequence[Migration]) -> None:
        self.data, self.migrations = data, list(migrations)
        self.path = data.path(DB_FILE)
        self._db: Database | None = None

    @property
    def schema_version(self) -> int:
        return 0 if self._db is None else applied_version(self._db, OWNER)

    def open(self) -> "ModuleDatabase":
        """Открыть и довести схему; при сбое — откат файла к состоянию до миграции."""
        existed = self.path.exists()
        backups = self.data.path("backups")
        before = set(backups.glob("*.bak")) if backups.exists() else set()
        db = Database(self.path)
        try:
            migrate(db, OWNER, self.migrations, backup_dir=backups)
        except MigrationError:
            db.close()
            created = set(backups.glob("*.bak")) - before if backups.exists() else set()
            self._discard()
            if existed and created:
                shutil.copyfile(max(created, key=lambda p: p.stat().st_mtime), self.path)
            raise
        # Удалённые модулем строки затираются в файле, а не остаются в свободных страницах.
        db.execute("PRAGMA secure_delete=ON")
        self._db = db
        return self

    def compact(self) -> dict[str, Any]:
        """После удаления данных: журнал WAL — в базу и обнулить, старые копии миграций — удалить.

        Копии нужны только на время самой миграции (откат при сбое); после удачного открытия
        в них могли остаться записи, которые пользователь уже удалил.
        """
        db = self._require("compact")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        removed = 0
        backups = self.data.path("backups")
        if backups.exists():
            for item in backups.glob("*.bak"):
                item.unlink()
                removed += 1
        return {"backups_removed": removed}

    def _discard(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.path) + suffix).unlink(missing_ok=True)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _require(self, sql: Any) -> Database:
        if self._db is None:
            raise RuntimeError("the module database is not open")
        if not isinstance(sql, str) or not sql.strip() or len(sql) > MAX_SQL:
            raise ValueError("sql must be a non-empty string of up to 100000 characters")
        return self._db

    def query(self, sql: str, params: Any = None) -> dict[str, Any]:
        """Только чтение; не больше MAX_ROWS строк."""
        db = self._require(sql)
        values = _scalar_params(params)
        db.set_authorizer(_authorizer(read_only=True))
        try:
            rows = db.all(sql, values)
        except sqlite3.DatabaseError as error:
            raise PermissionError(f"query rejected: {error}") from None
        finally:
            db.set_authorizer(None)
        if len(rows) > MAX_ROWS:
            raise ValueError(f"more than {MAX_ROWS} rows — add LIMIT")
        return {"columns": list(rows[0].keys()) if rows else [], "rows": [list(r) for r in rows]}

    def execute(self, sql: str, params: Any = None) -> dict[str, Any]:
        """Изменение данных (INSERT/UPDATE/DELETE) в одной транзакции; схема — только миграциями."""
        db = self._require(sql)
        values = _scalar_params(params)
        db.set_authorizer(_authorizer(read_only=False))
        try:
            with db.transaction():
                cursor = db.execute(sql, values)
        except sqlite3.DatabaseError as error:
            raise PermissionError(f"query rejected: {error}") from None
        finally:
            db.set_authorizer(None)
        return {"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}

    def snapshot(self, target: Path) -> None:
        """Согласованная копия базы (во время работы модуля)."""
        if self._db is not None:
            self._db.backup(target)
        elif self.path.exists():
            with Database(self.path) as db:
                db.backup(target)


def export_namespace(
    data: ScopedData, database: ModuleDatabase | None, target: Path, meta: dict[str, Any]
) -> dict[str, Any]:
    """ZIP пространства модуля: JSON-данные, согласованная копия базы, EXPORT.json."""
    target.parent.mkdir(parents=True, exist_ok=True)
    skip = {DB_FILE, DB_FILE + "-wal", DB_FILE + "-shm"}
    with tempfile.TemporaryDirectory(prefix=".export-") as folder:
        snapshot = Path(folder) / DB_FILE
        db = database or ModuleDatabase(data, [])
        db.snapshot(snapshot)
        tmp = target.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            files = 0
            for path in sorted(data.root.rglob("*")):
                relative = path.relative_to(data.root).as_posix()
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and relative not in skip
                    and not relative.startswith("backups/")
                ):
                    archive.write(path, relative)
                    files += 1
            if snapshot.exists():
                archive.write(snapshot, DB_FILE)
                files += 1
            info = {
                **meta,
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": files,
            }
            archive.writestr("EXPORT.json", json.dumps(info, ensure_ascii=False, indent=2))
        os.replace(tmp, target)
    return {"path": str(target), "bytes": target.stat().st_size, **info}
