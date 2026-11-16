"""Миграции схемы с владельцем.

У каждого владельца (`core` или модуль) своя цепочка версий 1, 2, 3 … в общей
таблице `schema_migrations`. Цепочки независимы: обновление модуля не трогает
версию ядра и других модулей. Это основа пространств данных модулей (пункт 3.12);
проверку, что модуль создаёт только свои таблицы, добавляет 3.12.

Правила:
- миграция применяется в своей транзакции целиком или не применяется вовсе;
- применённую миграцию менять нельзя: при расхождении контрольной суммы — ошибка;
- база новее кода (есть версии, неизвестные коду) не открывается — понижения нет;
- перед применением к непустой базе делается резервная копия файла.
"""

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ramvi_storage.database import Database, StorageError

OWNER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")

_BOOKKEEPING = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    owner TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at REAL NOT NULL,
    PRIMARY KEY (owner, version)
) STRICT
"""


class MigrationError(StorageError):
    pass


@dataclass(frozen=True)
class Migration:
    """Шаг схемы: SQL-скрипт и/или функция над открытой транзакцией.

    Контрольная сумма считается по SQL и имени; у функции — по `code_version`,
    который автор увеличивает при любом изменении функции.
    """

    version: int
    name: str
    sql: str = ""
    run: Callable[[Database], None] | None = None
    code_version: int = 1

    @property
    def checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(b"\0")
        digest.update(self.sql.strip().encode())
        if self.run is not None:
            digest.update(f"\0run:{self.code_version}".encode())
        return digest.hexdigest()


def _check_owner(owner: str) -> str:
    if not OWNER_RE.fullmatch(owner):
        raise MigrationError(f"invalid migration owner: {owner!r}")
    return owner


def _check_chain(migrations: Sequence[Migration]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration versions must be consecutive from 1, got {versions}")
    for m in migrations:
        if not m.name.strip() or (not m.sql.strip() and m.run is None):
            raise MigrationError(f"migration {m.version}: name and SQL or a function are required")


def _ensure_bookkeeping(db: Database) -> None:
    with db.transaction():
        db.execute(_BOOKKEEPING)


def applied_version(db: Database, owner: str) -> int:
    """Последняя применённая версия владельца; 0 — ничего не применено."""
    _ensure_bookkeeping(db)
    row = db.one(
        "SELECT MAX(version) FROM schema_migrations WHERE owner = ?", (_check_owner(owner),)
    )
    return int(row[0]) if row is not None and row[0] is not None else 0


def migrate(
    db: Database,
    owner: str,
    migrations: Sequence[Migration],
    *,
    backup_dir: str | Path | None = None,
) -> int:
    """Довести схему владельца до последней версии. Вернуть итоговую версию."""
    _check_owner(owner)
    _check_chain(migrations)
    _ensure_bookkeeping(db)
    applied = {
        int(r["version"]): str(r["checksum"])
        for r in db.all("SELECT version, checksum FROM schema_migrations WHERE owner = ?", (owner,))
    }
    known = {m.version: m for m in migrations}
    newer = sorted(v for v in applied if v not in known)
    if newer:
        raise MigrationError(
            f"{owner}: the database was created by a newer version (schema {max(newer)}, code knows "
            f"{len(migrations)}); downgrade is not supported"
        )
    for version, checksum in applied.items():
        if known[version].checksum != checksum:
            raise MigrationError(
                f"{owner}: migration {version} was changed after it was applied — not allowed, "
                "add a new migration"
            )
    pending = [m for m in migrations if m.version not in applied]
    if not pending:
        return len(migrations)
    if applied and backup_dir is not None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        name = f"{db.path.stem}.{owner}.v{max(applied)}.{stamp}.bak"
        db.backup(Path(backup_dir) / name)
    for m in pending:
        try:
            with db.transaction():
                if m.sql.strip():
                    db.executescript_in_transaction(m.sql)
                if m.run is not None:
                    m.run(db)
                db.execute(
                    "INSERT INTO schema_migrations (owner, version, name, checksum, applied_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (owner, m.version, m.name, m.checksum, time.time()),
                )
        except MigrationError:
            raise
        except Exception as error:
            raise MigrationError(
                f"{owner}: migration {m.version} ({m.name}) was not applied: {error}"
            ) from error
    return len(migrations)
