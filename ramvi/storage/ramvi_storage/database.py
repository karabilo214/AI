"""Соединение с SQLite: WAL, внешние ключи, транзакции, резервная копия."""

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

Params = Sequence[Any] | dict[str, Any]

MIN_SQLITE = (3, 37, 0)  # STRICT-таблицы


class StorageError(Exception):
    """Ошибка хранилища, понятная вызывающему коду (не сырая sqlite3)."""


class ConstraintError(StorageError):
    """Нарушено ограничение схемы: внешний ключ, уникальность, CHECK, NOT NULL."""


class Database:
    """Одно соединение на процесс сервиса; все обращения под общей блокировкой.

    `transaction()` — BEGIN IMMEDIATE: запись другого процесса ждёт `busy_timeout`,
    а не портит данные. Вложенные `transaction()` в одном потоке — одна транзакция.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise StorageError(f"SQLite >= 3.37 is required, installed {sqlite3.sqlite_version}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            self._conn.close()
            raise StorageError(f"could not enable WAL (mode {mode})")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

    def set_authorizer(self, callback: Any) -> None:
        """Проверка каждого действия SQL (sqlite3 authorizer); None — снять (3.12)."""
        with self._lock:
            self._conn.set_authorizer(callback)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outer:
                    self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Params = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._conn.execute(sql, params)
            except sqlite3.IntegrityError as error:
                raise ConstraintError(str(error)) from error

    def executescript_in_transaction(self, sql: str) -> None:
        """Выполнить несколько операторов внутри уже открытой транзакции.

        `sqlite3.executescript` сам делает COMMIT, поэтому операторы выполняются по одному.
        """
        with self._lock:
            if self._depth == 0:
                raise StorageError("executescript_in_transaction outside a transaction")
            for statement in _split_sql(sql):
                self._conn.execute(statement)

    def one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def all(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def backup(self, target: str | Path) -> Path:
        """Согласованная копия базы (онлайн backup API SQLite), без остановки работы."""
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            copy = sqlite3.connect(str(destination))
            try:
                self._conn.backup(copy)
            finally:
                copy.close()
        return destination

    def integrity_ok(self) -> bool:
        row = self.one("PRAGMA integrity_check")
        return row is not None and row[0] == "ok"


def _split_sql(sql: str) -> list[str]:
    """Разбить скрипт на операторы по `;` с учётом строк и комментариев (без триггеров)."""
    statements: list[str] = []
    start = 0
    for index, char in enumerate(sql):
        if char == ";" and sqlite3.complete_statement(sql[start : index + 1]):
            text = sql[start : index + 1].strip()
            if _has_code(text):
                statements.append(text)
            start = index + 1
    if _has_code(sql[start:]):
        raise StorageError("unterminated SQL statement in migration")
    return statements


def _has_code(text: str) -> bool:
    for line in text.splitlines():
        code = line.split("--", 1)[0].strip()
        if code and code != ";":
            return True
    return False
