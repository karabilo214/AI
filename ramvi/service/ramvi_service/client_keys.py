"""7.2 API-02: ключи клиентов локального API.

Хранится только SHA-256 ключа; сам ключ показывается один раз при создании.
У каждого ключа имя, права, last_seen, счётчик запросов; отзыв — навсегда.
SQLite (WAL) — до общего хранилища 3.2 таблица живёт в своём файле.
"""

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PREFIX = "ramvi-"
PERMISSIONS = frozenset({"chat", "models", "agent"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS client_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    permissions TEXT NOT NULL,
    created REAL NOT NULL,
    last_seen REAL,
    requests INTEGER NOT NULL DEFAULT 0,
    revoked REAL
)
"""


@dataclass(frozen=True)
class ClientKey:
    id: str
    name: str
    permissions: frozenset[str]
    created: float
    last_seen: float | None
    requests: int
    revoked: float | None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def create(self, name: str, permissions: set[str] | frozenset[str]) -> tuple[ClientKey, str]:
        """Вернуть запись и ключ. Ключ больше нигде не сохраняется и не показывается."""
        name = name.strip()
        if not name:
            raise ValueError("нужно имя клиента")
        unknown = set(permissions) - PERMISSIONS
        if unknown or not permissions:
            raise ValueError(f"права из {sorted(PERMISSIONS)}, получено {sorted(permissions)}")
        key = PREFIX + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(6)
        with self._lock:
            self._db.execute(
                "INSERT INTO client_keys (id, name, key_hash, permissions, created) VALUES (?,?,?,?,?)",
                (key_id, name, _hash(key), ",".join(sorted(permissions)), time.time()),
            )
        return self.get(key_id), key

    def get(self, key_id: str) -> ClientKey:
        row = self._db.execute(
            "SELECT id, name, permissions, created, last_seen, requests, revoked FROM client_keys WHERE id=?",
            (key_id,),
        ).fetchone()
        if row is None:
            raise KeyError(key_id)
        return ClientKey(row[0], row[1], frozenset(row[2].split(",")), *row[3:])

    def list(self) -> list[ClientKey]:
        ids = [r[0] for r in self._db.execute("SELECT id FROM client_keys ORDER BY created")]
        return [self.get(i) for i in ids]

    def revoke(self, key_id: str) -> None:
        with self._lock:
            cur = self._db.execute(
                "UPDATE client_keys SET revoked=? WHERE id=? AND revoked IS NULL", (time.time(), key_id)
            )
        if cur.rowcount == 0:
            self.get(key_id)  # KeyError, если такого нет; повторный отзыв — без ошибки

    def verify(self, key: str | None, permission: str | None = None) -> ClientKey | None:
        """Ключ действителен и (если задано) имеет право: обновить last_seen и счётчик."""
        if not key or not key.startswith(PREFIX):
            return None
        digest = _hash(key)
        row = self._db.execute(
            "SELECT id, key_hash, permissions, revoked FROM client_keys WHERE key_hash=?", (digest,)
        ).fetchone()
        if row is None or not hmac.compare_digest(row[1], digest) or row[3] is not None:
            return None
        if permission is not None and permission not in row[2].split(","):
            return None
        with self._lock:
            self._db.execute(
                "UPDATE client_keys SET last_seen=?, requests=requests+1 WHERE id=?", (time.time(), row[0])
            )
        return self.get(row[0])


_CHAT_ROUTES = ("/api/chat", "/api/generate", "/v1/chat/completions", "/v1/completions", "/v1/messages")


def permission_for(path: str) -> str:
    """Генерация требует chat; список моделей, версии и состояние движка — models."""
    route = path.split("?", 1)[0].rstrip("/")
    return "chat" if route in _CHAT_ROUTES else "models"


def checker(store: KeyStore):
    """Функция для App(api_key=...): ключ действителен и имеет право на маршрут."""
    return lambda token, path: store.verify(token, permission_for(path)) is not None
