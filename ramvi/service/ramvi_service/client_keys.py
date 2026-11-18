"""7.2 API-02: ключи клиентов локального API.

Хранится только SHA-256 ключа; сам ключ показывается один раз при создании.
У каждого ключа имя, права, last_seen, счётчик запросов; отзыв — навсегда.
Таблица `client_keys` — в общем хранилище ядра (`ramvi_storage`, пункт 3.2).
"""

import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramvi_storage import CORE_MIGRATIONS, CORE_OWNER, Database, migrate

from ramvi_service.i18n import UserError

PREFIX = "ramvi-"
# remote (7.4) — ключ можно использовать из сети; без него ключ работает только локально.
PERMISSIONS = frozenset({"chat", "models", "agent", "remote"})


class ClientKeyError(UserError, ValueError):
    """Неверный запрос ключа: ключ сообщения `client_keys.*` (3.14)."""


@dataclass(frozen=True)
class ClientKey:
    id: str
    name: str
    permissions: frozenset[str]
    created: float
    last_seen: float | None
    requests: int
    revoked: float | None
    key_hint: str = ""  # «ramvi-…a1b2»: узнать ключ в списке; целиком не показывается
    prompt_tokens: int = 0
    completion_tokens: int = 0
    busy_seconds: float = 0.0  # сколько модель работала на этого клиента
    errors: int = 0

    @property
    def state(self) -> str:
        return "revoked" if self.revoked is not None else "active"

    def json(self) -> dict[str, Any]:
        """Для экрана «Подключения»: без ключа и хеша."""
        return {
            "id": self.id,
            "name": self.name,
            "permissions": sorted(self.permissions),
            "created": self.created,
            "last_seen": self.last_seen,
            "state": self.state,
            "revoked": self.revoked,
            "key_hint": self.key_hint,
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "busy_seconds": round(self.busy_seconds, 3),
            "errors": self.errors,
        }


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


_COLUMNS = (
    "id, client_name, permissions, created_at, last_seen, requests, revoked_at,"
    " key_hint, prompt_tokens, completion_tokens, busy_seconds, errors"
)


class KeyStore:
    """Ключи в общей базе. Путь — открыть отдельную базу (схема ядра доводится до текущей)."""

    def __init__(self, store: str | Path | Database):
        self._lock = threading.Lock()
        self._owned = not isinstance(store, Database)
        self._db = store if isinstance(store, Database) else Database(store)
        if self._owned:
            migrate(self._db, CORE_OWNER, CORE_MIGRATIONS)

    def close(self) -> None:
        if self._owned:
            self._db.close()

    def create(self, name: str, permissions: set[str] | frozenset[str]) -> tuple[ClientKey, str]:
        """Вернуть запись и ключ. Ключ больше нигде не сохраняется и не показывается."""
        name = name.strip()
        if not name:
            raise ClientKeyError("client_keys.no_name")
        unknown = set(permissions) - PERMISSIONS
        if unknown or not permissions:
            raise ClientKeyError(
                "client_keys.bad_permissions",
                allowed=", ".join(sorted(PERMISSIONS)),
                got=", ".join(sorted(permissions)),
            )
        key = PREFIX + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(6)
        with self._lock, self._db.transaction():
            self._db.execute(
                "INSERT INTO client_keys"
                " (id, client_name, secret_hash, permissions, created_at, key_hint)"
                " VALUES (?,?,?,?,?,?)",
                (
                    key_id,
                    name,
                    _hash(key),
                    ",".join(sorted(permissions)),
                    time.time(),
                    f"{PREFIX}…{key[-4:]}",
                ),
            )
        return self.get(key_id), key

    def get(self, key_id: str) -> ClientKey:
        row = self._db.one(f"SELECT {_COLUMNS} FROM client_keys WHERE id=?", (key_id,))
        if row is None:
            raise KeyError(key_id)
        return ClientKey(
            row[0],
            row[1],
            frozenset(row[2].split(",")),
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
        )

    def add_usage(
        self, key_id: str, prompt: int, completion: int, seconds: float, *, error: bool = False
    ) -> None:
        """Учесть запрос клиента: токены, время работы модели, ошибку (API-02)."""
        with self._lock, self._db.transaction():
            self._db.execute(
                "UPDATE client_keys SET prompt_tokens=prompt_tokens+?,"
                " completion_tokens=completion_tokens+?, busy_seconds=busy_seconds+?,"
                " errors=errors+? WHERE id=?",
                (max(0, prompt), max(0, completion), max(0.0, seconds), int(error), key_id),
            )

    def list(self) -> list[ClientKey]:
        ids = [r[0] for r in self._db.all("SELECT id FROM client_keys ORDER BY created_at, id")]
        return [self.get(i) for i in ids]

    def revoke(self, key_id: str) -> None:
        with self._lock, self._db.transaction():
            cur = self._db.execute(
                "UPDATE client_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
        if cur.rowcount == 0:
            self.get(key_id)  # KeyError, если такого нет; повторный отзыв — без ошибки

    def verify(self, key: str | None, permission: str | None = None) -> ClientKey | None:
        """Ключ действителен и (если задано) имеет право: обновить last_seen и счётчик."""
        if not key or not key.startswith(PREFIX):
            return None
        digest = _hash(key)
        row = self._db.one(
            "SELECT id, secret_hash, permissions, revoked_at FROM client_keys WHERE secret_hash=?",
            (digest,),
        )
        if row is None or not hmac.compare_digest(row[1], digest) or row[3] is not None:
            return None
        if permission is not None and permission not in row[2].split(","):
            return None
        with self._lock, self._db.transaction():
            self._db.execute(
                "UPDATE client_keys SET last_seen=?, requests=requests+1 WHERE id=?",
                (time.time(), row[0]),
            )
        return self.get(row[0])


_CHAT_ROUTES = (
    "/api/chat",
    "/api/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
)


def permission_for(path: str) -> str:
    """Генерация требует chat; список моделей, версии и состояние движка — models."""
    route = path.split("?", 1)[0].rstrip("/")
    return "chat" if route in _CHAT_ROUTES else "models"


def checker(store: KeyStore) -> Callable[[str | None, str], bool]:
    """Функция для App(api_key=...): ключ действителен и имеет право на маршрут."""
    return lambda token, path: store.verify(token, permission_for(path)) is not None
