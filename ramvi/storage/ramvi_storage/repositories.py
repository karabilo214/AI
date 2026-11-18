"""Репозитории сущностей ядра. Только чтение/запись строк, без бизнес-логики.

Идентификаторы — случайные hex-строки, если вызывающий не передал свой.
Время по умолчанию — `time.time()`; для проверок его можно передать явно.
"""

import json
import secrets
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ramvi_storage.database import Database, StorageError
from ramvi_storage.schema import (
    AGENT_STATUSES,
    DOWNLOAD_STATES,
    MESSAGE_FINISH,
    MESSAGE_ROLES,
    TOOL_PERMISSIONS,
)


class NotFound(StorageError, KeyError):
    pass


def new_id() -> str:
    return secrets.token_hex(8)


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, allow_nan=False)


def _load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _now(value: float | None) -> float:
    return time.time() if value is None else value


def _choice(value: str, allowed: Iterable[str], what: str) -> str:
    allowed = tuple(allowed)
    if value not in allowed:
        raise ValueError(f"{what}: {value!r} is not one of {allowed}")
    return value


# --- Модели и профили -----------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    architecture: str
    path: str
    installed_at: float
    quant: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    license: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    hardware_hash_local: str
    context: int
    plan: dict[str, Any]
    tested_at: float
    ttft: float | None = None
    tokps: float | None = None
    peak_ram: int | None = None
    peak_vram: int | None = None


def _model(row: sqlite3.Row) -> Model:
    return Model(
        id=row["id"],
        provider=row["provider"],
        architecture=row["architecture"],
        path=row["path"],
        installed_at=row["installed_at"],
        quant=row["quant"],
        sha256=row["sha256"],
        bytes=row["bytes"],
        license=row["license"],
        capabilities=_load(row["capabilities"]),
    )


def _profile(row: sqlite3.Row) -> ModelProfile:
    return ModelProfile(
        model_id=row["model_id"],
        hardware_hash_local=row["hardware_hash_local"],
        context=row["context"],
        plan=_load(row["plan"]),
        tested_at=row["tested_at"],
        ttft=row["ttft"],
        tokps=row["tokps"],
        peak_ram=row["peak_ram"],
        peak_vram=row["peak_vram"],
    )


class Models:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, model: Model) -> Model:
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO models (id, provider, architecture, quant, sha256, bytes, path,"
                " license, installed_at, capabilities) VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET provider=excluded.provider,"
                " architecture=excluded.architecture, quant=excluded.quant,"
                " sha256=excluded.sha256, bytes=excluded.bytes, path=excluded.path,"
                " license=excluded.license, installed_at=excluded.installed_at,"
                " capabilities=excluded.capabilities",
                (
                    model.id,
                    model.provider,
                    model.architecture,
                    model.quant,
                    model.sha256,
                    model.bytes,
                    model.path,
                    model.license,
                    model.installed_at,
                    _json(model.capabilities),
                ),
            )
        return self.get(model.id)

    def get(self, model_id: str) -> Model:
        row = self.db.one("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            raise NotFound(model_id)
        return _model(row)

    def all(self) -> list[Model]:
        return [_model(r) for r in self.db.all("SELECT * FROM models ORDER BY installed_at, id")]

    def delete(self, model_id: str) -> None:
        """Удалить запись (профили — каскадом, у чатов model_id станет NULL). Файл не трогается."""
        with self.db.transaction():
            if self.db.execute("DELETE FROM models WHERE id = ?", (model_id,)).rowcount == 0:
                raise NotFound(model_id)

    def add_profile(self, profile: ModelProfile) -> None:
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO model_profiles (model_id, hardware_hash_local, context, plan, ttft,"
                " tokps, peak_ram, peak_vram, tested_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    profile.model_id,
                    profile.hardware_hash_local,
                    profile.context,
                    _json(profile.plan),
                    profile.ttft,
                    profile.tokps,
                    profile.peak_ram,
                    profile.peak_vram,
                    profile.tested_at,
                ),
            )

    def latest_profile(self, model_id: str, hardware_hash_local: str) -> ModelProfile | None:
        row = self.db.one(
            "SELECT * FROM model_profiles WHERE model_id = ? AND hardware_hash_local = ?"
            " ORDER BY tested_at DESC, id DESC LIMIT 1",
            (model_id, hardware_hash_local),
        )
        return None if row is None else _profile(row)


# --- Чаты и сообщения -----------------------------------------------------------------


@dataclass(frozen=True)
class Chat:
    id: str
    title: str
    model_id: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Message:
    id: str
    chat_id: str
    role: str
    content: str
    created_at: float
    token_usage: dict[str, Any] | None = None
    reasoning: str = ""
    finish: str | None = None
    attachments: list[dict[str, Any]] | None = None


def _chat(row: sqlite3.Row) -> Chat:
    return Chat(row["id"], row["title"], row["model_id"], row["created_at"], row["updated_at"])


def _message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        chat_id=row["chat_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        token_usage=_load(row["token_usage"]),
        reasoning=row["reasoning"],
        finish=row["finish"],
        attachments=_load(row["attachments"]),
    )


class Chats:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        title: str,
        model_id: str | None = None,
        *,
        chat_id: str | None = None,
        now: float | None = None,
    ) -> Chat:
        stamp = _now(now)
        cid = chat_id or new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO chats (id, title, model_id, created_at, updated_at) VALUES (?,?,?,?,?)",
                (cid, title, model_id, stamp, stamp),
            )
        return self.get(cid)

    def get(self, chat_id: str) -> Chat:
        row = self.db.one("SELECT * FROM chats WHERE id = ?", (chat_id,))
        if row is None:
            raise NotFound(chat_id)
        return _chat(row)

    def recent(self, limit: int = 200, offset: int = 0) -> list[Chat]:
        """Сначала недавно изменённые."""
        rows = self.db.all(
            "SELECT * FROM chats ORDER BY updated_at DESC, id LIMIT ? OFFSET ?", (limit, offset)
        )
        return [_chat(r) for r in rows]

    def update(
        self,
        chat_id: str,
        *,
        title: str | None = None,
        model_id: str | None = None,
        now: float | None = None,
    ) -> Chat:
        current = self.get(chat_id)
        with self.db.transaction():
            self.db.execute(
                "UPDATE chats SET title = ?, model_id = ?, updated_at = ? WHERE id = ?",
                (
                    current.title if title is None else title,
                    current.model_id if model_id is None else model_id,
                    _now(now),
                    chat_id,
                ),
            )
        return self.get(chat_id)

    def delete(self, chat_id: str) -> None:
        with self.db.transaction():
            if self.db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount == 0:
                raise NotFound(chat_id)

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        token_usage: dict[str, Any] | None = None,
        reasoning: str = "",
        finish: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        now: float | None = None,
    ) -> Message:
        _choice(role, MESSAGE_ROLES, "role")
        if finish is not None:
            _choice(finish, MESSAGE_FINISH, "finish reason")
        stamp = _now(now)
        mid = message_id or new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO messages (id, chat_id, role, content, token_usage, created_at,"
                " reasoning, finish, attachments) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    chat_id,
                    role,
                    content,
                    _json(token_usage),
                    stamp,
                    reasoning,
                    finish,
                    _json(attachments),
                ),
            )
            self.db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (stamp, chat_id))
        row = self.db.one("SELECT * FROM messages WHERE id = ?", (mid,))
        assert row is not None
        return _message(row)

    def messages(self, chat_id: str) -> list[Message]:
        """В порядке добавления (не по времени: часы могут идти назад)."""
        self.get(chat_id)
        rows = self.db.all("SELECT * FROM messages WHERE chat_id = ? ORDER BY seq", (chat_id,))
        return [_message(r) for r in rows]

    def delete_messages_from(self, chat_id: str, message_id: str) -> int:
        """Удалить сообщение и все после него (Regenerate). Вернуть число удалённых."""
        with self.db.transaction():
            row = self.db.one(
                "SELECT seq FROM messages WHERE id = ? AND chat_id = ?", (message_id, chat_id)
            )
            if row is None:
                raise NotFound(message_id)
            cursor = self.db.execute(
                "DELETE FROM messages WHERE chat_id = ? AND seq >= ?", (chat_id, row["seq"])
            )
            return cursor.rowcount

    def clear(self, chat_id: str) -> int:
        self.get(chat_id)
        with self.db.transaction():
            return self.db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,)).rowcount


# --- Загрузки -------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadRecord:
    id: str
    model_id: str
    source: str
    temp_path: str
    checksum: str
    state: str
    bytes_done: int = 0
    total_bytes: int | None = None


def _download(row: sqlite3.Row) -> DownloadRecord:
    return DownloadRecord(
        id=row["id"],
        model_id=row["model_id"],
        source=row["source"],
        temp_path=row["temp_path"],
        checksum=row["checksum"],
        state=row["state"],
        bytes_done=row["bytes_done"],
        total_bytes=row["total_bytes"],
    )


class Downloads:
    """`model_id` — идентификатор каталога; записи в `models` до окончания загрузки нет."""

    UNFINISHED = ("queued", "running", "paused", "verifying")

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        model_id: str,
        source: str,
        temp_path: str,
        checksum: str,
        total_bytes: int | None = None,
        *,
        download_id: str | None = None,
    ) -> DownloadRecord:
        did = download_id or new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO downloads (id, model_id, source, temp_path, total_bytes, state,"
                " checksum) VALUES (?,?,?,?,?,'queued',?)",
                (did, model_id, source, temp_path, total_bytes, checksum.lower()),
            )
        return self.get(did)

    def get(self, download_id: str) -> DownloadRecord:
        row = self.db.one("SELECT * FROM downloads WHERE id = ?", (download_id,))
        if row is None:
            raise NotFound(download_id)
        return _download(row)

    def update(
        self,
        download_id: str,
        *,
        state: str | None = None,
        bytes_done: int | None = None,
        total_bytes: int | None = None,
    ) -> DownloadRecord:
        current = self.get(download_id)
        if state is not None:
            _choice(state, DOWNLOAD_STATES, "download state")
        with self.db.transaction():
            self.db.execute(
                "UPDATE downloads SET state = ?, bytes_done = ?, total_bytes = ? WHERE id = ?",
                (
                    current.state if state is None else state,
                    current.bytes_done if bytes_done is None else bytes_done,
                    current.total_bytes if total_bytes is None else total_bytes,
                    download_id,
                ),
            )
        return self.get(download_id)

    def unfinished(self) -> list[DownloadRecord]:
        """Что продолжить после перезапуска приложения."""
        marks = ",".join("?" * len(self.UNFINISHED))
        rows = self.db.all(
            f"SELECT * FROM downloads WHERE state IN ({marks}) ORDER BY rowid", self.UNFINISHED
        )
        return [_download(r) for r in rows]

    def delete(self, download_id: str) -> None:
        with self.db.transaction():
            if self.db.execute("DELETE FROM downloads WHERE id = ?", (download_id,)).rowcount == 0:
                raise NotFound(download_id)


# --- Агент ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRun:
    id: str
    objective: str
    workspace: str
    model_id: str | None
    status: str
    steps: int
    started_at: float
    ended_at: float | None = None
    token_usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    agent_run_id: str
    tool_name: str
    args_redacted: dict[str, Any]
    permission: str
    result: str | None = None
    duration: float | None = None


def _run(row: sqlite3.Row) -> AgentRun:
    return AgentRun(
        id=row["id"],
        objective=row["objective"],
        workspace=row["workspace"],
        model_id=row["model_id"],
        status=row["status"],
        steps=row["steps"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        token_usage=_load(row["token_usage"]),
    )


def _call(row: sqlite3.Row) -> ToolCall:
    return ToolCall(
        id=row["id"],
        agent_run_id=row["agent_run_id"],
        tool_name=row["tool_name"],
        args_redacted=_load(row["args_redacted"]),
        permission=row["permission"],
        result=row["result"],
        duration=row["duration"],
    )


class AgentRuns:
    FINAL = ("done", "failed", "cancelled")

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        objective: str,
        workspace: str,
        model_id: str | None = None,
        *,
        run_id: str | None = None,
        now: float | None = None,
    ) -> AgentRun:
        rid = run_id or new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO agent_runs (id, objective, workspace, model_id, status, started_at)"
                " VALUES (?,?,?,?,'planned',?)",
                (rid, objective, workspace, model_id, _now(now)),
            )
        return self.get(rid)

    def get(self, run_id: str) -> AgentRun:
        row = self.db.one("SELECT * FROM agent_runs WHERE id = ?", (run_id,))
        if row is None:
            raise NotFound(run_id)
        return _run(row)

    def recent(self, limit: int = 100) -> list[AgentRun]:
        rows = self.db.all(
            "SELECT * FROM agent_runs ORDER BY started_at DESC, id LIMIT ?", (limit,)
        )
        return [_run(r) for r in rows]

    def update(
        self,
        run_id: str,
        *,
        status: str | None = None,
        steps: int | None = None,
        token_usage: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> AgentRun:
        """Конечный статус (done/failed/cancelled) ставит ended_at; после него статус не меняется."""
        current = self.get(run_id)
        if status is not None:
            _choice(status, AGENT_STATUSES, "agent status")
            if current.status in self.FINAL and status != current.status:
                raise ValueError(f"run {run_id} is already finished ({current.status})")
        new_status = current.status if status is None else status
        ended = current.ended_at
        if new_status in self.FINAL and ended is None:
            ended = _now(now)
        with self.db.transaction():
            self.db.execute(
                "UPDATE agent_runs SET status = ?, steps = ?, token_usage = ?, ended_at = ?"
                " WHERE id = ?",
                (
                    new_status,
                    current.steps if steps is None else steps,
                    _json(current.token_usage if token_usage is None else token_usage),
                    ended,
                    run_id,
                ),
            )
        return self.get(run_id)

    def add_tool_call(
        self,
        run_id: str,
        tool_name: str,
        args_redacted: dict[str, Any],
        permission: str,
        *,
        result: str | None = None,
        duration: float | None = None,
        call_id: str | None = None,
    ) -> ToolCall:
        """Аргументы должны быть уже очищены от секретов — это делает Agent Runtime."""
        _choice(permission, TOOL_PERMISSIONS, "tool permission")
        cid = call_id or new_id()
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO tool_calls (id, agent_run_id, tool_name, args_redacted, permission,"
                " result, duration) VALUES (?,?,?,?,?,?,?)",
                (cid, run_id, tool_name, _json(args_redacted), permission, result, duration),
            )
        row = self.db.one("SELECT * FROM tool_calls WHERE id = ?", (cid,))
        assert row is not None
        return _call(row)

    def finish_tool_call(
        self,
        call_id: str,
        result: str,
        duration: float,
        *,
        permission: str | None = None,
    ) -> ToolCall:
        """Дописать результат вызова, записанного до выполнения (журнал AG-02)."""
        if permission is not None:
            _choice(permission, TOOL_PERMISSIONS, "tool permission")
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE tool_calls SET result = ?, duration = ?,"
                " permission = COALESCE(?, permission) WHERE id = ?",
                (result, max(0.0, duration), permission, call_id),
            )
            if cursor.rowcount == 0:
                raise NotFound(call_id)
        row = self.db.one("SELECT * FROM tool_calls WHERE id = ?", (call_id,))
        assert row is not None
        return _call(row)

    def tool_calls(self, run_id: str) -> list[ToolCall]:
        self.get(run_id)
        rows = self.db.all(
            "SELECT * FROM tool_calls WHERE agent_run_id = ? ORDER BY seq", (run_id,)
        )
        return [_call(r) for r in rows]
