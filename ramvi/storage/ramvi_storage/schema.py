"""Схема ядра RAMVI (раздел 10 ТЗ): модели, профили, чаты, агент, ключи, загрузки.

Время — секунды Unix (REAL, UTC). JSON-поля — TEXT с проверкой `json_valid`.
Файлы моделей лежат вне базы (`models.path`); каталог моделей — JSON-кэш вне базы.
Таблицы модулей сюда не входят: модуль регистрирует свою цепочку миграций (3.12).
"""

from pathlib import Path

from ramvi_storage.database import Database
from ramvi_storage.migrations import Migration, migrate

CORE_OWNER = "core"
FILE_NAME = "ramvi.sqlite3"

DOWNLOAD_STATES = ("queued", "running", "paused", "verifying", "done", "failed", "cancelled")
AGENT_STATUSES = ("planned", "running", "waiting_approval", "done", "failed", "cancelled")
MESSAGE_ROLES = ("system", "user", "assistant", "tool")
TOOL_PERMISSIONS = ("auto", "approved", "denied")
MESSAGE_FINISH = ("stop", "length", "cancelled", "error")


def _in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


_V1 = f"""
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    architecture TEXT NOT NULL,
    quant TEXT,
    sha256 TEXT CHECK (sha256 IS NULL OR length(sha256) = 64),
    bytes INTEGER CHECK (bytes IS NULL OR bytes >= 0),
    path TEXT NOT NULL,
    license TEXT,
    installed_at REAL NOT NULL,
    capabilities TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(capabilities))
) STRICT;

CREATE TABLE model_profiles (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    hardware_hash_local TEXT NOT NULL,
    context INTEGER NOT NULL CHECK (context > 0),
    plan TEXT NOT NULL CHECK (json_valid(plan)),
    ttft REAL CHECK (ttft IS NULL OR ttft >= 0),
    tokps REAL CHECK (tokps IS NULL OR tokps >= 0),
    peak_ram INTEGER CHECK (peak_ram IS NULL OR peak_ram >= 0),
    peak_vram INTEGER CHECK (peak_vram IS NULL OR peak_vram >= 0),
    tested_at REAL NOT NULL
) STRICT;
CREATE INDEX model_profiles_lookup ON model_profiles (model_id, hardware_hash_local, tested_at);

CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model_id TEXT REFERENCES models(id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
) STRICT;
CREATE INDEX chats_recent ON chats (updated_at);

CREATE TABLE messages (
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ({_in(MESSAGE_ROLES)})),
    content TEXT NOT NULL,
    token_usage TEXT CHECK (token_usage IS NULL OR json_valid(token_usage)),
    created_at REAL NOT NULL
) STRICT;
CREATE INDEX messages_by_chat ON messages (chat_id, seq);

CREATE TABLE agent_runs (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    workspace TEXT NOT NULL,
    model_id TEXT REFERENCES models(id) ON DELETE SET NULL,
    status TEXT NOT NULL CHECK (status IN ({_in(AGENT_STATUSES)})),
    steps INTEGER NOT NULL DEFAULT 0 CHECK (steps >= 0),
    token_usage TEXT CHECK (token_usage IS NULL OR json_valid(token_usage)),
    started_at REAL NOT NULL,
    ended_at REAL
) STRICT;

CREATE TABLE tool_calls (
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    agent_run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    args_redacted TEXT NOT NULL CHECK (json_valid(args_redacted)),
    permission TEXT NOT NULL CHECK (permission IN ({_in(TOOL_PERMISSIONS)})),
    result TEXT,
    duration REAL CHECK (duration IS NULL OR duration >= 0)
) STRICT;
CREATE INDEX tool_calls_by_run ON tool_calls (agent_run_id, seq);

CREATE TABLE client_keys (
    id TEXT PRIMARY KEY,
    client_name TEXT NOT NULL,
    permissions TEXT NOT NULL,
    secret_hash TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    last_seen REAL,
    requests INTEGER NOT NULL DEFAULT 0,
    revoked_at REAL
) STRICT;

CREATE TABLE downloads (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    source TEXT NOT NULL,
    temp_path TEXT NOT NULL,
    bytes_done INTEGER NOT NULL DEFAULT 0 CHECK (bytes_done >= 0),
    total_bytes INTEGER CHECK (total_bytes IS NULL OR total_bytes >= 0),
    state TEXT NOT NULL CHECK (state IN ({_in(DOWNLOAD_STATES)})),
    checksum TEXT NOT NULL CHECK (length(checksum) = 64)
) STRICT;
CREATE INDEX downloads_by_state ON downloads (state);
"""

# v2 (3.4, чат): thinking отдельно от ответа, причина завершения (остановлен
# пользователем — cancelled) и вложенные текстовые файлы сообщения.
_V2 = f"""
ALTER TABLE messages ADD COLUMN reasoning TEXT NOT NULL DEFAULT '';
ALTER TABLE messages ADD COLUMN finish TEXT
    CHECK (finish IS NULL OR finish IN ({_in(MESSAGE_FINISH)}));
ALTER TABLE messages ADD COLUMN attachments TEXT
    CHECK (attachments IS NULL OR json_valid(attachments));
"""

CORE_MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "core entities (TZ section 10)", _V1),
    Migration(2, "chat: reasoning, finish, attachments", _V2),
)


def open_core(data_dir: str | Path, *, file_name: str = FILE_NAME) -> Database:
    """Открыть базу приложения в папке данных и довести схему ядра до текущей."""
    folder = Path(data_dir)
    db = Database(folder / file_name)
    try:
        migrate(db, CORE_OWNER, CORE_MIGRATIONS, backup_dir=folder / "backups")
    except BaseException:
        db.close()
        raise
    return db
