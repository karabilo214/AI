-- CHAR-14: память и состояние персонажа в собственной базе модуля (не в Core).
-- Каждая запись принадлежит пространству «пользователь + персонаж»; модуль обращается
-- к ней только с явным namespace_id, поэтому записи разных пользователей не смешиваются.

CREATE TABLE memory_namespace (
    id           TEXT PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 160),
    user_id      TEXT NOT NULL CHECK (length(user_id) BETWEEN 1 AND 64),
    character_id TEXT NOT NULL CHECK (length(character_id) BETWEEN 1 AND 64),
    created_at   TEXT NOT NULL,
    UNIQUE (user_id, character_id)
);

-- Виды памяти ТЗ 10.2. origin различает сказанное пользователем, подтверждённое наблюдение,
-- интерпретацию модели и предположение (ТЗ 10.5): предположение не становится фактом само.
CREATE TABLE memory_record (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    id           TEXT NOT NULL CHECK (length(id) BETWEEN 1 AND 64),
    kind         TEXT NOT NULL CHECK (kind IN ('working', 'profile', 'episodic', 'semantic', 'behavioral')),
    type         TEXT NOT NULL CHECK (length(type) BETWEEN 1 AND 64),
    subject      TEXT NOT NULL CHECK (length(subject) BETWEEN 1 AND 256),
    value        TEXT NOT NULL CHECK (length(value) BETWEEN 1 AND 4096),
    confidence   REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    occurrences  INTEGER NOT NULL CHECK (occurrences >= 1),
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL CHECK (last_seen >= first_seen),
    status       TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'disputed', 'superseded')),
    origin       TEXT NOT NULL CHECK (origin IN ('user_said', 'observed', 'interpretation', 'assumption')),
    source_ids   TEXT NOT NULL CHECK (json_valid(source_ids) AND json_type(source_ids) = 'array'),
    expires_at   TEXT CHECK (expires_at IS NULL OR kind = 'working'),
    CHECK (status <> 'active' OR origin IN ('user_said', 'observed')),
    PRIMARY KEY (namespace_id, id)
);

CREATE INDEX memory_record_by_kind ON memory_record (namespace_id, kind, status, last_seen);
CREATE INDEX memory_record_by_subject ON memory_record (namespace_id, subject);

-- Character State: текущее состояние, не тождественное воспоминаниям (ТЗ 10.2).
CREATE TABLE character_state (
    namespace_id TEXT PRIMARY KEY REFERENCES memory_namespace (id) ON DELETE CASCADE,
    schema       TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (json_valid(state) AND json_type(state) = 'object'),
    revision     INTEGER NOT NULL CHECK (revision >= 1),
    updated_at   TEXT NOT NULL
);
