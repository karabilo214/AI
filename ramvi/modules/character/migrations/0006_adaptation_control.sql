-- CHAR-20: включение/выключение подстройки, точки сохранения, сброс и откат.
-- Выключение хранится отдельно и не трогает ни поправки, ни память, ни профиль.
-- Перед каждым изменением поправок (шаг, сброс, откат) пишется снимок полного состояния
-- профиля — откат к нему воспроизводит прежние значения точно.

CREATE TABLE adaptation_setting (
    namespace_id TEXT PRIMARY KEY REFERENCES memory_namespace (id) ON DELETE CASCADE,
    enabled      INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    changed_at   TEXT NOT NULL
);

CREATE TABLE adaptation_snapshot (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    id           TEXT NOT NULL CHECK (length(id) BETWEEN 1 AND 64),
    profile      TEXT NOT NULL CHECK (length(profile) BETWEEN 1 AND 96),
    kind         TEXT NOT NULL CHECK (kind IN ('saved', 'before_step', 'before_reset',
                     'before_restore')),
    traits       TEXT NOT NULL CHECK (json_valid(traits) AND json_type(traits) = 'object'),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (namespace_id, id)
);

CREATE INDEX adaptation_snapshot_by_profile
    ON adaptation_snapshot (namespace_id, profile, created_at);

ALTER TABLE adaptation_change ADD COLUMN action TEXT DEFAULT 'step'
    CHECK (action IS NOT NULL AND action IN ('step', 'reset', 'restore'));
ALTER TABLE adaptation_change ADD COLUMN snapshot_id TEXT;
