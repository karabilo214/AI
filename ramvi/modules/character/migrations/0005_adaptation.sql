-- CHAR-19: персональные поправки черт поверх базового профиля, без изменения весов LLM.
-- Поправка своя у каждого пространства (пользователь + персонаж) и профиля; диапазон
-- ограничен и в схеме. Каждое изменение пишется в журнал (для истории и отката в 9.20).

CREATE TABLE adaptation_state (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    profile      TEXT NOT NULL CHECK (length(profile) BETWEEN 1 AND 96),
    trait        TEXT NOT NULL CHECK (trait IN ('warmth', 'humor', 'sarcasm', 'strictness',
                     'curiosity', 'verbosity', 'initiative', 'empathy')),
    value        REAL NOT NULL CHECK (value >= -0.2 AND value <= 0.2),
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (namespace_id, profile, trait)
);

CREATE TABLE adaptation_change (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    id           TEXT NOT NULL CHECK (length(id) BETWEEN 1 AND 64),
    profile      TEXT NOT NULL,
    trait        TEXT NOT NULL,
    before       REAL NOT NULL CHECK (before >= -0.2 AND before <= 0.2),
    after        REAL NOT NULL CHECK (after >= -0.2 AND after <= 0.2),
    reason       TEXT NOT NULL CHECK (json_valid(reason) AND json_type(reason) = 'object'),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (namespace_id, id)
);

CREATE INDEX adaptation_change_by_trait
    ON adaptation_change (namespace_id, profile, trait, created_at);
