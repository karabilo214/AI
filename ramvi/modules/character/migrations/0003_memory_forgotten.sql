-- CHAR-17: след удалённой записи, чтобы модель не восстановила её сама из тех же источников.
-- Текста здесь нет: только солёные (id пространства) хэши темы и основ слов и id источников
-- (ответы и реплики истории), поэтому по следу нельзя прочитать, что было забыто.

CREATE TABLE memory_forgotten (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    id           TEXT NOT NULL CHECK (length(id) BETWEEN 1 AND 64),
    kind         TEXT NOT NULL,
    subject_hash TEXT NOT NULL,
    stem_hashes  TEXT NOT NULL CHECK (json_valid(stem_hashes) AND json_type(stem_hashes) = 'array'),
    source_ids   TEXT NOT NULL CHECK (json_valid(source_ids) AND json_type(source_ids) = 'array'),
    forgotten_at TEXT NOT NULL,
    PRIMARY KEY (namespace_id, id)
);

CREATE INDEX memory_forgotten_by_subject ON memory_forgotten (namespace_id, subject_hash);
