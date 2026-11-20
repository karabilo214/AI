-- CHAR-15: сигналы комбинированного поиска и индексы, чтобы не читать всю историю.
-- Темы и сущности приходят из кода модуля уже нормализованными (casefold), потому что
-- lower() в SQLite не понимает кириллицу.

-- NOT NULL вместе с CHECK в ADD COLUMN SQLite отклоняет на непустой таблице,
-- поэтому проверка на пустое значение стоит внутри CHECK.
ALTER TABLE memory_record ADD COLUMN importance REAL DEFAULT 0.5
    CHECK (importance IS NOT NULL AND importance >= 0 AND importance <= 1);
ALTER TABLE memory_record ADD COLUMN unfinished INTEGER DEFAULT 0
    CHECK (unfinished IS NOT NULL AND unfinished IN (0, 1));
ALTER TABLE memory_record ADD COLUMN topics TEXT DEFAULT '[]'
    CHECK (topics IS NOT NULL AND json_valid(topics) AND json_type(topics) = 'array');
ALTER TABLE memory_record ADD COLUMN entities TEXT DEFAULT '[]'
    CHECK (entities IS NOT NULL AND json_valid(entities) AND json_type(entities) = 'array');

CREATE INDEX memory_record_unfinished ON memory_record (namespace_id, last_seen)
    WHERE unfinished = 1;
CREATE INDEX memory_record_salient ON memory_record (namespace_id, importance, last_seen);

-- Точный индекс тем и сущностей: поиск по ним идёт по B-дереву, а не перебором записей.
CREATE TABLE memory_tag (
    namespace_id TEXT NOT NULL,
    memory_id    TEXT NOT NULL,
    tag_kind     TEXT NOT NULL CHECK (tag_kind IN ('topic', 'entity')),
    tag          TEXT NOT NULL,
    PRIMARY KEY (namespace_id, tag, tag_kind, memory_id),
    FOREIGN KEY (namespace_id, memory_id) REFERENCES memory_record (namespace_id, id)
        ON DELETE CASCADE
);
CREATE INDEX memory_tag_by_record ON memory_tag (namespace_id, memory_id);

-- Полнотекстовый индекс по субъекту, значению, темам и сущностям.
CREATE VIRTUAL TABLE memory_fts USING fts5 (
    namespace_id UNINDEXED,
    memory_id UNINDEXED,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER memory_record_indexed AFTER INSERT ON memory_record BEGIN
    INSERT INTO memory_fts (namespace_id, memory_id, body) VALUES (
        new.namespace_id, new.id,
        new.subject || ' ' || new.value || ' ' ||
        (SELECT coalesce(group_concat(value, ' '), '') FROM json_each(new.topics)) || ' ' ||
        (SELECT coalesce(group_concat(value, ' '), '') FROM json_each(new.entities))
    );
    INSERT OR IGNORE INTO memory_tag (namespace_id, memory_id, tag_kind, tag)
        SELECT new.namespace_id, new.id, 'topic', value FROM json_each(new.topics);
    INSERT OR IGNORE INTO memory_tag (namespace_id, memory_id, tag_kind, tag)
        SELECT new.namespace_id, new.id, 'entity', value FROM json_each(new.entities);
END;

CREATE TRIGGER memory_record_unindexed AFTER DELETE ON memory_record BEGIN
    DELETE FROM memory_fts WHERE namespace_id = old.namespace_id AND memory_id = old.id;
    DELETE FROM memory_tag WHERE namespace_id = old.namespace_id AND memory_id = old.id;
END;

CREATE TRIGGER memory_record_reindexed
AFTER UPDATE OF subject, value, topics, entities ON memory_record BEGIN
    DELETE FROM memory_fts WHERE namespace_id = old.namespace_id AND memory_id = old.id;
    DELETE FROM memory_tag WHERE namespace_id = old.namespace_id AND memory_id = old.id;
    INSERT INTO memory_fts (namespace_id, memory_id, body) VALUES (
        new.namespace_id, new.id,
        new.subject || ' ' || new.value || ' ' ||
        (SELECT coalesce(group_concat(value, ' '), '') FROM json_each(new.topics)) || ' ' ||
        (SELECT coalesce(group_concat(value, ' '), '') FROM json_each(new.entities))
    );
    INSERT OR IGNORE INTO memory_tag (namespace_id, memory_id, tag_kind, tag)
        SELECT new.namespace_id, new.id, 'topic', value FROM json_each(new.topics);
    INSERT OR IGNORE INTO memory_tag (namespace_id, memory_id, tag_kind, tag)
        SELECT new.namespace_id, new.id, 'entity', value FROM json_each(new.entities);
END;

-- Записи, сделанные до этой миграции, тоже попадают в индекс.
INSERT INTO memory_fts (namespace_id, memory_id, body)
    SELECT namespace_id, id, subject || ' ' || value FROM memory_record;
