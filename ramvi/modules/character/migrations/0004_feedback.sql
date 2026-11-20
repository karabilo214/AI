-- CHAR-18: явная обратная связь и предпочтения с уверенностью и происхождением сигнала.
-- Сигнал относится к ответу персонажа (response_id) и к одной стороне манеры (dimension).
-- reliable решает код: кнопка или дословные однозначные слова владельца; догадка модели,
-- неявная или неоднозначная реакция хранятся, но достоверной оценкой не считаются.

CREATE TABLE feedback_signal (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    id           TEXT NOT NULL CHECK (length(id) BETWEEN 1 AND 64),
    response_id  TEXT NOT NULL CHECK (length(response_id) BETWEEN 1 AND 64),
    dimension    TEXT NOT NULL CHECK (dimension IN ('overall', 'warmth', 'humor', 'sarcasm',
                     'strictness', 'curiosity', 'verbosity', 'initiative', 'empathy')),
    direction    INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    confidence   REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    origin       TEXT NOT NULL CHECK (origin IN ('explicit_ui', 'user_said', 'interpretation', 'implicit')),
    ambiguous    INTEGER NOT NULL CHECK (ambiguous IN (0, 1)),
    reliable     INTEGER NOT NULL CHECK (reliable IN (0, 1)),
    quote        TEXT CHECK (quote IS NULL OR length(quote) BETWEEN 1 AND 256),
    created_at   TEXT NOT NULL,
    CHECK (reliable = 0 OR (ambiguous = 0 AND origin IN ('explicit_ui', 'user_said'))),
    PRIMARY KEY (namespace_id, id)
);

CREATE INDEX feedback_by_dimension ON feedback_signal (namespace_id, dimension, created_at);
CREATE INDEX feedback_by_response ON feedback_signal (namespace_id, response_id);
