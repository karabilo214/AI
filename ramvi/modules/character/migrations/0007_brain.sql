-- CHAR-BRAIN-02 (9.31): какую реакцию выбрал контроллер для ответа персонажа. Вместе с
-- достоверными оценками владельца (feedback_signal по response_id) — память исходов стратегий.
-- Сама реплика и рассуждения модели здесь не хранятся: только событие, стратегия и контроллер.

CREATE TABLE brain_decision (
    namespace_id TEXT NOT NULL REFERENCES memory_namespace (id) ON DELETE CASCADE,
    response_id  TEXT NOT NULL CHECK (length(response_id) BETWEEN 1 AND 64),
    event        TEXT NOT NULL CHECK (length(event) BETWEEN 1 AND 40),
    strategy     TEXT NOT NULL CHECK (length(strategy) BETWEEN 1 AND 40),
    controller   TEXT NOT NULL CHECK (length(controller) BETWEEN 1 AND 40),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (namespace_id, response_id)
);

CREATE INDEX brain_by_event ON brain_decision (namespace_id, event, strategy);
