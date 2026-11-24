-- CHAR-IMPLICIT (9.33): как владелец отреагировал на этот ответ следующей репликой
-- (оценка модели в шаге «опиши событие»: laughed, picked_up, irritated, hurt, ...).
-- Кнопки оценки убраны — это основной сигнал памяти исходов. Текст реплик не хранится.

ALTER TABLE brain_decision ADD COLUMN reaction TEXT
    CHECK (reaction IS NULL OR length(reaction) BETWEEN 1 AND 40);
