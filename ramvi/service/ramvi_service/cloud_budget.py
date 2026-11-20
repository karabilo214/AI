"""7.10 REMOTE-LLM-04: учёт облачных ответов — токены, стоимость, расход за день и месяц.

Каждый ответ облачной модели — строка `cloud_usage.sqlite3` в папке данных: когда, модель,
откуда (приложение или клиент внешнего API), токены из ответа сервиса и стоимость по цене,
которую задал владелец (цена с источником и датой). Нет цены — стоимость неизвестна (NULL),
а не ноль. День и месяц — по местному времени компьютера.
"""

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    day TEXT NOT NULL,
    month TEXT NOT NULL,
    model TEXT NOT NULL,
    source TEXT NOT NULL,
    prompt INTEGER NOT NULL,
    completion INTEGER NOT NULL,
    cost REAL,
    state TEXT NOT NULL CHECK (state IN ('done', 'cut', 'error'))
);
CREATE INDEX IF NOT EXISTS usage_day ON usage (day);
CREATE INDEX IF NOT EXISTS usage_month ON usage (month);
"""


def cost_of(usage: dict[str, int], price: dict[str, Any] | None) -> float | None:
    """Стоимость в долларах по цене за миллион входных и выходных токенов."""
    if price is None:
        return None
    return round(
        (
            usage["prompt_tokens"] * float(price["input"])
            + usage["completion_tokens"] * float(price["output"])
        )
        / 1_000_000,
        8,
    )


class Budget:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _periods(self, at: float) -> tuple[str, str]:
        local = time.localtime(at)
        return time.strftime("%Y-%m-%d", local), time.strftime("%Y-%m", local)

    def record(
        self,
        model: str,
        source: str,
        usage: dict[str, int] | None,
        price: dict[str, Any] | None,
        state: str,
    ) -> dict[str, Any]:
        usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
        at = self.clock()
        day, month = self._periods(at)
        cost = (
            cost_of(usage, price) if usage["prompt_tokens"] or usage["completion_tokens"] else 0.0
        )
        with self._lock:
            self._db.execute(
                "INSERT INTO usage (at, day, month, model, source, prompt, completion, cost, state)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    at,
                    day,
                    month,
                    model,
                    source,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    cost,
                    state,
                ),
            )
        return {"model": model, "cost": cost, **usage}

    def spent(self, period: str) -> tuple[float, int]:
        """Расход за текущий день/месяц и сколько ответов без известной стоимости."""
        day, month = self._periods(self.clock())
        column, value = ("day", day) if period == "day" else ("month", month)
        with self._lock:
            total, unknown = self._db.execute(
                f"SELECT COALESCE(SUM(cost), 0), SUM(cost IS NULL) FROM usage WHERE {column} = ?",  # noqa: S608
                (value,),
            ).fetchone()
        return round(float(total), 6), int(unknown or 0)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT at, model, source, prompt, completion, cost, state FROM usage ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("at", "model", "source", "prompt_tokens", "completion_tokens", "cost", "state")
        return [dict(zip(keys, row, strict=True)) for row in rows]
