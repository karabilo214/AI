"""Bounded replay engine-событий, без изменения стандартного SSE/NDJSON."""

import copy
import threading
from collections import deque
from typing import Any

from ramvi_engine.protocol import validate_event


class EventLog:
    def __init__(self, capacity: int = 512) -> None:
        if not 1 <= capacity <= 10000:
            raise ValueError("invalid event capacity")
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._cursor = 0

    def publish(self, event: dict[str, Any]) -> None:
        event = copy.deepcopy(event)
        validate_event(event)
        with self._lock:
            self._cursor += 1
            self._events.append({"cursor": self._cursor, "event": copy.deepcopy(event)})

    def read(self, after: int = 0, request_id: str | None = None) -> dict[str, Any]:
        if after < 0:
            raise ValueError("invalid event cursor")
        with self._lock:
            first = self._events[0]["cursor"] if self._events else self._cursor + 1
            return {
                "cursor": self._cursor,
                "gap": bool(after and after < first - 1),
                "events": [
                    copy.deepcopy(row)
                    for row in self._events
                    if row["cursor"] > after
                    and (request_id is None or row["event"]["request_id"] == request_id)
                ],
            }
