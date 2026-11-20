"""CHAR-15: combined memory search over a bounded set of SQL candidates.

Candidates come only from indexed queries with LIMIT (full-text, exact topics/entities, the
current task, unfinished events, salient records), so the whole history is never loaded.
Every candidate is then scored by several signals. Semantic similarity is an optional
plug-in that only re-ranks those candidates with a bounded weight: it can neither add a
record nobody else found nor outweigh text, topics and the task together.
"""

import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from ramvi_character.memory import MemoryStore

# similarity(query, texts) -> one value in [0, 1] per text; e.g. embeddings cosine.
Similarity = Callable[[str, Sequence[str]], Sequence[float]]

WEIGHTS = {
    "text": 0.28,
    "tags": 0.22,
    "task": 0.14,
    "semantic": 0.14,
    "importance": 0.08,
    "recency": 0.06,
    "unfinished": 0.05,
    "frequency": 0.03,
}
CANDIDATES = 48  # per source query
RECENCY_HALF_LIFE_DAYS = 30.0
STOP = frozenset(
    "и в во на не что это как а но или ли же бы то за по из у о об от до для с со к ко"
    " я ты он она мы вы они мне меня тебе тебя его её ее нам вас их мой твой наш ваш"
    " был была были быть есть было будет очень там тут где когда так уже ещё еще"
    " the a an and or but is are was were be to of in on at for with it this that"
    " i you he she we they me my your our their do does did not what how".split()
)
WORD = re.compile(r"\w+", re.UNICODE)


def terms(value: str, limit: int = 16) -> list[str]:
    result = []
    for word in WORD.findall(value.casefold()):
        if word in STOP or len(word) < 2 or (word.isdigit() and len(word) < 3):
            continue
        if word not in result:
            result.append(word)
    return result[:limit]


def fts_query(words: list[str]) -> str | None:
    """OR of prefix terms; a short prefix catches Russian word endings (файл/файлы/файлом)."""
    parts = []
    for word in words:
        stem = word[: max(4, len(word) - 2)] if len(word) >= 5 else word
        parts.append('"' + stem.replace('"', "") + '"*')
    return " OR ".join(parts) if parts else None


def tag_keys(words: list[str]) -> list[str]:
    """Single words plus neighbour pairs, for multi-word topics like «английский язык»."""
    pairs = [a + " " + b for a, b in zip(words, words[1:], strict=False)]
    pairs += [a + "_" + b for a, b in zip(words, words[1:], strict=False)]
    return list(dict.fromkeys(words + pairs))[:64]


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _age_days(stamp: str, now: str) -> float:
    return max(0.0, (_parse(now) - _parse(stamp)).total_seconds() / 86_400)


class Recall:
    """Search one namespace of a MemoryStore."""

    def __init__(self, store: MemoryStore, similarity: Similarity | None = None) -> None:
        self.store, self.similarity = store, similarity
        self.rows_read = 0  # diagnostics/tests: how many rows SQL actually returned

    def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        rows = self.store._query(sql, params)
        self.rows_read += len(rows)
        return rows

    def _live(self) -> str:
        return "r.status <> 'superseded' AND (r.expires_at IS NULL OR r.expires_at > ?)"

    def _text(self, query: str | None, now: str) -> dict[str, float]:
        if query is None:
            return {}
        rows = self._rows(
            "SELECT r.id, bm25(memory_fts) AS rank FROM memory_fts"
            " JOIN memory_record r ON r.namespace_id = memory_fts.namespace_id"
            " AND r.id = memory_fts.memory_id"
            " WHERE memory_fts MATCH ? AND memory_fts.namespace_id = ? AND "
            + self._live()
            + " ORDER BY rank LIMIT ?",
            [query, self.store.namespace, now, CANDIDATES],
        )
        best = min((row["rank"] for row in rows), default=0.0)
        # bm25 is negative, smaller is better: scale to (0, 1] relative to the best match.
        return {row["id"]: (row["rank"] / best if best < 0 else 1.0) for row in rows}

    def _tags(self, keys: list[str], now: str) -> dict[str, float]:
        if not keys:
            return {}
        rows = self._rows(
            "SELECT r.id, count(*) AS hits FROM memory_tag t"
            " JOIN memory_record r ON r.namespace_id = t.namespace_id AND r.id = t.memory_id"
            " WHERE t.namespace_id = ? AND t.tag IN ("
            + ", ".join("?" for _ in keys)
            + ") AND "
            + self._live()
            + " GROUP BY r.id ORDER BY hits DESC, r.last_seen DESC LIMIT ?",
            [self.store.namespace, *keys, now, CANDIDATES],
        )
        return {row["id"]: min(1.0, row["hits"] / 2) for row in rows}

    def _extra(self, condition: str, order: str, now: str) -> list[str]:
        rows = self._rows(
            "SELECT r.id FROM memory_record r WHERE r.namespace_id = ? AND "
            + condition
            + " AND "
            + self._live()
            + " ORDER BY "
            + order
            + " LIMIT ?",
            [self.store.namespace, now, CANDIDATES // 3],
        )
        return [row["id"] for row in rows]

    def search(self, query: str, task: str = "", limit: int = 8) -> list[dict[str, Any]]:
        """Best records with relevance in [0, 1] and the signals that produced it."""
        if not isinstance(query, str) or not isinstance(task, str):
            raise ValueError("search text must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
            raise ValueError("invalid search limit")
        now = self.store.clock()
        words, task_words = terms(query), terms(task)
        text = self._text(fts_query(words), now)
        task_text = self._text(fts_query(task_words), now)
        tags = self._tags(tag_keys(words + task_words), now)
        ids = list(dict.fromkeys([*text, *tags, *task_text]))
        ids += self._extra("r.unfinished = 1", "r.last_seen DESC", now)
        ids += self._extra("1", "r.importance DESC, r.last_seen DESC", now)
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        rows = self._rows(
            "SELECT * FROM memory_record WHERE namespace_id = ? AND id IN ("
            + ", ".join("?" for _ in ids)
            + ")",
            [self.store.namespace, *ids],
        )
        records = [self.store._row(row) for row in rows]
        semantic = [0.0] * len(records)
        if self.similarity is not None and records:
            texts = [r["subject"] + ": " + r["value"] for r in records]
            values = list(self.similarity(query + "\n" + task, texts))
            if len(values) != len(records):
                raise ValueError("similarity must return one value per text")
            semantic = [min(1.0, max(0.0, float(v))) if math.isfinite(v) else 0.0 for v in values]
        found = []
        for record, similar in zip(records, semantic, strict=True):
            signals = {
                "text": text.get(record["id"], 0.0),
                "tags": tags.get(record["id"], 0.0),
                "task": task_text.get(record["id"], 0.0),
                "semantic": similar,
                "importance": record["importance"],
                "recency": 0.5 ** (_age_days(record["last_seen"], now) / RECENCY_HALF_LIFE_DAYS),
                "unfinished": 1.0 if record["unfinished"] else 0.0,
                "frequency": min(1.0, math.log2(1 + record["occurrences"]) / 5),
            }
            relevance = sum(WEIGHTS[k] * v for k, v in signals.items())
            found.append({**record, "relevance": round(relevance, 6), "signals": signals})
        found.sort(key=lambda r: (-r["relevance"], -r["confidence"], r["id"]))
        return found[:limit]

    def context_items(
        self, query: str, task: str, scope: str, limit: int = 16
    ) -> list[dict[str, Any]]:
        """The ContextOrchestrator memory format; origin and status travel with the text."""
        status = {"active": "confirmed", "candidate": "unconfirmed", "disputed": "disputed"}
        return [
            {
                "id": r["id"],
                "scope": scope,
                "text": f"[{r['kind']}/{r['type']}, {r['origin']}] {r['subject']}: {r['value']}",
                "relevance": min(1.0, r["relevance"]),
                "confidence": r["confidence"],
                "status": status[r["status"]],
            }
            for r in self.search(query, task, limit)
        ]
