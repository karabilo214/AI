"""CHAR-14: memory records and Character State in the module's own SQLite database.

Every statement carries an explicit namespace (user + character), so records of different
users or characters never mix. The database file itself belongs to this module only (3.12).
Search (9.15) and consolidation of candidates (9.16) build on these tables.
"""

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from typing import Any

from ramvi_character.words import norm, stems

KINDS = ("working", "profile", "episodic", "semantic", "behavioral")
STATUSES = ("candidate", "active", "disputed", "superseded")
# Who the knowledge comes from (spec 10.5): only the owner's words and confirmed
# observations may become active; interpretations and assumptions stay unconfirmed.
ORIGINS = ("user_said", "observed", "interpretation", "assumption")
CONFIRMABLE = frozenset({"user_said", "observed"})
FIELDS = (
    "id",
    "kind",
    "type",
    "subject",
    "value",
    "confidence",
    "occurrences",
    "first_seen",
    "last_seen",
    "status",
    "origin",
    "source_ids",
    "expires_at",
    "importance",
    "unfinished",
    "topics",
    "entities",
)
JSON_FIELDS = frozenset({"source_ids", "topics", "entities"})
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
TYPE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
SOURCE = re.compile(r"[A-Za-z0-9_:.-]{1,96}\Z")
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
MAX_SOURCES = 32
MAX_TAGS = 16
STATE_SCHEMA = "ramvi.character.live_state/1"
# Relative sources ("current", "task") point to a different message every turn: not a trace.
STABLE_SOURCE = re.compile(r"(response|recent|memory|owner):[A-Za-z0-9_.:-]+\Z")
FORGOTTEN_SIMILAR = 0.6


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _text(value: Any, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid memory {name}")
    return value


def _stamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not STAMP.fullmatch(value):
        raise ValueError(f"invalid memory {name}")
    return value


def _sources(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_SOURCES
        or not all(isinstance(s, str) and SOURCE.fullmatch(s) for s in value)
    ):
        raise ValueError("invalid memory source ids")
    return list(dict.fromkeys(value))


def _tags(value: Any) -> list[str]:
    """Topics/entities are compared exactly, so they are case-folded here (SQLite lower() is ASCII)."""
    if not isinstance(value, list) or len(value) > MAX_TAGS:
        raise ValueError("invalid memory tags")
    result = []
    for tag in value:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > 64:
            raise ValueError("invalid memory tag")
        result.append(" ".join(tag.casefold().split()))
    return list(dict.fromkeys(result))


def validate(record: Any, now: str) -> dict[str, Any]:
    """Full validated record; missing occurrences/first/last seen are filled for a new fact."""
    if not isinstance(record, dict) or set(record) - set(FIELDS):
        raise ValueError("invalid memory record fields")
    item = {
        "id": record.get("id"),
        "kind": record.get("kind"),
        "type": record.get("type"),
        "subject": _text(record.get("subject"), 256, "subject"),
        "value": _text(record.get("value"), 4096, "value"),
        "confidence": record.get("confidence"),
        "occurrences": record.get("occurrences", 1),
        "first_seen": _stamp(record.get("first_seen", now), "first_seen"),
        "last_seen": _stamp(record.get("last_seen", now), "last_seen"),
        "status": record.get("status", "candidate"),
        "origin": record.get("origin"),
        "source_ids": _sources(record.get("source_ids", [])),
        "expires_at": record.get("expires_at"),
        "importance": record.get("importance", 0.5),
        "unfinished": record.get("unfinished", False),
        "topics": _tags(record.get("topics", [])),
        "entities": _tags(record.get("entities", [])),
    }
    if not isinstance(item["id"], str) or not IDENTIFIER.fullmatch(item["id"]):
        raise ValueError("invalid memory id")
    if item["kind"] not in KINDS:
        raise ValueError("invalid memory kind")
    if not isinstance(item["type"], str) or not TYPE.fullmatch(item["type"]):
        raise ValueError("invalid memory type")
    confidence = item["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("invalid memory confidence")
    item["confidence"] = float(confidence)
    importance = item["importance"]
    if (
        isinstance(importance, bool)
        or not isinstance(importance, (int, float))
        or not math.isfinite(importance)
        or not 0 <= importance <= 1
    ):
        raise ValueError("invalid memory importance")
    item["importance"] = float(importance)
    if not isinstance(item["unfinished"], bool):
        raise ValueError("invalid memory unfinished flag")
    occurrences = item["occurrences"]
    if isinstance(occurrences, bool) or not isinstance(occurrences, int) or occurrences < 1:
        raise ValueError("invalid memory occurrences")
    if item["status"] not in STATUSES:
        raise ValueError("invalid memory status")
    if item["origin"] not in ORIGINS:
        raise ValueError("invalid memory origin")
    if item["status"] == "active" and item["origin"] not in CONFIRMABLE:
        raise ValueError("interpretations and assumptions cannot be active facts")
    if item["last_seen"] < item["first_seen"]:
        raise ValueError("memory last_seen precedes first_seen")
    if item["expires_at"] is not None:
        _stamp(item["expires_at"], "expires_at")
        if item["kind"] != "working":
            raise ValueError("only working memory expires")
    return item


class MemoryStore:
    """Scoped access to memory_record / character_state for one user + character."""

    def __init__(
        self,
        services: Any,
        user_id: str = "owner",
        character_id: str = "default",
        clock: Callable[[], str] = utc_now,
    ) -> None:
        for value in (user_id, character_id):
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                raise ValueError("invalid memory namespace")
        self.services, self.clock = services, clock
        self.user_id, self.character_id = user_id, character_id
        self.namespace = "u:" + user_id + "/c:" + character_id
        self._execute(
            "INSERT OR IGNORE INTO memory_namespace (id, user_id, character_id, created_at)"
            " VALUES (?, ?, ?, ?)",
            [self.namespace, user_id, character_id, self.clock()],
        )

    def _query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        result = self.services.call("data.query", {"sql": sql, "params": params})
        return [dict(zip(result["columns"], row, strict=True)) for row in result["rows"]]

    def _execute(self, sql: str, params: list[Any]) -> int:
        rowcount = self.services.call("data.execute", {"sql": sql, "params": params})["rowcount"]
        return int(rowcount)

    @staticmethod
    def _row(row: dict[str, Any]) -> dict[str, Any]:
        item = {key: row[key] for key in FIELDS}
        for key in JSON_FIELDS:
            item[key] = json.loads(row[key])
        item["unfinished"] = bool(row["unfinished"])
        return item

    def put(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or fully replace one record of this namespace."""
        item = validate(record, self.clock())
        self._execute(
            "INSERT INTO memory_record (namespace_id, "
            + ", ".join(FIELDS)
            + ") VALUES (?, "
            + ", ".join("?" for _ in FIELDS)
            + ") ON CONFLICT (namespace_id, id) DO UPDATE SET "
            + ", ".join(f"{key} = excluded.{key}" for key in FIELDS[1:]),
            [self.namespace]
            + [
                json.dumps(item[k], ensure_ascii=False)
                if k in JSON_FIELDS
                else int(item[k])
                if k == "unfinished"
                else item[k]
                for k in FIELDS
            ],
        )
        return copy.deepcopy(item)

    def get(self, identifier: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT * FROM memory_record WHERE namespace_id = ? AND id = ?",
            [self.namespace, identifier],
        )
        return self._row(rows[0]) if rows else None

    def records(
        self, kind: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Newest first; expired working memory is not returned."""
        if kind is not None and kind not in KINDS or status is not None and status not in STATUSES:
            raise ValueError("invalid memory filter")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("invalid memory limit")
        rows = self._query(
            "SELECT * FROM memory_record WHERE namespace_id = ?"
            " AND (? IS NULL OR kind = ?) AND (? IS NULL OR status = ?)"
            " AND (expires_at IS NULL OR expires_at > ?)"
            " ORDER BY last_seen DESC, id LIMIT ?",
            [self.namespace, kind, kind, status, status, self.clock(), limit],
        )
        return [self._row(row) for row in rows]

    def by_subject(self, subject: str, kind: str | None = None) -> list[dict[str, Any]]:
        """Exact subject lookup through its index (consolidation compares within a subject)."""
        if kind is not None and kind not in KINDS:
            raise ValueError("invalid memory filter")
        rows = self._query(
            "SELECT * FROM memory_record WHERE namespace_id = ? AND subject = ?"
            " AND (? IS NULL OR kind = ?) ORDER BY last_seen DESC, id LIMIT 100",
            [self.namespace, _text(subject, 256, "subject"), kind, kind],
        )
        return [self._row(row) for row in rows]

    def observe(self, identifier: str, source_ids: list[str]) -> dict[str, Any]:
        """The same knowledge met again: occurrences + 1, last_seen now, sources merged."""
        added = _sources(source_ids)
        for _attempt in range(3):
            current = self.get(identifier)
            if current is None:
                raise KeyError(identifier)
            merged = list(dict.fromkeys(current["source_ids"] + added))[-MAX_SOURCES:]
            now = max(self.clock(), current["last_seen"])
            changed = self._execute(
                "UPDATE memory_record SET occurrences = occurrences + 1, last_seen = ?,"
                " source_ids = ? WHERE namespace_id = ? AND id = ? AND occurrences = ?",
                [now, json.dumps(merged), self.namespace, identifier, current["occurrences"]],
            )
            if changed == 1:
                updated = self.get(identifier)
                assert updated is not None
                return updated
        raise RuntimeError("memory record is being changed concurrently")

    def _hash(self, value: str) -> str:
        return hashlib.sha256((self.namespace + "\n" + value).encode()).hexdigest()[:32]

    def forget(self, identifier: str, optimize: bool = True) -> bool:
        """Delete on the owner's request. Only a textless trace stays, so the same knowledge
        is not restored automatically from the same sources; the search index is cleaned."""
        record = self.get(identifier)
        if record is None:
            return False
        trace = sorted({self._hash(s) for s in stems(record["subject"] + " " + record["value"])})
        self._execute(
            "INSERT OR REPLACE INTO memory_forgotten (namespace_id, id, kind, subject_hash,"
            " stem_hashes, source_ids, forgotten_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                self.namespace,
                self._hash("record\n" + identifier),
                record["kind"],
                self._hash(norm(record["subject"])),
                json.dumps(trace),
                json.dumps([s for s in record["source_ids"] if STABLE_SOURCE.fullmatch(s)]),
                self.clock(),
            ],
        )
        deleted = self._execute(
            "DELETE FROM memory_record WHERE namespace_id = ? AND id = ?",
            [self.namespace, identifier],
        )
        if optimize:
            self.optimize()
        return deleted == 1

    def optimize(self) -> None:
        # FTS5 keeps deleted terms in old segments until a merge.
        self._execute("INSERT INTO memory_fts (memory_fts) VALUES ('optimize')", [])

    def forgotten(self, kind: str, subject: str, value: str, source_ids: list[str]) -> bool:
        """True when this knowledge was forgotten before: same stable sources, or the same
        subject with mostly the same words (compared through salted hashes only)."""
        stable = [s for s in source_ids if STABLE_SOURCE.fullmatch(s)]
        if stable:
            rows = self._query(
                "SELECT 1 FROM memory_forgotten f, json_each(f.source_ids) j"
                " WHERE f.namespace_id = ? AND j.value IN ("
                + ", ".join("?" for _ in stable)
                + ") LIMIT 1",
                [self.namespace, *stable],
            )
            if rows:
                return True
        words = {self._hash(s) for s in stems(subject + " " + value)}
        rows = self._query(
            "SELECT stem_hashes FROM memory_forgotten"
            " WHERE namespace_id = ? AND subject_hash = ? AND kind = ? LIMIT 100",
            [self.namespace, self._hash(norm(subject)), kind],
        )
        for row in rows:
            old = set(json.loads(row["stem_hashes"]))
            if words and old and len(words & old) / len(words | old) >= FORGOTTEN_SIMILAR:
                return True
        return False

    def forget_all(self) -> int:
        """Forget every record of this namespace (other users and characters untouched)."""
        count = 0
        while True:
            batch = self._query(
                "SELECT id FROM memory_record WHERE namespace_id = ? LIMIT 500", [self.namespace]
            )
            if not batch:
                self.optimize()
                return count
            for row in batch:
                count += int(self.forget(row["id"], optimize=False))

    def forget_sources(self, sources: set[str]) -> int:
        """Deleted history: records learned only from it are forgotten; others lose those
        sources and stay. Returns how many records were forgotten."""
        stable = sorted(s for s in sources if STABLE_SOURCE.fullmatch(s))
        if not stable:
            return 0
        rows = self._query(
            "SELECT DISTINCT r.id FROM memory_record r, json_each(r.source_ids) j"
            " WHERE r.namespace_id = ? AND j.value IN ("
            + ", ".join("?" for _ in stable)
            + ") LIMIT 1000",
            [self.namespace, *stable],
        )
        forgotten = 0
        for row in rows:
            record = self.get(row["id"])
            if record is None:
                continue
            kept = [s for s in record["source_ids"] if s not in sources]
            if any(STABLE_SOURCE.fullmatch(s) for s in kept):
                self.put({**record, "source_ids": kept})
            else:
                forgotten += int(self.forget(record["id"], optimize=False))
        if forgotten:
            self.optimize()
        return forgotten

    def compact(self) -> dict[str, Any]:
        """After deletions: flush the WAL journal and drop old migration copies."""
        return dict(self.services.call("data.compact", {}))

    def save_state(self, state: dict[str, Any]) -> int:
        """Character State is stored apart from memories; returns its new revision."""
        if not isinstance(state, dict):
            raise ValueError("invalid character state")
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 64_000:
            raise ValueError("character state too large")
        self._execute(
            "INSERT INTO character_state (namespace_id, schema, state, revision, updated_at)"
            " VALUES (?, ?, ?, 1, ?) ON CONFLICT (namespace_id) DO UPDATE SET"
            " schema = excluded.schema, state = excluded.state,"
            " revision = character_state.revision + 1, updated_at = excluded.updated_at",
            [self.namespace, STATE_SCHEMA, encoded, self.clock()],
        )
        rows = self._query(
            "SELECT revision FROM character_state WHERE namespace_id = ?", [self.namespace]
        )
        return int(rows[0]["revision"])

    def load_state(self) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT schema, state FROM character_state WHERE namespace_id = ?", [self.namespace]
        )
        if not rows:
            return None
        if rows[0]["schema"] != STATE_SCHEMA:
            raise ValueError("unsupported character state schema")
        state: dict[str, Any] = json.loads(rows[0]["state"])
        return state
