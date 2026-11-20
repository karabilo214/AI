"""CHAR-19: bounded personal corrections of traits, without touching LLM weights or the base
profile.

A step happens only for a settled preference (CHAR-18) built from signals that arrived after
the previous step on that trait, and not more often than once per INTERVAL. Each step is
STEP in size; the total correction stays within LIMIT of the base profile (also enforced by
the schema). The owner's explicit settings and boundaries outrank adaptation: a trait the
owner set is not adapted, and a trait a boundary switches off is not pushed up. Corrections
belong to one namespace (user + character) and one profile version.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from ramvi_character.feedback import FeedbackStore
from ramvi_character.memory import MemoryStore

TRAITS = (
    "warmth",
    "humor",
    "sarcasm",
    "strictness",
    "curiosity",
    "verbosity",
    "initiative",
    "empathy",
)
STEP = 0.03
LIMIT = 0.2
INTERVAL = timedelta(hours=6)
# A boundary that switches a trait off forbids adapting it upwards.
BLOCKED_UP = {"sarcasm": "teasing", "initiative": "initiative"}


def _time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


class Adaptation:
    def __init__(self, memory: MemoryStore, feedback: FeedbackStore) -> None:
        self.memory, self.feedback = memory, feedback
        self.namespace = memory.namespace

    def corrections(self, profile: str) -> dict[str, float]:
        rows = self.memory._query(
            "SELECT trait, value FROM adaptation_state WHERE namespace_id = ? AND profile = ?",
            [self.namespace, profile],
        )
        return {row["trait"]: float(row["value"]) for row in rows if row["value"] != 0}

    def history(self, profile: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.memory._query(
            "SELECT id, profile, trait, before, after, reason, created_at FROM adaptation_change"
            " WHERE namespace_id = ? AND (? IS NULL OR profile = ?)"
            " ORDER BY created_at DESC, id LIMIT ?",
            [self.namespace, profile, profile, limit],
        )
        for row in rows:
            row["reason"] = json.loads(row["reason"])
        return rows

    def _last_steps(self, profile: str) -> dict[str, str]:
        rows = self.memory._query(
            "SELECT trait, max(created_at) AS at FROM adaptation_change"
            " WHERE namespace_id = ? AND profile = ? GROUP BY trait",
            [self.namespace, profile],
        )
        return {row["trait"]: row["at"] for row in rows}

    def update(
        self, profile: str, settings: dict[str, float], boundaries: dict[str, bool]
    ) -> list[dict[str, Any]]:
        """Take at most one bounded step per trait from fresh settled evidence."""
        now = self.memory.clock()
        last = self._last_steps(profile)
        current = self.corrections(profile)
        changes = []
        for preference in self.feedback.preferences(since=last):
            trait, direction = preference["dimension"], preference["direction"]
            if trait not in TRAITS or not preference["established"] or trait in settings:
                continue
            if direction > 0 and not boundaries.get(BLOCKED_UP.get(trait, ""), True):
                continue
            if trait in last and _time(now) - _time(last[trait]) < INTERVAL:
                continue
            before = current.get(trait, 0.0)
            after = round(min(LIMIT, max(-LIMIT, before + STEP * direction)), 4)
            if after == before:
                continue
            reason = {
                "direction": direction,
                "reliable": preference["reliable"],
                "responses": preference["responses"],
                "agreement": preference["agreement"],
            }
            identifier = (
                "a_"
                + hashlib.sha256(
                    "\n".join([profile, trait, now, str(before)]).encode()
                ).hexdigest()[:24]
            )
            self.memory._execute(
                "INSERT INTO adaptation_change (namespace_id, id, profile, trait, before, after,"
                " reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    self.namespace,
                    identifier,
                    profile,
                    trait,
                    before,
                    after,
                    json.dumps(reason),
                    now,
                ],
            )
            self.memory._execute(
                "INSERT INTO adaptation_state (namespace_id, profile, trait, value, updated_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT (namespace_id, profile, trait)"
                " DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                [self.namespace, profile, trait, after, now],
            )
            changes.append({"trait": trait, "before": before, "after": after, "reason": reason})
        return changes
