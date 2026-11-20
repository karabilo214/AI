"""CHAR-19: bounded personal corrections of traits, without touching LLM weights or the base
profile.

A step happens only for a settled preference (CHAR-18) built from signals that arrived after
the previous step on that trait, and not more often than once per INTERVAL. Each step is
STEP in size; the total correction stays within LIMIT of the base profile (also enforced by
the schema). The owner's explicit settings and boundaries outrank adaptation: a trait the
owner set is not adapted, and a trait a boundary switches off is not pushed up. Corrections
belong to one namespace (user + character) and one profile version.

CHAR-20: the owner can switch adaptation off (stored corrections, memory and profile stay;
they are simply not applied and no new steps are taken), save the current setting, reset it,
and restore an earlier one. Before every change (step, reset, restore) a snapshot of the whole
profile state is written, so restoring it reproduces the earlier values exactly.
"""

import json
import uuid
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
KEEP_SNAPSHOTS = 50  # per profile; saved points are kept apart from automatic ones
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
            "SELECT id, profile, trait, before, after, reason, action, snapshot_id, created_at"
            " FROM adaptation_change WHERE namespace_id = ? AND (? IS NULL OR profile = ?)"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?",
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

    # --- CHAR-20: switch, snapshots, reset, restore -------------------------------------

    def _setting(self) -> dict[str, Any]:
        rows = self.memory._query(
            "SELECT enabled, changed_at FROM adaptation_setting WHERE namespace_id = ?",
            [self.namespace],
        )
        return (
            {"enabled": bool(rows[0]["enabled"]), "changed_at": rows[0]["changed_at"]}
            if rows
            else {"enabled": True, "changed_at": None}
        )

    @property
    def enabled(self) -> bool:
        return bool(self._setting()["enabled"])

    def set_enabled(self, enabled: bool) -> None:
        """Only the switch changes: corrections, memory and feedback stay as they are."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be true or false")
        if enabled == self.enabled:
            return  # no-op: switching on again must not discard earlier ratings
        self.memory._execute(
            "INSERT INTO adaptation_setting (namespace_id, enabled, changed_at) VALUES (?, ?, ?)"
            " ON CONFLICT (namespace_id) DO UPDATE SET enabled = excluded.enabled,"
            " changed_at = excluded.changed_at",
            [self.namespace, int(enabled), self.memory.clock()],
        )

    def applied(self, profile: str) -> dict[str, float]:
        """What the character actually uses: nothing while adaptation is switched off."""
        return self.corrections(profile) if self.enabled else {}

    def snapshots(self, profile: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.memory._query(
            "SELECT id, profile, kind, traits, created_at FROM adaptation_snapshot"
            " WHERE namespace_id = ? AND profile = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            [self.namespace, profile, limit],
        )
        for row in rows:
            row["traits"] = json.loads(row["traits"])
        return rows

    def _snapshot(self, profile: str, kind: str, now: str) -> str:
        identifier = "s_" + uuid.uuid4().hex[:24]
        self.memory._execute(
            "INSERT INTO adaptation_snapshot (namespace_id, id, profile, kind, traits, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                self.namespace,
                identifier,
                profile,
                kind,
                json.dumps(self.corrections(profile), sort_keys=True),
                now,
            ],
        )
        # Automatic snapshots are bounded; points the owner saved are never pruned here.
        self.memory._execute(
            "DELETE FROM adaptation_snapshot WHERE namespace_id = ? AND profile = ?"
            " AND kind != 'saved' AND rowid NOT IN (SELECT rowid FROM adaptation_snapshot"
            " WHERE namespace_id = ? AND profile = ? AND kind != 'saved'"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?)",
            [self.namespace, profile, self.namespace, profile, KEEP_SNAPSHOTS],
        )
        return identifier

    def export(self) -> dict[str, Any]:
        """Everything adaptation keeps for this owner, across profiles."""
        state = self.memory._query(
            "SELECT profile, trait, value, updated_at FROM adaptation_state"
            " WHERE namespace_id = ? ORDER BY profile, trait",
            [self.namespace],
        )
        snapshots = self.memory._query(
            "SELECT id, profile, kind, traits, created_at FROM adaptation_snapshot"
            " WHERE namespace_id = ? ORDER BY created_at, rowid",
            [self.namespace],
        )
        for row in snapshots:
            row["traits"] = json.loads(row["traits"])
        return {
            "enabled": self.enabled,
            "state": state,
            "history": self.history(None, 1000),
            "snapshots": snapshots,
        }

    def save(self, profile: str) -> dict[str, Any]:
        """Remember the current setting as a point to come back to."""
        identifier = self._snapshot(profile, "saved", self.memory.clock())
        return next(
            s for s in self.snapshots(profile, KEEP_SNAPSHOTS + 50) if s["id"] == identifier
        )

    def _write(
        self,
        profile: str,
        trait: str,
        before: float,
        after: float,
        reason: dict[str, Any],
        action: str,
        snapshot: str,
        now: str,
    ) -> None:
        self.memory._execute(
            "INSERT INTO adaptation_change (namespace_id, id, profile, trait, before, after,"
            " reason, action, snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.namespace,
                "a_" + uuid.uuid4().hex[:24],
                profile,
                trait,
                before,
                after,
                json.dumps(reason),
                action,
                snapshot,
                now,
            ],
        )
        if after == 0:
            self.memory._execute(
                "DELETE FROM adaptation_state WHERE namespace_id = ? AND profile = ? AND trait = ?",
                [self.namespace, profile, trait],
            )
            return
        self.memory._execute(
            "INSERT INTO adaptation_state (namespace_id, profile, trait, value, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT (namespace_id, profile, trait)"
            " DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [self.namespace, profile, trait, after, now],
        )

    def _set_all(
        self, profile: str, target: dict[str, float], action: str, kind: str
    ) -> list[dict[str, Any]]:
        current = self.corrections(profile)
        traits = sorted(set(current) | set(target))
        changed = [t for t in traits if current.get(t, 0.0) != target.get(t, 0.0)]
        if not changed:
            return []
        now = self.memory.clock()
        snapshot = self._snapshot(profile, kind, now)
        changes = []
        for trait in changed:
            before, after = current.get(trait, 0.0), target.get(trait, 0.0)
            self._write(profile, trait, before, after, {"action": action}, action, snapshot, now)
            changes.append({"trait": trait, "before": before, "after": after})
        return changes

    def reset(self, profile: str) -> list[dict[str, Any]]:
        """Back to the base profile; the previous setting stays restorable."""
        return self._set_all(profile, {}, "reset", "before_reset")

    def restore(self, profile: str, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        """Put back a snapshot (by default the state before the last change: undo)."""
        if snapshot_id is None:
            # The state before the last change; points the owner saved are restored by id.
            latest = [s for s in self.snapshots(profile, 200) if s["kind"] != "saved"][:1]
            if not latest:
                raise ValueError("nothing to restore")
            snapshot = latest[0]
        else:
            rows = self.memory._query(
                "SELECT traits FROM adaptation_snapshot WHERE namespace_id = ? AND profile = ?"
                " AND id = ?",
                [self.namespace, profile, snapshot_id],
            )
            if not rows:
                raise ValueError("unknown snapshot")
            snapshot = {"traits": json.loads(rows[0]["traits"])}
        target = snapshot["traits"]
        if not isinstance(target, dict) or not all(
            trait in TRAITS and isinstance(value, int | float) and abs(value) <= LIMIT
            for trait, value in target.items()
        ):
            raise ValueError("invalid snapshot")
        return self._set_all(
            profile, {t: float(v) for t, v in target.items()}, "restore", "before_restore"
        )

    def update(
        self, profile: str, settings: dict[str, float], boundaries: dict[str, bool]
    ) -> list[dict[str, Any]]:
        """Take at most one bounded step per trait from fresh settled evidence."""
        setting = self._setting()
        if not setting["enabled"]:
            return []
        now = self.memory.clock()
        last = self._last_steps(profile)
        since = dict(last)
        if setting["changed_at"] is not None:
            # Ratings given while adaptation was off do not count once it is back on.
            for trait in TRAITS:
                since[trait] = max(since.get(trait, ""), setting["changed_at"])
        current = self.corrections(profile)
        changes = []
        snapshot: str | None = None
        for preference in self.feedback.preferences(since=since):
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
            if snapshot is None:
                snapshot = self._snapshot(profile, "before_step", now)
            self._write(profile, trait, before, after, reason, "step", snapshot, now)
            changes.append({"trait": trait, "before": before, "after": after, "reason": reason})
        return changes
