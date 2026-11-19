"""Explicit-session initiative budget. No scheduler, microphone or model runs here."""

import threading
import time
from collections.abc import Callable
from typing import Any

from ramvi_character.context import integer
from ramvi_character.personality import number

DEFAULTS = {"enabled": False, "cooldown_s": 60.0, "max_reactions": 3, "token_budget": 8192}


class InitiativeGate:
    def __init__(
        self, saved: dict[str, Any] | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock, self._lock = clock, threading.RLock()
        self._settings = dict(DEFAULTS)
        self._active = False
        self._spent = self._reactions = 0
        self._last: float | None = None
        if saved is not None:
            self.configure(saved)

    def configure(self, settings: dict[str, Any]) -> None:
        if (
            not isinstance(settings, dict)
            or set(settings) != set(DEFAULTS)
            or not isinstance(settings["enabled"], bool)
        ):
            raise ValueError("invalid initiative preferences")
        validated = {
            "enabled": settings["enabled"],
            "cooldown_s": number(settings["cooldown_s"], 1, 86400),
            "max_reactions": integer(settings["max_reactions"], 1, 1000),
            "token_budget": integer(settings["token_budget"], 1, 1_000_000),
        }
        with self._lock:
            self._settings = validated

    def begin(self) -> None:
        with self._lock:
            if not self._active:
                self._active, self._spent, self._reactions, self._last = True, 0, 0, None

    def end(self) -> None:
        with self._lock:
            self._active = False

    def claim(self, tokens: int, *, permitted: bool) -> dict[str, Any]:
        integer(tokens, 1, 1_000_000)
        if not isinstance(permitted, bool):
            raise ValueError("owner permission must be explicit")
        with self._lock:
            now = number(self._clock(), 0, 1e12)
            reason = (
                "inactive_session"
                if not self._active
                else "disabled"
                if not self._settings["enabled"]
                else "owner_boundary"
                if not permitted
                else "reaction_limit"
                if self._reactions >= self._settings["max_reactions"]
                else "token_budget"
                if self._spent + tokens > self._settings["token_budget"]
                else "cooldown"
                if self._last is not None and now - self._last < self._settings["cooldown_s"]
                else "allowed"
            )
            if reason == "allowed":
                # Charge the declared input+max-output cost before queueing the call.
                # Failed calls are not refunded, preventing unbounded automatic retries.
                self._spent += tokens
                self._reactions += 1
                self._last = now
            return {"allowed": reason == "allowed", "reason": reason, **self.status()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_active": self._active,
                "preferences": dict(self._settings),
                "spent_tokens": self._spent,
                "reactions": self._reactions,
            }

    def checkpoint(self) -> dict[str, Any]:
        # Opening the app must not resurrect an automatic reaction session.
        with self._lock:
            return dict(self._settings)
