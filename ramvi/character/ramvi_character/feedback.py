"""CHAR-18: explicit feedback and preferences, each signal with confidence and origin.

The code, not the model, decides whether a signal is a reliable assessment:
- explicit_ui (the owner pressed a button) is reliable;
- user_said is reliable only with the owner's verbatim, unambiguous words;
- interpretation (the model's reading), implicit reactions and anything ambiguous are kept
  for history but never counted as approval or disapproval.
A preference is "established" only from several agreeing reliable signals across different
responses, so one remark cannot rewrite the character. Applying preferences to traits is
CHAR-19; here they are only measured.
"""

import hashlib
import math
import re
from typing import Any

from ramvi_character.memory import MemoryStore
from ramvi_character.words import norm

DIMENSIONS = (
    "overall",
    "warmth",
    "humor",
    "sarcasm",
    "strictness",
    "curiosity",
    "verbosity",
    "initiative",
    "empathy",
)
ORIGINS = ("explicit_ui", "user_said", "interpretation", "implicit")
# Short acknowledgements say nothing about approval: "ок" after a joke is not a laugh.
AMBIGUOUS = frozenset(
    "ок окей ok okay ага угу ну ну да да ясно понятно ладно хм хмм мм ммм пусть допустим"
    " sure fine hmm mhm yeah yep k".split()
) | frozenset({"ну да", "ну ок", "ну ладно", "как скажешь", "ну хорошо", "i see", "if you say so"})
MIN_CONFIDENCE = 0.6
ESTABLISH_SIGNALS = 3
ESTABLISH_RESPONSES = 2
ESTABLISH_AGREEMENT = 0.75
WINDOW = 100  # newest signals per dimension taken into account
RESPONSE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def ambiguous_words(quote: str) -> bool:
    text = norm(re.sub(r"[^\w\s]", " ", quote))
    return not text or text in AMBIGUOUS


class FeedbackStore:
    """Signals of one namespace in the module database."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.namespace = memory.namespace

    def _signal_id(self, response_id: str, dimension: str, origin: str, quote: str) -> str:
        digest = hashlib.sha256(
            "\n".join([response_id, dimension, origin, norm(quote)]).encode()
        ).hexdigest()
        return "f_" + digest[:24]

    def add(
        self,
        response_id: str,
        dimension: str,
        direction: int,
        origin: str,
        *,
        confidence: float = 1.0,
        quote: str | None = None,
        ambiguous: bool = False,
        owner_text: str | None = None,
    ) -> dict[str, Any]:
        """Record one signal; reliability is decided here. One signal per response, dimension,
        origin and wording: pressing a button again replaces it instead of piling up."""
        if not isinstance(response_id, str) or not RESPONSE.fullmatch(response_id):
            raise ValueError("invalid feedback response")
        if dimension not in DIMENSIONS or origin not in ORIGINS:
            raise ValueError("invalid feedback dimension or origin")
        if isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError("invalid feedback direction")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid feedback confidence")
        if quote is not None and (
            not isinstance(quote, str) or not quote.strip() or len(quote) > 256
        ):
            raise ValueError("invalid feedback quote")
        if origin == "user_said":
            # The owner's words must really be in what the owner wrote.
            if quote is None or owner_text is None or norm(quote) not in norm(owner_text):
                origin = "interpretation"
        unclear = bool(ambiguous) or (quote is not None and ambiguous_words(quote))
        reliable = origin == "explicit_ui" or (
            origin == "user_said" and not unclear and confidence >= MIN_CONFIDENCE
        )
        if origin == "explicit_ui":
            unclear = False
        item = {
            "id": self._signal_id(response_id, dimension, origin, quote or ""),
            "response_id": response_id,
            "dimension": dimension,
            "direction": direction,
            "confidence": float(confidence),
            "origin": origin,
            "ambiguous": unclear,
            "reliable": reliable,
            "quote": quote,
            "created_at": self.memory.clock(),
        }
        self.memory._execute(
            "INSERT OR REPLACE INTO feedback_signal (namespace_id, id, response_id, dimension,"
            " direction, confidence, origin, ambiguous, reliable, quote, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.namespace,
                item["id"],
                response_id,
                dimension,
                direction,
                item["confidence"],
                origin,
                int(unclear),
                int(reliable),
                quote,
                item["created_at"],
            ],
        )
        return item

    def signals(
        self, dimension: str | None = None, limit: int = WINDOW, since: str | None = None
    ) -> list[dict[str, Any]]:
        if dimension is not None and dimension not in DIMENSIONS:
            raise ValueError("invalid feedback dimension")
        rows = self.memory._query(
            "SELECT * FROM feedback_signal WHERE namespace_id = ? AND (? IS NULL OR dimension = ?)"
            " AND (? IS NULL OR created_at > ?) ORDER BY created_at DESC, id LIMIT ?",
            [self.namespace, dimension, dimension, since, since, limit],
        )
        result = []
        for row in rows:
            row.pop("namespace_id")
            row["ambiguous"], row["reliable"] = bool(row["ambiguous"]), bool(row["reliable"])
            result.append(row)
        return result

    def preferences(self, since: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Per dimension: which way the reliable evidence points and whether it is settled.
        since: dimension -> moment; only newer signals count (fresh evidence for a new step)."""
        result = []
        for dimension in DIMENSIONS:
            signals = self.signals(dimension, since=(since or {}).get(dimension))
            if not signals:
                continue
            reliable = [s for s in signals if s["reliable"]]
            plus = sum(s["confidence"] for s in reliable if s["direction"] > 0)
            minus = sum(s["confidence"] for s in reliable if s["direction"] < 0)
            total = plus + minus
            direction = 0 if total == 0 or plus == minus else (1 if plus > minus else -1)
            agreement = max(plus, minus) / total if total else 0.0
            responses = len({s["response_id"] for s in reliable})
            result.append(
                {
                    "dimension": dimension,
                    "direction": direction,
                    "agreement": round(agreement, 4),
                    "reliable": len(reliable),
                    "unreliable": len(signals) - len(reliable),
                    "responses": responses,
                    "established": direction != 0
                    and len(reliable) >= ESTABLISH_SIGNALS
                    and responses >= ESTABLISH_RESPONSES
                    and agreement >= ESTABLISH_AGREEMENT,
                }
            )
        return result

    def forget_responses(self, response_ids: list[str]) -> int:
        """Deleted history takes its feedback (with the owner's quoted words) along."""
        if not response_ids:
            return 0
        return self.memory._execute(
            "DELETE FROM feedback_signal WHERE namespace_id = ? AND response_id IN ("
            + ", ".join("?" for _ in response_ids)
            + ")",
            [self.namespace, *response_ids],
        )

    def clear(self) -> int:
        return self.memory._execute(
            "DELETE FROM feedback_signal WHERE namespace_id = ?", [self.namespace]
        )
