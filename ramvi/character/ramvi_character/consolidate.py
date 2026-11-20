"""CHAR-16: memory candidates -> records, in a bounded background worker.

The model only proposes. The code decides where knowledge comes from:
- user_said: the candidate quotes the owner's words and the quote is really there;
- interpretation: it cites the owner's message but the words are the model's reading;
- assumption: no owner message behind it, or the model itself is unsure.
New records are never active facts. The owner's own words become active only after they
recur in another response; interpretations and assumptions need the owner's confirmation.
A different value for the same subject is a contradiction: it is stored as disputed and
never replaces what is known. Only completed responses submit candidates.
"""

import hashlib
import queue
import threading
from typing import Any

from ramvi_character.memory import MemoryStore
from ramvi_character.words import norm as _norm
from ramvi_character.words import stems as _stems
from ramvi_character.words import terms

KIND = {"preference": "profile", "event": "episodic", "fact": "semantic"}
QUEUE = 16
SAME = 0.6  # word overlap that counts as the same knowledge in other words


def same_knowledge(a: str, b: str) -> bool:
    left, right = _stems(a), _stems(b)
    if not left or not right:
        return _norm(a) == _norm(b)
    return len(left & right) / len(left | right) >= SAME


def origin(candidate: dict[str, Any], owner_words: dict[str, str]) -> str:
    """owner_words: source id ("current", "recent:<id>") -> what the owner actually said."""
    cited = [owner_words[e] for e in candidate["evidence"] if e in owner_words]
    quote = candidate.get("quote")
    if quote and any(_norm(quote) in _norm(words) for words in cited):
        return "user_said"
    if cited and candidate["confidence"] >= 0.5:
        return "interpretation"
    return "assumption"


def record_id(kind: str, subject: str, value: str) -> str:
    digest = hashlib.sha256((kind + "\n" + _norm(subject) + "\n" + _norm(value)).encode())
    return "m_" + digest.hexdigest()[:20]


class Consolidator:
    """Applies candidates one by one off the response path; stops with the module."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(QUEUE)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self.stats = {"processed": 0, "dropped": 0, "errors": 0, "forgotten": 0}

    def submit(
        self, response_id: str, candidates: list[dict[str, Any]], owner_words: dict[str, str]
    ) -> bool:
        """Queue a completed response; a full queue drops it instead of blocking speech."""
        if self._stopped.is_set() or not candidates:
            return False
        try:
            self._queue.put_nowait(
                {"response": response_id, "candidates": candidates, "owner": dict(owner_words)}
            )
        except queue.Full:
            with self._lock:
                self.stats["dropped"] += 1
            return False
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._work, name="character-memory", daemon=True
                )
                self._thread.start()
        return True

    def _work(self) -> None:
        # The thread lives only while there is work: no permanent background activity.
        while not self._stopped.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                return
            for candidate in job["candidates"]:
                if self._stopped.is_set():
                    return
                try:
                    self.apply(job["response"], candidate, job["owner"])
                    with self._lock:
                        self.stats["processed"] += 1
                except Exception:
                    with self._lock:
                        self.stats["errors"] += 1

    def apply(
        self, response_id: str, candidate: dict[str, Any], owner_words: dict[str, str]
    ) -> dict[str, Any] | None:
        kind = KIND[candidate["kind"]]
        value = candidate["text"]
        subject = _norm(candidate.get("subject") or " ".join(terms(value, 6)) or value)[:256]
        source = origin(candidate, owner_words)
        # "current" is this turn's owner message; its history id keeps the trace stable.
        evidence = [
            "recent:" + response_id + "u" if e == "current" else e for e in candidate["evidence"]
        ]
        sources = list(dict.fromkeys(["response:" + response_id, *evidence]))[:8]
        fresh = source == "user_said" and "current" in candidate["evidence"]
        if not fresh and self.store.forgotten(kind, subject, value, sources):
            # The owner forgot it: the model may not bring it back from the same material.
            with self._lock:
                self.stats["forgotten"] += 1
            return None
        confidence = float(candidate["confidence"])
        known = [r for r in self.store.by_subject(subject, kind) if r["status"] != "superseded"]
        for record in known:
            if same_knowledge(record["value"], value):
                repeated = "response:" + response_id not in record["source_ids"]
                updated = self.store.observe(record["id"], sources)
                upgrade = source == "user_said" and record["origin"] != "user_said"
                # The owner's own words met again in another response become a fact.
                promote = (
                    repeated
                    and record["status"] == "candidate"
                    and record["origin"] == "user_said"
                    and source == "user_said"
                )
                if upgrade or promote or confidence > record["confidence"]:
                    updated = self.store.put(
                        {
                            **updated,
                            "origin": "user_said" if upgrade else updated["origin"],
                            "status": "active" if promote else updated["status"],
                            "confidence": max(confidence, updated["confidence"]),
                        }
                    )
                return updated
        disputed = bool(known)
        for record in known:
            if record["status"] == "candidate":  # two unconfirmed versions: both disputed
                self.store.put({**record, "status": "disputed"})
        return self.store.put(
            {
                "id": record_id(kind, subject, value),
                "kind": kind,
                "type": candidate["kind"],
                "subject": subject,
                "value": value,
                "confidence": confidence,
                "status": "disputed" if disputed else "candidate",
                "origin": source,
                "source_ids": sources,
            }
        )

    def confirm(self, identifier: str) -> dict[str, Any]:
        """The owner confirms: it becomes their own words; conflicting versions are superseded."""
        record = self.store.get(identifier)
        if record is None:
            raise KeyError(identifier)
        for other in self.store.by_subject(record["subject"], record["kind"]):
            if (
                other["id"] != identifier
                and other["status"] != "superseded"
                and not same_knowledge(other["value"], record["value"])
            ):
                self.store.put({**other, "status": "superseded"})
        return self.store.put(
            {
                **record,
                "status": "active",
                "origin": "user_said",
                "confidence": max(0.9, record["confidence"]),
            }
        )

    def correct(self, identifier: str, value: str, subject: str | None = None) -> dict[str, Any]:
        """The owner's correction replaces the record; the wrong version is forgotten."""
        record = self.store.get(identifier)
        if record is None:
            raise KeyError(identifier)
        subject = _norm(subject or record["subject"])[:256]
        self.store.forget(identifier)
        return self.store.put(
            {
                "id": record_id(record["kind"], subject, value),
                "kind": record["kind"],
                "type": record["type"],
                "subject": subject,
                "value": value,
                "confidence": 1.0,
                "status": "active",
                "origin": "user_said",
                "source_ids": ["owner:correction"],
                "importance": record["importance"],
                "topics": record["topics"],
                "entities": record["entities"],
                "first_seen": record["first_seen"],
            }
        )

    def wait(self, timeout: float = 5) -> bool:
        """Tests/diagnostics: True when the queue is drained."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self._queue.empty() and (thread is None or not thread.is_alive())

    def stop(self, wait: bool = True) -> None:
        self._stopped.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if wait and self._thread is not None:
            self._thread.join(timeout=5)
