"""Character output policy. Audio failures preserve text and require explicit recovery."""

import copy
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from ramvi_character.delivery import ReactionBank
from ramvi_character.delivery import settings as delivery_settings

SEGMENT_QUEUE = 3
SAFE_ERROR = re.compile(
    r"speech\.(?:disabled|remote_text_disabled|key_missing|secrets_unavailable|voice_not_supported|voice_missing|busy|device_unavailable|device_lost|device_timeout|device_close_failed|provider_timeout|audio_timeout|deadline|provider_failed|cleanup_failed|failed|closed|api_http_[1-5]\d\d)\b"
)


# 9.11: подача словами — облачный голос лучше понимает описание, чем ярлык (замер на Gemini:
# «вскрик» поднимает высоту голоса ~186 → ~281 Гц). Указание в самом тексте не годится:
# голос читает его вслух.
VOICE_DELIVERY = {
    "neutral": "natural, even delivery",
    "excited": "excited and energetic, faster",
    "intriguing": "intriguing, slightly lowered voice with a mysterious pause",
    "calm": "calm and steady",
    "warm": "warm and kind",
    "playful": "playful and teasing, with a smile in the voice",
    "sarcastic": "dry and sarcastic, mocking intonation",
    "irritated": "irritated, clipped and impatient",
    "shouting": "loud, almost shouting",
    "whisper": "whispering, quiet and breathy",
    "sad": "sad and quiet, slower",
    "thoughtful": "thoughtful, unhurried, with pauses",
    "startled": "a sudden loud startled exclamation, almost a shout",
    "reproachful": "calmer but annoyed and reproachful, with a weary intonation",
    "grumbling": "grumbling under the breath, low and displeased",
    "laughing": "laughing while speaking",
    "weary": "tired and weary, dragging the words",
}


# Владелец 2026-11-24 (прослушивание 9.11): ответ из частей одной тональности звучит лучше одним
# запросом (интонация не рвётся), а при резкой смене подачи — частями (смена слышна). Части одного
# семейства склеиваются; вскрик и крик — всегда отдельно.
DELIVERY_FAMILY = {
    "irritated": "harsh",
    "sarcastic": "harsh",
    "grumbling": "harsh",
    "reproachful": "harsh",
    "excited": "bright",
    "playful": "bright",
    "laughing": "bright",
    "warm": "bright",
    "neutral": "even",
    "calm": "even",
    "thoughtful": "even",
    "intriguing": "even",
    "sad": "low",
    "weary": "low",
}
BURSTS = frozenset({"startled", "shouting", "whisper"})


def same_delivery(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Можно ли озвучить две соседние части одним запросом."""
    a, b = first.get("voice_style", "neutral"), second.get("voice_style", "neutral")
    if a in BURSTS or b in BURSTS or first.get("_preface") or second.get("_preface"):
        return False
    family = DELIVERY_FAMILY.get(a)
    return family is not None and family == DELIVERY_FAMILY.get(b)


def merge_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Одна запись для голоса: текст подряд, подача каждой части словами по порядку."""
    if len(parts) == 1:
        return parts[0]
    plan = " Then ".join(
        f'"{p["text"]}" — {p["emotion"]} at intensity {p["intensity"]:.2f}: '
        f"{VOICE_DELIVERY.get(p['voice_style'], p['voice_style'])}."
        for p in parts
    )
    return {
        **parts[0],
        "text": " ".join(p["text"] for p in parts),
        "intensity": max(p["intensity"] for p in parts),
        "_style": "One continuous natural utterance; the delivery changes mid-way. " + plan,
        "_merged": [p["id"] for p in parts],
    }


def voice_style(emotion: str, intensity: float, delivery: str) -> str:
    """Пометка для облачного голоса: настроение, сила и подача словами."""
    described = VOICE_DELIVERY.get(delivery, delivery)
    return f"Speak with {emotion} at intensity {intensity:.2f} (0 — barely, 1 — very strongly): {described}."


class SpeechOutput:
    def __init__(self, services: Any = None, session_id: str | None = None) -> None:
        self.services = services
        self.session_id = session_id or uuid.uuid4().hex
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None
        self.queue: deque[tuple[int, str, dict[str, Any]]] = deque()
        self.enabled = self.armed = False
        # Response identity is independent of the audio attempt. Device/provider
        # failure or a voice change must not invalidate an in-progress text reply.
        self.response_id: str | None = None
        self.generation = 0
        self.audio_generation = 0
        self.audio_response_id: str | None = None
        self.minimum_generation = 0
        self.offered: dict[str, dict[str, Any]] = {}
        self.last_contact = time.monotonic()
        self.state = "off"
        self.mode = "off"
        self.error: str | None = None
        self.last: dict[str, Any] | None = None
        self.sync: dict[str, Any] | None = None
        self.sync_at = 0.0
        self.preview_mode = False
        self.high_water = self.backpressure_waits = 0
        self.delivery = delivery_settings({})
        self.bank = ReactionBank()
        self.pending: deque[tuple[int, str, dict[str, Any]]] = deque()
        self.finished = False
        self.began_at = 0.0
        self.reaction: dict[str, Any] | None = None
        self.reaction_used = False
        self.local_provider = True
        if services is not None:
            saved = services.call("data.get", {"path": "delivery.json"}).get("value")
            if saved is not None:
                self.delivery = delivery_settings(saved)
            recent = services.call("data.get", {"path": "reaction_history.json"}).get("value")
            self.bank = ReactionBank(recent)

    def catalog(self, refresh: bool = False) -> dict[str, Any]:
        if self.services is None:
            return {"available": False}
        try:
            return {"available": True, **self.services.call("speech.catalog", {"refresh": refresh})}
        except (NotImplementedError, RuntimeError, OSError):
            return {"available": False}

    def _send_fence(self, generation: int, identifier: str | None) -> None:
        if self.services is None:
            return
        try:
            self.services.call(
                "speech.advance", {"generation": generation, "response_id": identifier}
            )
        except (RuntimeError, PermissionError, OSError):
            with self.lock:
                if generation == self.audio_generation and not self.closed.is_set():
                    raise

    def _audio_fence(
        self,
        identifier: str | None,
        pause: bool = False,
        future_only: bool = False,
        expected_generation: int | None = None,
        error: str | None = None,
    ) -> int:
        with self.changed:
            if expected_generation is not None and expected_generation != self.audio_generation:
                return self.audio_generation
            if error is not None:
                self.error = error
                self.mode, self.state = "text", "error"
            notify = self.services is not None and (self.armed or bool(self.thread))
            self.audio_generation += 1
            generation = self.audio_generation
            self.audio_response_id = identifier
            self.queue.clear()
            self.pending.clear()
            self.finished = False
            self.reaction = None
            self.reaction_used = False
            self.offered.clear()
            self.preview_mode = False
            self.last = None
            self.sync = None
            if pause:
                self.armed = False
            if future_only:
                self.minimum_generation = self.generation + 1
            self.changed.notify_all()
        if notify:
            self._send_fence(generation, identifier)
        return generation

    def current(self, response_id: str, generation: int) -> bool:
        with self.lock:
            return self.response_id == response_id and self.generation == generation

    def _audio_current(self, identifier: str, generation: int) -> bool:
        with self.lock:
            return self.audio_response_id == identifier and self.audio_generation == generation

    def _degrade(self, error: BaseException | str, expected_generation: int | None = None) -> None:
        code = SAFE_ERROR.search(str(error))
        try:
            self._audio_fence(
                None,
                pause=True,
                future_only=True,
                expected_generation=expected_generation,
                error=code[0] if code else "speech.provider_failed",
            )
        except (RuntimeError, PermissionError, OSError):
            pass

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        self.pause()
        result = self.services.call("speech.configure", {"settings": settings})
        with self.lock:
            self.enabled = result["settings"]["enabled"]
            self.local_provider = result["settings"].get("provider", "system") == "system"
        self._arm()
        return {**result, **self.view()}

    def _arm(self) -> None:
        with self.lock:
            self.armed = self.enabled
            self.last_contact = time.monotonic()
            self.error = None
            self.mode = "speech" if self.enabled else "off"
            self.state = "idle" if self.enabled else "off"
        self._audio_fence(None, future_only=True)
        self._ensure()

    def arm(self) -> dict[str, Any]:
        catalog = self.catalog()
        with self.lock:
            self.enabled = catalog.get("settings", {}).get("enabled", False)
            self.local_provider = catalog.get("settings", {}).get("provider", "system") == "system"
        self._arm()
        return self.view()

    def pause(self) -> dict[str, Any]:
        self._audio_fence(None, pause=True, future_only=True)
        with self.lock:
            self.mode, self.state = "text" if self.enabled else "off", "off"
        return self.view()

    def preview(self) -> dict[str, Any]:
        if self.services is None:
            raise RuntimeError("speech is unavailable")
        with self.lock:
            self.armed = True
            self.last_contact = time.monotonic()
            self.error = None
            self.mode = "speech"
        identifier = "preview-" + uuid.uuid4().hex
        generation = self._audio_fence(identifier, future_only=True)
        with self.changed:
            self.preview_mode = True
            self.queue.append(
                (
                    generation,
                    identifier,
                    {
                        "id": "preview",
                        "text": "Проверка голоса. Hello!",
                        "emotion": "neutral",
                        "intensity": 0,
                        "voice_style": "neutral",
                    },
                )
            )
            self.changed.notify_all()
        self._ensure()
        return self.view()

    def touch(self) -> None:
        self.last_contact = time.monotonic()

    def _ensure(self) -> None:
        with self.lock:
            if self.armed and not (self.thread and self.thread.is_alive()):
                self.closed = threading.Event()
                self.thread = threading.Thread(
                    target=self._run, name="character-speech", daemon=True
                )
                self.thread.start()

    def begin(self, response_id: str) -> int:
        with self.lock:
            self.generation += 1
            self.response_id = response_id
            generation = self.generation
        self._audio_fence(response_id)
        self.began_at = time.monotonic()
        return generation

    def offer(
        self,
        response_id: str,
        record: dict[str, Any],
        generation: int | None = None,
        on_wait: Callable[[], None] | None = None,
    ) -> bool:
        epoch = self.generation if generation is None else generation
        keepalive = time.monotonic() + 1
        while True:
            with self.changed:
                if (
                    not self.armed
                    or not self.enabled
                    or not self.current(response_id, epoch)
                    or epoch < self.minimum_generation
                ):
                    return False
                previous = self.offered.get(record["id"])
                if previous is not None:
                    if previous != record:
                        raise ValueError("speech.segment_conflict")
                    return True
                if len(self.offered) >= 8:
                    raise ValueError("speech.segment_limit")
                if self.delivery["mode"] == "complete" or len(self.queue) < SEGMENT_QUEUE:
                    self.offered[record["id"]] = copy.deepcopy(record)
                    target = self.pending if self.delivery["mode"] == "complete" else self.queue
                    target.append((self.audio_generation, response_id, copy.deepcopy(record)))
                    self.high_water = max(self.high_water, len(self.queue))
                    self.changed.notify_all()
                    return True
                self.backpressure_waits += 1
                self.changed.wait(0.05)
            if on_wait is not None and time.monotonic() >= keepalive:
                on_wait()
                keepalive = time.monotonic() + 1

    def policy(self, proposed: dict[str, Any] | None = None) -> dict[str, Any]:
        if proposed is not None:
            result = delivery_settings({**self.delivery, **proposed})
            self.pause()
            if self.services is not None:
                self.services.call("data.set", {"path": "delivery.json", "value": result})
            with self.lock:
                self.delivery = result
        return {"settings": copy.deepcopy(self.delivery), "reaction_provider": "system"}

    def react(self, response_id: str, record: dict[str, Any], generation: int) -> None:
        with self.changed:
            if (
                not self.current(response_id, generation)
                or not self.delivery["reactions"]
                or not self.local_provider
                or not self.enabled
                or not self.armed
                or generation < self.minimum_generation
            ):
                return
            chosen = self.bank.choose(
                record.get("reaction", "none"), record.get("reaction_language", "")
            )
            if chosen is not None and self.reaction is None:
                self.reaction = {
                    "id": "~reaction",
                    "text": chosen[1],
                    "emotion": record["emotion"],
                    "expression": record.get("expression", "neutral"),
                    "intensity": record["intensity"],
                    "voice_style": "neutral",
                    "animation": "idle",
                    "_preface": True,
                    "_reaction_id": chosen[0],
                }
                self.changed.notify_all()

    def finish(self, response_id: str, generation: int) -> None:
        with self.changed:
            if not self.current(response_id, generation):
                return
            self.finished = True
            self.changed.notify_all()

    def _take(self) -> tuple[int, str, dict[str, Any]]:
        """Первая часть очереди и следующие за ней части той же подачи — одним запросом."""
        generation, identifier, record = self.queue.popleft()
        parts = [record]
        while (
            self.queue
            and self.queue[0][:2] == (generation, identifier)
            and same_delivery(parts[-1], self.queue[0][2])
        ):
            parts.append(self.queue.popleft()[2])
        return generation, identifier, merge_parts(parts)

    def _next(self) -> tuple[int, str, dict[str, Any]] | None:
        if not self.armed:
            return None
        if self.finished:
            while self.pending and len(self.queue) < SEGMENT_QUEUE:
                self.queue.append(self.pending.popleft())
        ready = self.finished or len(self.queue) >= self.delivery["buffer"]
        if self.queue and (ready or self.preview_mode):
            self.reaction_used = True  # A fast answer suppresses the waiting phrase.
            return self._take()
        # This is a configured grace period, not a measured response latency.
        if (
            self.reaction
            and not self.reaction_used
            and time.monotonic() - self.began_at >= 0.6
            and self.audio_response_id
        ):
            self.reaction_used = True
            self.bank.used(self.reaction["_reaction_id"])
            return self.audio_generation, self.audio_response_id, self.reaction
        return None

    def cancel(self, pause: bool = False) -> dict[str, Any]:
        with self.lock:
            self.generation += 1
            self.response_id = None
        self._audio_fence(None, pause)
        return self.view()

    def close(self) -> None:
        self.cancel(pause=True)
        self.closed.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)

    def view(self) -> dict[str, Any]:
        with self.lock:
            sync = copy.deepcopy(self.sync)
            if sync is not None:
                age = (time.monotonic() - self.sync_at) * 1000
                for window in sync["playback"].get("windows", []):
                    window["start_ms"] -= age
                    window["end_ms"] -= age
                sync["playback"]["windows"] = [
                    w for w in sync["playback"].get("windows", []) if w["end_ms"] > 0
                ]
            return copy.deepcopy(
                {
                    "sync": sync,
                    "delivery": copy.deepcopy(self.delivery),
                    "held": len(self.pending),
                    "session_id": self.session_id,
                    "enabled": self.enabled,
                    "armed": self.armed,
                    "generation": self.generation,
                    "response_id": self.response_id,
                    "audio_generation": self.audio_generation,
                    "mode": self.mode,
                    "state": self.state,
                    "error": self.error,
                    "last": self.last,
                    "queued": len(self.queue),
                    "queue_capacity": SEGMENT_QUEUE,
                    "high_water": self.high_water,
                    "backpressure_waits": self.backpressure_waits,
                }
            )

    def _start(self, generation: int, identifier: str, record: dict[str, Any]) -> dict[str, Any]:
        request = {
            "text": record["text"],
            "language": "auto",
            "emotion": record["emotion"],
            "intensity": record["intensity"],
            "style": record.get("_style")
            or voice_style(record["emotion"], record["intensity"], record["voice_style"]),
        }
        result: dict[str, Any] = self.services.call(
            "speech.start",
            {
                "request": request,
                "generation": generation,
                "response_id": identifier,
                "segment_id": record["id"],
                "preview": record["id"] == "preview",
                "preface": bool(record.get("_preface")),
            },
        )
        return result

    def _prefetch(self) -> tuple[str, int] | None:
        """Следующая готовая фраза того же ответа — в синтез заранее; реплика-ожидание не берётся."""
        with self.changed:
            if not self.queue or self.preview_mode:
                return None
            if not (self.finished or len(self.queue) >= self.delivery["buffer"]):
                return None
            generation, identifier, record = self.queue[0]
            if record.get("_preface") or not self._audio_current(identifier, generation):
                return None
            generation, identifier, record = self._take()
            self.changed.notify_all()
        try:
            return self._start(generation, identifier, record)["job_id"], generation
        except RuntimeError as error:
            if str(error) != "speech.busy":
                raise
            with self.changed:
                self.queue.appendleft((generation, identifier, record))
            return None

    def _run(self) -> None:
        job_id: str | None = None
        job_generation = -1
        failed = False
        # Следующая фраза синтезируется, пока звучит текущая (служба играет её после).
        next_job: tuple[str, int] | None = None
        try:
            while not self.closed.is_set():
                operation_generation = self.audio_generation
                try:
                    if self.armed and time.monotonic() - self.last_contact > 5:
                        self.pause()
                    if job_id:
                        operation_generation = job_generation
                        requested_at = time.monotonic()
                        packet = self.services.call("speech.status", {"job_id": job_id})["job"]
                        if packet:
                            with self.lock:
                                current = job_generation == self.audio_generation
                                if current:
                                    self.state = packet["state"]
                                    record = (
                                        self.reaction
                                        if packet.get("segment_id") == "~reaction"
                                        else self.offered.get(packet.get("segment_id", ""))
                                    )
                                    self.sync = None
                                    if (
                                        record is not None
                                        and packet.get("response_id") == self.audio_response_id
                                    ):
                                        self.sync = {
                                            "generation": job_generation,
                                            "response_id": packet["response_id"],
                                            "segment_id": record["id"],
                                            "emotion": record["emotion"],
                                            # Лицо набора (anger → furious); оболочка берёт его.
                                            "expression": record.get("expression", "neutral"),
                                            "eyes": record.get("eyes"),
                                            "mouth": record.get("mouth"),
                                            "moves": record.get("moves", []),
                                            "animation": record.get("animation", "idle"),
                                            "reaction_label": record.get("_reaction_id", "").split(
                                                ":"
                                            )[0],
                                            "playback": copy.deepcopy(
                                                packet.get(
                                                    "playback",
                                                    {"clock": "unavailable", "windows": []},
                                                )
                                            ),
                                        }
                                        self.sync_at = requested_at
                                    metadata = packet["metadata"]
                                    self.last = {
                                        k: copy.deepcopy(v)
                                        for k, v in metadata.items()
                                        if k != "alignment"
                                    }
                                    self.last["timing_count"] = len(
                                        (metadata.get("alignment") or {}).get("starts", [])
                                    )
                            if current and packet["state"] == "error" and not failed:
                                failed = True
                                self._degrade(
                                    packet.get("error") or "speech.failed", job_generation
                                )
                            if (
                                not packet.get("settled")
                                and next_job is None
                                and packet["state"] == "playing"
                                and job_generation == self.audio_generation
                            ):
                                next_job = self._prefetch()
                            if packet.get("settled"):
                                job_id = None
                                if next_job is not None:
                                    job_id, job_generation = next_job
                                    next_job, failed = None, False
                                with self.lock:
                                    if (
                                        self.preview_mode
                                        and job_generation == self.audio_generation
                                    ):
                                        self.preview_mode = False
                                        self.armed = self.enabled
                                        self.mode = "speech" if self.enabled else "off"
                        else:
                            job_id = None
                    if not job_id:
                        next_job = None
                        with self.changed:
                            item = self._next()
                            self.changed.notify_all()
                            if item is None and self.error is None:
                                self.state = "idle" if self.armed else "off"
                        if item:
                            generation, identifier, record = item
                            operation_generation = generation
                            if not self._audio_current(identifier, generation):
                                continue
                            if record.get("_preface"):
                                self.services.call(
                                    "data.set",
                                    {"path": "reaction_history.json", "value": self.bank.recent},
                                )
                            try:
                                result = self._start(generation, identifier, record)
                                job_id, job_generation, failed = result["job_id"], generation, False
                            except (RuntimeError, PermissionError, OSError):
                                if self._audio_current(identifier, generation):
                                    raise
                except Exception as error:
                    self._degrade(error, operation_generation)
                    job_id = None
                with self.changed:
                    self.changed.wait(0.02)
        finally:
            if self.services is not None:
                try:
                    self.services.call("speech.cancel", {})
                except (OSError, RuntimeError, PermissionError):
                    pass
            with self.changed:
                self.armed = False
                self.queue.clear()
                self.pending.clear()
                self.changed.notify_all()
