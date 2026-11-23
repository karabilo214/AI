"""Headless text dialogue using only scoped data and LLM services."""

import copy
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from ramvi_character.context import (
    Budget,
    ContextOrchestrator,
    ContextOverflow,
    encode,
    text,
    tokenizer_counter,
)
from ramvi_character.pack import GESTURES, load_pack
from ramvi_character.recall import Recall
from ramvi_character.response import ResponsePlanner, plain_segments, strip_service_phrases

CHOOSABLE = {"nod", "shake", *GESTURES}


# Файлы к реплике персонажу (B.15). Число и размер задаёт пользователь в настройках приложения,
# их проверяет оболочка; модуль проверяет только вид. Файл идёт в текущий ответ целиком, без
# обрезки; не помещается в запрос — понятная ошибка с числом токенов (job["overflow"]).
FILES_ONLY_INPUT = "The owner attached files without a message. Look at them and react."


def check_files(files: Any) -> list[dict[str, str]]:
    """Только текст и имя без пути; иначе ValueError."""
    if files is None:
        return []
    if not isinstance(files, list):
        raise ValueError("invalid files")
    result = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"name", "text"}:
            raise ValueError("invalid file")
        name, body = item["name"], item["text"]
        if not isinstance(name, str) or not isinstance(body, str) or "\x00" in body:
            raise ValueError("invalid file")
        name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or len(name) > 255 or any(c in name for c in '<>"\n'):
            raise ValueError("invalid file name")
        result.append({"name": name, "text": body})
    return result


def with_files(user_input: str, files: list[dict[str, str]]) -> str:
    """Текст для модели: слова владельца и файлы — как в чате, тегами <file>."""
    blocks = "".join(f'\n\n<file name="{f["name"]}">\n{f["text"]}\n</file>' for f in files)
    return ((user_input.strip() or (FILES_ONLY_INPUT if files else "")) + blocks).strip()


def history_text(user_input: str, files: list[dict[str, str]]) -> str:
    """В историю — слова владельца и имена файлов; содержимое файлов не хранится."""
    names = "\n".join("[file: " + f["name"] + "]" for f in files)
    combined = (user_input.strip() + "\n" + names).strip() if names else user_input
    return combined if len(combined.encode("utf-8")) <= 4096 else user_input


# Сильная эмоция без выбранного движения: персонаж реагирует телом сам.
AUTO_GESTURES = {
    "joy": "jump",
    "surprise": "jump",
    "anger": "shout",
    "frustration": "turn_away",
    "confidence": "nod",
}
AUTO_INTENSITY = 0.6
# Подача голоса для части ответа (9.11): уходит в облачный голос подсказкой стиля.
VOICE_STYLES = [
    "excited",
    "intriguing",
    "calm",
    "warm",
    "playful",
    "sarcastic",
    "irritated",
    "shouting",
    "whisper",
    "sad",
    "thoughtful",
]


class Dialogue:
    def __init__(self, owner: Any, services: Any = None) -> None:
        self.owner, self.services = owner, services
        self.lock = threading.RLock()
        self.history: list[dict[str, Any]] = []
        self.job: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.cancelled = threading.Event()
        self.stream_id: str | None = None
        self.counter: Any = None
        self.connected = False
        if services is not None:
            saved = services.call("data.get", {"path": "dialogue.json"}).get("value")
            if saved:
                if (
                    not isinstance(saved, dict)
                    or saved.get("schema") != "ramvi.character.dialogue/1"
                ):
                    raise ValueError("invalid dialogue history")
                rows = saved.get("turns")
                if not isinstance(rows, list) or len(rows) > 16:
                    raise ValueError("invalid history turns")
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {
                        "id",
                        "user",
                        "assistant",
                        "model",
                        "status",
                    }:
                        raise ValueError("invalid history turn")
                    for key, limit in [
                        ("id", 64),
                        ("user", 4096),
                        ("assistant", 16384),
                        ("model", 255),
                    ]:
                        text(row[key], limit)
                    if row["status"] not in {"completed", "cancelled", "error"}:
                        raise ValueError("invalid history status")
                if len(encode(rows).encode()) > 32_000:
                    raise ValueError("oversized history")
                self.history = copy.deepcopy(rows)

    def _save(self) -> None:
        while len(self.history) > 16 or len(encode(self.history).encode()) > 32_000:
            self.history.pop(0)
        if self.services is not None:
            self.services.call(
                "data.set",
                {
                    "path": "dialogue.json",
                    "value": {"schema": "ramvi.character.dialogue/1", "turns": self.history},
                },
            )

    def forget(self, identifiers: list[str] | None) -> list[str]:
        """Delete history turns (all when None); returns removed ids. A running response is
        stopped first, and the finished job's events (its text) are dropped too."""
        self.cancel(wait=True)
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():  # its turn would be appended after the deletion
                raise RuntimeError("response is still stopping")
        with self.lock:
            removed = [
                row["id"] for row in self.history if identifiers is None or row["id"] in identifiers
            ]
            if identifiers is not None and set(identifiers) - set(removed):
                raise ValueError("unknown history turn")
            self.history = [row for row in self.history if row["id"] not in removed]
            if self.job is not None and self.job["id"] in removed:
                self.job = None
            self._save()
        return removed

    def view(self, response_id: str | None = None, cursor: int = 0) -> dict[str, Any]:
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("invalid event cursor")
        with self.lock:
            job = self.job
            result: dict[str, Any] = {
                "session_id": self.owner._speech.session_id,
                "history": copy.deepcopy(self.history),
                "job": None,
            }
            if job and (response_id is None or job["id"] == response_id):
                result["job"] = {
                    key: copy.deepcopy(value)
                    for key, value in job.items()
                    if key not in {"events", "started_at"}
                }
                result["job"]["events"] = copy.deepcopy(job["events"][cursor:])
                result["job"]["cursor"] = len(job["events"])
                result["job"]["settled"] = not (self.thread and self.thread.is_alive())
            return result

    def cancel(self, wait: bool = False) -> None:
        self.owner._speech.cancel()
        with self.lock:
            self.cancelled.set()
            stream_id = self.stream_id
            if self.job and self.job["state"] == "running":
                self.job["state"] = "cancelled"
                self.job["events"] = []  # Never deliver queued directives after Stop.
        if stream_id and self.services:
            try:
                self.services.call("llm.cancel", {"stream_id": stream_id})
            except (OSError, RuntimeError, ValueError, PermissionError):
                pass
        if wait and self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def start(self, user_input: str, model: str, files: Any = None) -> dict[str, Any]:
        user_input, model = text(user_input, 4096), text(model, 255)
        attached = check_files(files)
        if not (user_input.strip() or attached) or not model.strip():
            raise ValueError("input and model required")
        if self.services is None:
            raise RuntimeError("LLM services unavailable")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("previous response is still stopping")
            self.cancelled = threading.Event()
            identifier = uuid.uuid4().hex
            generation = self.owner._speech.begin(identifier)
            self.job = {
                "id": identifier,
                "generation": generation,
                "session_id": self.owner._speech.session_id,
                "state": "running",
                "events": [],
                "error": None,
                "fallback": False,
                "started_at": time.monotonic(),
                "first_segment_ms": None,
            }
            snapshot = {
                "personality": self.owner._personality.view(),
                "state": self.owner._emotions.view(),
                "task": self.owner._task,
                "history": copy.deepcopy(self.history),
            }
            self.thread = threading.Thread(
                target=self._run,
                args=(identifier, user_input, model, snapshot, attached),
                name="character-response",
                daemon=True,
            )
            self.thread.start()
            return {
                "response_id": identifier,
                "generation": generation,
                "session_id": self.owner._speech.session_id,
            }

    def _check(self, identifier: str, revision: int) -> None:
        if (
            self.cancelled.is_set()
            or self.job is None
            or self.job["id"] != identifier
            or not self.owner._speech.current(identifier, self.job["generation"])
            or self.owner._personality.view()["policy_revision"] != revision
        ):
            raise InterruptedError("response cancelled")

    def _gestures(self) -> list[str]:
        """Движения, которые модель может выбрать: кивок, отрицание и живые жесты набора."""
        pack = getattr(self.owner, "_pack", None) or load_pack()
        return [n for n in pack["animations"] if n in CHOOSABLE]

    def _presentable(self, record: dict[str, Any]) -> dict[str, Any]:
        """Лицо набора по эмоции; сильная эмоция без движения — живой жест сам."""
        if record["type"] not in ("start", "segment"):
            return record
        result = {**record, "expression": self.owner._emotions.expression(record["emotion"])}
        if record["type"] == "segment" and record.get("animation", "none") == "none":
            gesture = AUTO_GESTURES.get(record["emotion"])
            if gesture in self._gestures() and float(record.get("intensity", 0)) >= AUTO_INTENSITY:
                result["animation"] = gesture
        return result

    def _emit(self, identifier: str, revision: int, records: list[dict[str, Any]]) -> None:
        records = [self._presentable(r) for r in records]
        for record in records:
            with self.lock:
                self._check(identifier, revision)
                assert self.job is not None
                generation = self.job["generation"]
                if record["type"] == "segment" and self.job["first_segment_ms"] is None:
                    self.job["first_segment_ms"] = (
                        time.monotonic() - self.job["started_at"]
                    ) * 1000
                if record["type"] != "end":
                    self.job["events"].append(
                        {
                            "response_id": identifier,
                            "generation": generation,
                            "session_id": self.owner._speech.session_id,
                            **record,
                        }
                    )
            # Never wait for playback/backpressure while holding the dialogue lock:
            # Stop must be able to invalidate this generation and wake this producer.
            if record["type"] == "start":
                self.owner._speech.react(identifier, record, generation)
            if record["type"] == "segment":
                self.owner._speech.offer(identifier, record, generation, self._keepalive)
            with self.lock:
                self._check(identifier, revision)

    def _keepalive(self) -> None:
        with self.lock:
            stream_id = self.stream_id
        if stream_id:
            self.services.call("llm.read", {"stream_id": stream_id, "timeout": 0, "max_chunks": 0})

    def _request(self, request: dict[str, Any], streaming: bool) -> Iterator[str]:
        stream_id = self.services.call(
            "llm.stream", {"request": request, "timeout": 30, "single": not streaming}
        )["stream_id"]
        with self.lock:
            self.stream_id = stream_id
        finished = False
        deadline = time.monotonic() + 180
        try:
            while not finished:
                with self.lock:
                    job = self.job
                    if (
                        self.cancelled.is_set()
                        or job is None
                        or not self.owner._speech.current(job["id"], job["generation"])
                    ):
                        raise InterruptedError("response cancelled")
                if time.monotonic() > deadline:
                    raise TimeoutError("response deadline")
                packet = self.services.call("llm.read", {"stream_id": stream_id, "timeout": 0.5})
                for chunk in packet["chunks"]:
                    if chunk.get("type") == "text":
                        yield chunk["text"]
                    elif chunk.get("type") == "finish" and chunk.get("finish_reason") not in {
                        "stop",
                        "end_turn",
                    }:
                        raise ValueError("incomplete provider response")
                finished = packet["done"]
        finally:
            self.services.call("llm.cancel", {"stream_id": stream_id})
            with self.lock:
                if self.stream_id == stream_id:
                    self.stream_id = None

    def _run(
        self,
        identifier: str,
        user_input: str,
        model: str,
        snapshot: dict[str, Any],
        files: list[dict[str, str]] | None = None,
    ) -> None:
        files = files or []
        revision = snapshot["personality"]["policy_revision"]
        accepted: list[dict[str, Any]] = []
        candidates = []
        feedback: list[dict[str, Any]] = []
        status = "error"
        try:
            caps = self.services.call("llm.capabilities", {})
            if not caps["available"]:
                raise RuntimeError("LLM provider unavailable")
            self.connected = True
            streaming = caps["capabilities"].get("streaming", False)
            if self.counter is None:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]

                path = os.environ.get("RAMVI_TOKENIZER")
                if not path:
                    raise RuntimeError("provider tokenizer is not configured")
                tokenizer = AutoTokenizer.from_pretrained(
                    path, local_files_only=True, trust_remote_code=False
                )
                self.counter = tokenizer_counter(tokenizer, path)
            recent = []
            for row in snapshot["history"]:
                # Old replies may hold service phrases the model would copy back.
                said = strip_service_phrases(row["assistant"])
                if row["status"] == "completed" and said:
                    recent.extend(
                        [
                            {"id": row["id"] + "u", "role": "user", "text": row["user"]},
                            {"id": row["id"] + "a", "role": "assistant", "text": said},
                        ]
                    )
            emotions = self.owner._emotions.identifiers()
            gestures = self._gestures()
            pack = getattr(self.owner, "_pack", None) or load_pack()
            face = {
                "eyes": list(pack["eyes"]),
                "mouths": [m for m in pack["mouths"] if not m.startswith("talk_")],
                "voice_styles": VOICE_STYLES,
            }
            planner = ResponsePlanner(
                emotions=emotions,
                animations=gestures,
                **face,
                sources=["task", "current", *["recent:" + r["id"] for r in recent]],
            )
            memory_source = None
            store = getattr(self.owner, "_memory", None)
            if store is not None:
                recall = Recall(store)
                scope, task = self.owner._scope, snapshot["task"]

                def memory_source(_text: str, limit: int) -> list[dict[str, Any]]:
                    # Indexed, bounded search; the orchestrator then fits the memory budget.
                    return recall.context_items(user_input, task, scope, min(limit, 16))

            context = ContextOrchestrator(
                self.owner._scope,
                self.counter,
                memory_source=memory_source,
                system_prompt=planner.prompt(streaming),
            )
            prepared = context.build(
                task=snapshot["task"] or "Talk with the owner naturally, staying in character.",
                # Файлы — только в этот ответ; память ищется и пишется по словам владельца.
                user_input=with_files(user_input, files),
                personality=snapshot["personality"],
                state=snapshot["state"],
                recent=recent,
                budget=Budget(output=1536),
            )
            # Evidence must refer only to sources that survived trimming.
            planner = ResponsePlanner(
                emotions=emotions, animations=gestures, sources=list(prepared.sources), **face
            )
            request = {
                "model": model,
                "ramvi_model_id": model,
                "ramvi_min_context": 8192,
                "messages": prepared.messages,
                "max_tokens": 1536,
                "temperature": 0,
                "enable_thinking": False,
            }
            self._check(identifier, revision)
            try:
                if streaming:
                    for chunk in self._request(request, True):
                        records = planner.feed(chunk)
                        self._emit(identifier, revision, records)
                    self._emit(identifier, revision, planner.finish())
                else:
                    content = "".join(self._request(request, False))
                    self._emit(identifier, revision, planner.complete(content))
                accepted = planner.segments
                # Candidates are published only after the entire stream finished successfully.
                # The end record is retained by the planner, never written as a fact.
                candidates = planner.candidates
                feedback = planner.feedback
            except (ValueError, json.JSONDecodeError) as error:
                accepted = planner.segments if streaming else []
                if accepted:
                    raise ValueError("invalid partial structured response") from error
                self._check(identifier, revision)
                fallback: dict[str, Any] = copy.deepcopy(request)
                fallback["messages"] = copy.deepcopy(prepared.messages)
                fallback["messages"][0]["content"] = fallback["messages"][0]["content"].replace(
                    context.system_prompt,
                    "Answer the owner in plain text. Respect the profile and owner boundaries. No JSON, tools, or memory writes.",
                )
                content = "".join(self._request(fallback, streaming))
                accepted = plain_segments(content)
                self._emit(identifier, revision, accepted)
                with self.lock:
                    assert self.job is not None
                    self.job["fallback"] = True
            self.owner._speech.finish(identifier, self.job["generation"] if self.job else -1)
            with self.lock:
                self._check(identifier, revision)
                assert self.job is not None
                self.job["events"].append(
                    {
                        "type": "end",
                        "session_id": self.owner._speech.session_id,
                        "generation": self.job["generation"],
                        "response_id": identifier,
                        "memory_candidates": [{**c, "status": "candidate"} for c in candidates],
                    }
                )
                self.job["memory_rejected"] = planner.rejected
                status = "completed"
            consolidator = getattr(self.owner, "_consolidator", None)
            if consolidator is not None and (candidates or feedback):
                # Only a fully completed response reaches memory, and off the speech path.
                owner_words = {"current": user_input}
                for row in snapshot["history"]:
                    owner_words["recent:" + row["id"] + "u"] = row["user"]
                # The owner's reaction concerns the previous completed reply.
                previous = [r["id"] for r in snapshot["history"] if r["status"] == "completed"]
                consolidator.submit(
                    identifier,
                    candidates,
                    owner_words,
                    feedback=feedback,
                    target=previous[-1] if previous else None,
                )
        except InterruptedError:
            status = "cancelled"
        except Exception as error:
            with self.lock:
                if self.job and self.job["id"] == identifier:
                    self.job["error"] = type(error).__name__
                    if isinstance(error, ContextOverflow):
                        # Оболочка показывает, сколько нужно и сколько есть, а не «ошибка ответа».
                        self.job["overflow"] = {
                            "section": error.section,
                            "needed": error.needed,
                            "available": error.available,
                        }
        finally:
            if status == "error":
                self.owner._speech.cancel()
            with self.lock:
                self.connected = False
                if self.cancelled.is_set():
                    status = "cancelled"
                if self.job and self.job["id"] == identifier:
                    self.job["state"] = status
                self.history.append(
                    {
                        "id": identifier,
                        "user": history_text(user_input, files),
                        "assistant": "\n".join(s["text"] for s in accepted),
                        "model": model,
                        "status": status,
                    }
                )
                try:
                    self._save()
                except Exception:
                    if self.job:
                        self.job["error"] = "history_save_failed"
