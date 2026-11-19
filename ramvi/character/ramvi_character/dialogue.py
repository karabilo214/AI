"""Headless text dialogue using only scoped data and LLM services."""

import copy
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from ramvi_character.context import Budget, ContextOrchestrator, encode, text, tokenizer_counter
from ramvi_character.response import ResponsePlanner, plain_segments


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

    def view(self, response_id: str | None = None, cursor: int = 0) -> dict[str, Any]:
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("invalid event cursor")
        with self.lock:
            job = self.job
            result: dict[str, Any] = {"history": copy.deepcopy(self.history), "job": None}
            if job and (response_id is None or job["id"] == response_id):
                result["job"] = {
                    key: copy.deepcopy(value) for key, value in job.items() if key != "events"
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

    def start(self, user_input: str, model: str) -> dict[str, Any]:
        user_input, model = text(user_input, 4096), text(model, 255)
        if not user_input.strip() or not model.strip():
            raise ValueError("input and model required")
        if self.services is None:
            raise RuntimeError("LLM services unavailable")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("previous response is still stopping")
            self.cancelled = threading.Event()
            identifier = uuid.uuid4().hex
            self.owner._speech.begin(identifier)
            self.job = {
                "id": identifier,
                "state": "running",
                "events": [],
                "error": None,
                "fallback": False,
            }
            snapshot = {
                "personality": self.owner._personality.view(),
                "state": self.owner._emotions.view(),
                "task": self.owner._task,
                "history": copy.deepcopy(self.history),
            }
            self.thread = threading.Thread(
                target=self._run,
                args=(identifier, user_input, model, snapshot),
                name="character-response",
                daemon=True,
            )
            self.thread.start()
            return {"response_id": identifier}

    def _check(self, identifier: str, revision: int) -> None:
        if (
            self.cancelled.is_set()
            or self.job is None
            or self.job["id"] != identifier
            or self.owner._personality.view()["policy_revision"] != revision
        ):
            raise InterruptedError("response cancelled")

    def _emit(self, identifier: str, revision: int, records: list[dict[str, Any]]) -> None:
        with self.lock:
            self._check(identifier, revision)
            assert self.job is not None
            for record in records:
                if record["type"] != "end":
                    self.job["events"].append({"response_id": identifier, **record})
                    if record["type"] == "segment":
                        self.owner._speech.offer(identifier, record)

    def _request(self, request: dict[str, Any], streaming: bool) -> Iterator[str]:
        if not streaming:
            result = self.services.call("llm.complete", {"request": request, "timeout": 30})
            choice = result["choices"][0]
            if choice.get("finish_reason") not in {"stop", "end_turn"}:
                raise ValueError("incomplete provider response")
            yield choice["message"]["content"]
            return
        stream_id = self.services.call("llm.stream", {"request": request, "timeout": 30})[
            "stream_id"
        ]
        with self.lock:
            self.stream_id = stream_id
        finished = False
        deadline = time.monotonic() + 180
        try:
            while not finished:
                if self.cancelled.is_set():
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

    def _run(self, identifier: str, user_input: str, model: str, snapshot: dict[str, Any]) -> None:
        revision = snapshot["personality"]["policy_revision"]
        accepted: list[dict[str, Any]] = []
        candidates = []
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
                if row["status"] == "completed":
                    recent.extend(
                        [
                            {"id": row["id"] + "u", "role": "user", "text": row["user"]},
                            {"id": row["id"] + "a", "role": "assistant", "text": row["assistant"]},
                        ]
                    )
            emotions = self.owner._emotions.identifiers()
            planner = ResponsePlanner(
                emotions=emotions,
                sources=["task", "current", *["recent:" + r["id"] for r in recent]],
            )
            context = ContextOrchestrator(
                self.owner._scope, self.counter, system_prompt=planner.prompt(streaming)
            )
            prepared = context.build(
                task=snapshot["task"] or "Continue the conversation and help the owner.",
                user_input=user_input,
                personality=snapshot["personality"],
                state=snapshot["state"],
                recent=recent,
                budget=Budget(output=1536),
            )
            # Evidence must refer only to sources that survived trimming.
            planner = ResponsePlanner(emotions=emotions, sources=list(prepared.sources))
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
            with self.lock:
                self._check(identifier, revision)
                assert self.job is not None
                self.job["events"].append(
                    {
                        "type": "end",
                        "response_id": identifier,
                        "memory_candidates": [{**c, "status": "candidate"} for c in candidates],
                    }
                )
                status = "completed"
        except InterruptedError:
            status = "cancelled"
        except Exception as error:
            with self.lock:
                if self.job and self.job["id"] == identifier:
                    self.job["error"] = type(error).__name__
        finally:
            with self.lock:
                self.connected = False
                if self.cancelled.is_set():
                    status = "cancelled"
                if self.job and self.job["id"] == identifier:
                    self.job["state"] = status
                self.history.append(
                    {
                        "id": identifier,
                        "user": user_input,
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
