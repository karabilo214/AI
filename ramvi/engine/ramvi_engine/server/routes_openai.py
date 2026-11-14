"""Обработчики openai: формат запросов и ответов, включая поток."""

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from ramvi_engine.server.validation import _schema_from_openai, openai_messages


def openai_routes(app: Any) -> Any:
    """Связать обработчики OpenAI с одним приложением."""

    def openai_chat(self: Any, req: Any) -> Any:
        """Обслужить OpenAI chat/completions, включая поток и инструменты."""
        choice = req.get("tool_choice", "auto")
        if choice not in ("auto", "none"):
            raise NotImplementedError(
                "принудительный/конкретный tool_choice пока не поддерживается"
            )
        tools = req.get("tools") if choice != "none" else None
        think = req.get("reasoning_effort") not in (None, "", "none") or bool(req.get("think"))
        schema = _schema_from_openai(req.get("response_format"))
        if schema is not None:
            think = False
        ids = app.build_ids(openai_messages(req.get("messages", [])), tools, think)
        stop = req.get("stop")
        stop = [stop] if isinstance(stop, str) else stop or []
        gen = app.run(
            ids,
            int(req.get("max_completion_tokens", req.get("max_tokens", 1024))),
            float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0),
            float(req.get("top_p", 1.0) or 1.0),
            req.get("seed"),
            stop,
            think,
            app.constraint(schema),
        )
        cid, created = ("chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()))
        fr = {
            "stop": "stop",
            "stop_sequence": "stop",
            "length": "length",
            "tool_calls": "tool_calls",
        }

        def calls_of(fin: Any) -> Any:
            return [
                {
                    "id": "call_" + uuid.uuid4().hex[:20],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c["arguments"], ensure_ascii=False)
                        if not isinstance(c["arguments"], str)
                        else c["arguments"],
                    },
                }
                for c in fin["tool_calls"]
            ]

        usage: Callable[..., Any] = lambda f: {  # noqa: E731 - перенесённая функция протокола
            "prompt_tokens": f["prompt_n"],
            "completion_tokens": f["gen_n"],
            "total_tokens": f["prompt_n"] + f["gen_n"],
        }
        if not req.get("stream"):
            fin: Any = None
            for kind, v in gen:
                if kind == "end":
                    fin = v
            msg = {
                "role": "assistant",
                "content": fin["text"] if fin["text"] or not fin["tool_calls"] else None,
            }
            if fin["reasoning"]:
                msg["reasoning_content"] = fin["reasoning"]
            if fin["tool_calls"]:
                msg["tool_calls"] = calls_of(fin)
            return self._json(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": app.name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": fr[fin["finish"]]}],
                    "usage": usage(fin),
                }
            )
        self._start_stream("text/event-stream")
        chunk: Callable[..., Any] = lambda delta, finish=None: {  # noqa: E731 - перенесённая функция протокола
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": app.name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        self._sse(chunk({"role": "assistant", "content": ""}))
        for kind, v in gen:
            if kind == "text":
                self._sse(chunk({"content": v}))
            elif kind == "reasoning":
                self._sse(chunk({"reasoning_content": v}))
            else:
                if v["tool_calls"]:
                    self._sse(
                        chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls_of(v))]})
                    )
                self._sse(chunk({}, fr[v["finish"]]))
                if (req.get("stream_options") or {}).get("include_usage"):
                    self._sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": app.name,
                            "choices": [],
                            "usage": usage(v),
                        }
                    )
        self._sse("[DONE]")

    def openai_completion(self: Any, req: Any) -> Any:
        """Обслужить OpenAI completions для одного запроса."""
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            if prompt and all(
                isinstance(t, int) and (not isinstance(t, bool)) and (t >= 0) for t in prompt
            ):
                ids = prompt
            elif len(prompt) == 1 and isinstance(prompt[0], str):
                ids = app.tk.encode(prompt[0], add_special_tokens=False)
            else:
                raise ValueError("поддерживается один prompt: строка или список ID токенов")
        elif isinstance(prompt, str):
            ids = app.tk.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError("неверная форма prompt")
        stop = req.get("stop")
        stop = [stop] if isinstance(stop, str) else stop or []
        gen = app.run(
            ids,
            int(req.get("max_tokens", 256)),
            float(req.get("temperature", 1.0)),
            float(req.get("top_p", 1.0)),
            req.get("seed"),
            stop,
        )
        if req.get("stream"):
            cid = "cmpl-" + uuid.uuid4().hex[:24]
            self._start_stream("text/event-stream")
            for kind, value in gen:
                if kind == "reasoning":
                    continue
                finish = None
                text = value if kind == "text" else ""
                if kind == "end":
                    finish = "length" if value["finish"] == "length" else "stop"
                self._sse(
                    {
                        "id": cid,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": app.name,
                        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                    }
                )
            self._sse("[DONE]")
            return
        fin: Any = None
        for kind, value in gen:
            if kind == "end":
                fin = value
        return self._json(
            {
                "id": "cmpl-" + uuid.uuid4().hex[:24],
                "object": "text_completion",
                "created": int(time.time()),
                "model": app.name,
                "choices": [
                    {
                        "index": 0,
                        "text": fin["text"],
                        "finish_reason": "length" if fin["finish"] == "length" else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": fin["prompt_n"],
                    "completion_tokens": fin["gen_n"],
                    "total_tokens": fin["prompt_n"] + fin["gen_n"],
                },
            }
        )

    return (openai_chat, openai_completion)
