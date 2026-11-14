"""Обработчики anthropic: формат запросов и ответов, включая поток."""

import json
import uuid
from typing import Any

from ramvi_engine.server.validation import _maybe_json, anthropic_messages, anthropic_tools


def anthropic_routes(app: Any) -> Any:
    """Связать обработчик Anthropic с одним приложением."""

    def anthropic(self: Any, req: Any) -> Any:
        """Обслужить Anthropic messages, включая поток и инструменты."""
        choice = (req.get("tool_choice") or {}).get("type", "auto")
        if choice not in ("auto", "none"):
            raise NotImplementedError(
                "принудительный/конкретный tool_choice пока не поддерживается"
            )
        tools = anthropic_tools(req.get("tools")) if choice != "none" else None
        think = (req.get("thinking") or {}).get("type") == "enabled"
        ids = app.build_ids(anthropic_messages(req), tools, think)
        gen = app.run(
            ids,
            int(req.get("max_tokens", 1024)),
            float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0),
            float(req.get("top_p", 1.0) or 1.0),
            None,
            req.get("stop_sequences"),
            think,
        )
        mid = "msg_" + uuid.uuid4().hex[:24]
        sr = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "stop_sequence": "stop_sequence",
        }

        def blocks(fin: Any) -> Any:
            b = []
            if fin["reasoning"]:
                b.append({"type": "thinking", "thinking": fin["reasoning"], "signature": ""})
            if fin["text"]:
                b.append({"type": "text", "text": fin["text"]})
            for c in fin["tool_calls"]:
                b.append(
                    {
                        "type": "tool_use",
                        "id": "toolu_" + uuid.uuid4().hex[:24],
                        "name": c["name"],
                        "input": _maybe_json(c["arguments"])
                        if isinstance(c["arguments"], str)
                        else c["arguments"],
                    }
                )
            return b

        if not req.get("stream"):
            fin: Any = None
            for kind, v in gen:
                if kind == "end":
                    fin = v
            return self._json(
                {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": app.name,
                    "content": blocks(fin),
                    "stop_reason": sr[fin["finish"]],
                    "stop_sequence": fin.get("stop_sequence"),
                    "usage": {"input_tokens": fin["prompt_n"], "output_tokens": fin["gen_n"]},
                }
            )
        self._start_stream("text/event-stream", "anthropic")
        self._sse(
            {
                "type": "message_start",
                "message": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": app.name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": len(ids), "output_tokens": 0},
                },
            },
            "message_start",
        )
        idx, cur = (0, None)

        def open_block(kind: Any) -> Any:
            nonlocal cur
            if cur == kind:
                return
            close_block()
            cur = kind
            b = (
                {"type": "thinking", "thinking": "", "signature": ""}
                if kind == "thinking"
                else {"type": "text", "text": ""}
            )
            self._sse(
                {"type": "content_block_start", "index": idx, "content_block": b},
                "content_block_start",
            )

        def close_block() -> Any:
            nonlocal cur, idx
            if cur is not None:
                self._sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                idx += 1
                cur = None

        for kind, v in gen:
            if kind == "reasoning":
                open_block("thinking")
                self._sse(
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "thinking_delta", "thinking": v},
                    },
                    "content_block_delta",
                )
            elif kind == "text":
                open_block("text")
                self._sse(
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": v},
                    },
                    "content_block_delta",
                )
            else:
                close_block()
                for c in v["tool_calls"]:
                    args = (
                        c["arguments"]
                        if not isinstance(c["arguments"], str)
                        else _maybe_json(c["arguments"])
                    )
                    self._sse(
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": "toolu_" + uuid.uuid4().hex[:24],
                                "name": c["name"],
                                "input": {},
                            },
                        },
                        "content_block_start",
                    )
                    self._sse(
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(args, ensure_ascii=False),
                            },
                        },
                        "content_block_delta",
                    )
                    self._sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                    idx += 1
                self._sse(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": sr[v["finish"]],
                            "stop_sequence": v.get("stop_sequence"),
                        },
                        "usage": {"output_tokens": v["gen_n"]},
                    },
                    "message_delta",
                )
                self._sse({"type": "message_stop"}, "message_stop")

    return anthropic
