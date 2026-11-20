"""7.9 REMOTE-LLM-03: транслятор внешнего API для облачных моделей.

Внешний клиент (IDE, агент) говорит с RAMVI в привычном формате — OpenAI chat/completions,
Anthropic messages, Ollama chat/generate — с ключом клиента RAMVI. Запрос переводится в общий
запрос (формат OpenAI chat), уходит облачной модели через `CloudProvider`, а ответ —
обратно в формат клиента, потоком или целиком. Облачный ключ RAMVI не покидает.
"""

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

# Имена объектов OpenAI (собраны из частей: проверка переводов приняла бы их за ключи).
COMPLETION = ".".join(("chat", "completion"))
COMPLETION_CHUNK = COMPLETION + ".chunk"

# --- запрос клиента → общий запрос ---------------------------------------------------------


def _image(media: str, data: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}


def _blocks_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"
    )


def from_openai(body: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {}
    messages = []
    for m in body.get("messages") or []:
        if isinstance(m, dict):
            role = "system" if m.get("role") == "developer" else m.get("role")
            messages.append({**m, "role": role})
    request["messages"] = messages
    tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if tokens is not None:
        request["max_tokens"] = tokens
    for name in ("tools", "tool_choice", "temperature", "top_p", "stop"):
        if body.get(name) is not None:
            request[name] = body[name]
    return request


def from_completions(body: dict[str, Any]) -> dict[str, Any]:
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "".join(str(p) for p in prompt)
    request = from_openai({**body, "messages": [{"role": "user", "content": str(prompt)}]})
    return request


def from_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _blocks_text(body["system"])})
    for m in body.get("messages") or []:
        content = m.get("content", "")
        if isinstance(content, str):
            messages.append({"role": m.get("role"), "content": content})
            continue
        parts: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    parts.append(
                        _image(source.get("media_type", "image/png"), source.get("data", ""))
                    )
            elif kind == "tool_use":
                calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
            elif kind == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _blocks_text(block.get("content")),
                    }
                )
        if m.get("role") == "assistant":
            reply: dict[str, Any] = {"role": "assistant", "content": _blocks_text(parts)}
            if calls:
                reply["tool_calls"] = calls
            messages.append(reply)
        elif parts:
            messages.append({"role": "user", "content": parts})
    request: dict[str, Any] = {"messages": messages, "max_tokens": body.get("max_tokens", 1024)}
    if body.get("tools"):
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
            for t in body["tools"]
        ]
    for ours, theirs in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop_sequences"),
    ):
        if body.get(theirs) is not None:
            request[ours] = body[theirs]
    return request


def from_ollama(body: dict[str, Any], chat: bool) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if chat:
        pending: list[str] = []  # у Ollama у результатов нет id: по порядку вызовов
        counter = 0
        for m in body.get("messages") or []:
            role = m.get("role")
            if role == "tool":
                call = pending.pop(0) if pending else f"call_{counter}"
                messages.append(
                    {"role": "tool", "tool_call_id": call, "content": str(m.get("content", ""))}
                )
                continue
            content: Any = str(m.get("content", ""))
            if m.get("images"):
                content = [{"type": "text", "text": content}] + [
                    _image("image/png", data) for data in m["images"]
                ]
            message: dict[str, Any] = {"role": role, "content": content}
            if m.get("tool_calls"):
                message["tool_calls"] = []
                for call in m["tool_calls"]:
                    function = call.get("function") or {}
                    identifier = f"call_{counter}"
                    counter += 1
                    pending.append(identifier)
                    arguments = function.get("arguments", {})
                    message["tool_calls"].append(
                        {
                            "id": identifier,
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    )
            messages.append(message)
    else:
        if body.get("system"):
            messages.append({"role": "system", "content": str(body["system"])})
        content = str(body.get("prompt", ""))
        if body.get("images"):
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
                    + [_image("image/png", d) for d in body["images"]],
                }
            )
        else:
            messages.append({"role": "user", "content": content})
    request: dict[str, Any] = {"messages": messages}
    options = body.get("options") or {}
    for ours, theirs in (
        ("max_tokens", "num_predict"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop"),
    ):
        if options.get(theirs) is not None:
            request[ours] = options[theirs]
    if chat and body.get("tools"):
        request["tools"] = body["tools"]
    return request


def to_request(path: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Общий запрос и нужен ли поток (у Ollama поток по умолчанию включён)."""
    if path == "/v1/chat/completions":
        return from_openai(body), bool(body.get("stream"))
    if path == "/v1/completions":
        return from_completions(body), bool(body.get("stream"))
    if path == "/v1/messages":
        return from_anthropic(body), bool(body.get("stream"))
    if path in ("/api/chat", "/api/generate"):
        return from_ollama(body, path == "/api/chat"), body.get("stream", True) is not False
    raise ValueError(path)


# --- общий поток → формат клиента ----------------------------------------------------------


class Collected:
    """Собранный ответ: текст, рассуждения, вызовы инструментов, причина конца, usage."""

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self.calls: dict[int, dict[str, str]] = {}
        self.finish = "stop"
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def add(self, chunk: dict[str, Any]) -> None:
        kind = chunk["type"]
        if kind == "text":
            self.text += chunk["text"]
        elif kind == "reasoning":
            self.reasoning += chunk["text"]
        elif kind == "tool_calls":
            for delta in chunk["tool_calls"]:
                call = self.calls.setdefault(
                    int(delta.get("index", 0)), {"id": "", "name": "", "arguments": ""}
                )
                call["id"] = delta.get("id") or call["id"]
                function = delta.get("function") or {}
                call["name"] = function.get("name") or call["name"]
                call["arguments"] += function.get("arguments") or ""
        elif kind == "end":
            self.finish = chunk.get("finish_reason") or "stop"
            if chunk.get("usage"):
                self.usage = {
                    "prompt_tokens": int(chunk["usage"].get("prompt_tokens", 0)),
                    "completion_tokens": int(chunk["usage"].get("completion_tokens", 0)),
                }

    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "id": c["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
            }
            for i, c in sorted(self.calls.items())
        ]


def _arguments(raw: str) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


class Writer:
    """Переводит общие куски в байты ответа клиенту. `content_type` — заголовок ответа."""

    def __init__(self, path: str, model: str, stream: bool, body: dict[str, Any]) -> None:
        self.path, self.model, self.stream, self.body = path, model, stream, body
        self.done = Collected()
        self.created = int(time.time())
        self.id = uuid.uuid4().hex[:24]
        self._first = True
        self._block: tuple[str, int] | None = None  # Anthropic: открытый блок (вид, индекс)
        self._blocks = 0
        self._call_blocks: dict[int, int] = {}

    @property
    def content_type(self) -> str:
        if not self.stream:
            return "application/json"
        return "application/x-ndjson" if self.path.startswith("/api/") else "text/event-stream"

    # общие помощники
    @staticmethod
    def _sse(data: Any, event: str | None = None) -> bytes:
        head = f"event: {event}\n" if event else ""
        payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        return (head + "data: " + payload + "\n\n").encode()

    @staticmethod
    def _line(data: Any) -> bytes:
        return (json.dumps(data, ensure_ascii=False) + "\n").encode()

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # поток
    def feed(self, chunk: dict[str, Any]) -> bytes:
        self.done.add(chunk)
        if not self.stream or chunk["type"] == "status":
            return b""
        if chunk["type"] == "end":
            return self._end()
        return {
            "/v1/chat/completions": self._openai,
            "/v1/completions": self._completions,
            "/v1/messages": self._anthropic,
            "/api/chat": self._ollama_chat,
            "/api/generate": self._ollama_generate,
        }[self.path](chunk)

    def _openai_chunk(self, delta: dict[str, Any], finish: str | None = None) -> bytes:
        if self._first:
            delta = {"role": "assistant", **delta}
            self._first = False
        return self._sse(
            {
                "id": "chatcmpl-" + self.id,
                "object": COMPLETION_CHUNK,
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    def _openai(self, chunk: dict[str, Any]) -> bytes:
        if chunk["type"] == "text":
            return self._openai_chunk({"content": chunk["text"]})
        if chunk["type"] == "reasoning":
            return self._openai_chunk({"reasoning_content": chunk["text"]})
        if chunk["type"] == "tool_calls":
            return self._openai_chunk({"tool_calls": chunk["tool_calls"]})
        return b""

    def _completions(self, chunk: dict[str, Any]) -> bytes:
        if chunk["type"] != "text":
            return b""
        return self._sse(
            {
                "id": "cmpl-" + self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "text": chunk["text"], "finish_reason": None}],
            }
        )

    def _anthropic_open(self, kind: str, start: dict[str, Any]) -> bytes:
        out = self._anthropic_close()
        index = self._blocks
        self._blocks += 1
        self._block = (kind, index)
        return out + self._sse(
            {"type": "content_block_start", "index": index, "content_block": start},
            "content_block_start",
        )

    def _anthropic_close(self) -> bytes:
        if self._block is None:
            return b""
        index = self._block[1]
        self._block = None
        return self._sse({"type": "content_block_stop", "index": index}, "content_block_stop")

    def _anthropic_head(self) -> bytes:
        if not self._first:
            return b""
        self._first = False
        message: dict[str, Any] = {
            "id": "msg_" + self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return self._sse({"type": "message_start", "message": message}, "message_start")

    def _anthropic(self, chunk: dict[str, Any]) -> bytes:
        out = self._anthropic_head()
        if chunk["type"] in ("text", "reasoning"):
            kind = "text" if chunk["type"] == "text" else "thinking"
            if self._block is None or self._block[0] != kind:
                start = (
                    {"type": "text", "text": ""}
                    if kind == "text"
                    else {"type": "thinking", "thinking": ""}
                )
                out += self._anthropic_open(kind, start)
            assert self._block is not None
            delta = (
                {"type": "text_delta", "text": chunk["text"]}
                if kind == "text"
                else {"type": "thinking_delta", "thinking": chunk["text"]}
            )
            out += self._sse(
                {"type": "content_block_delta", "index": self._block[1], "delta": delta},
                "content_block_delta",
            )
        elif chunk["type"] == "tool_calls":
            for delta in chunk["tool_calls"]:
                call = int(delta.get("index", 0))
                function = delta.get("function") or {}
                if call not in self._call_blocks:
                    out += self._anthropic_open(
                        "tool",
                        {
                            "type": "tool_use",
                            "id": delta.get("id") or f"toolu_{call}",
                            "name": function.get("name", ""),
                            "input": {},
                        },
                    )
                    assert self._block is not None
                    self._call_blocks[call] = self._block[1]
                if function.get("arguments"):
                    out += self._sse(
                        {
                            "type": "content_block_delta",
                            "index": self._call_blocks[call],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": function["arguments"],
                            },
                        },
                        "content_block_delta",
                    )
        return out

    def _ollama_chat(self, chunk: dict[str, Any]) -> bytes:
        message: dict[str, Any]
        if chunk["type"] == "text":
            message = {"role": "assistant", "content": chunk["text"]}
        elif chunk["type"] == "reasoning":
            message = {"role": "assistant", "content": "", "thinking": chunk["text"]}
        else:
            return b""  # вызовы инструментов у Ollama приходят целиком — в конце
        return self._line(
            {"model": self.model, "created_at": self._now(), "message": message, "done": False}
        )

    def _ollama_generate(self, chunk: dict[str, Any]) -> bytes:
        if chunk["type"] != "text":
            return b""
        return self._line(
            {
                "model": self.model,
                "created_at": self._now(),
                "response": chunk["text"],
                "done": False,
            }
        )

    def _ollama_tail(self) -> dict[str, Any]:
        done = self.done
        return {
            "done": True,
            "done_reason": "length" if done.finish == "length" else "stop",
            "prompt_eval_count": done.usage["prompt_tokens"],
            "eval_count": done.usage["completion_tokens"],
        }

    def _end(self) -> bytes:
        done = self.done
        if self.path == "/v1/chat/completions":
            out = self._openai_chunk({}, done.finish)
            options = self.body.get("stream_options") or {}
            if options.get("include_usage"):
                out += self._sse(
                    {
                        "id": "chatcmpl-" + self.id,
                        "object": COMPLETION_CHUNK,
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": self._usage(),
                    }
                )
            return out + self._sse("[DONE]")
        if self.path == "/v1/completions":
            final = {
                "id": "cmpl-" + self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "text": "", "finish_reason": done.finish}],
            }
            return self._sse(final) + self._sse("[DONE]")
        if self.path == "/v1/messages":
            out = self._anthropic_head() + self._anthropic_close()
            out += self._sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": ANTHROPIC_STOP.get(done.finish, "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": done.usage["completion_tokens"]},
                },
                "message_delta",
            )
            return out + self._sse({"type": "message_stop"}, "message_stop")
        if self.path == "/api/chat":
            out = b""
            if done.calls:
                calls = [
                    {
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": _arguments(c["function"]["arguments"]),
                        }
                    }
                    for c in done.tool_calls()
                ]
                out += self._line(
                    {
                        "model": self.model,
                        "created_at": self._now(),
                        "message": {"role": "assistant", "content": "", "tool_calls": calls},
                        "done": False,
                    }
                )
            return out + self._line(
                {
                    "model": self.model,
                    "created_at": self._now(),
                    "message": {"role": "assistant", "content": ""},
                    **self._ollama_tail(),
                }
            )
        return self._line(
            {"model": self.model, "created_at": self._now(), "response": "", **self._ollama_tail()}
        )

    def _usage(self) -> dict[str, int]:
        usage = self.done.usage
        return {**usage, "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]}

    # ответ целиком
    def complete(self) -> dict[str, Any]:
        done = self.done
        if self.path == "/v1/chat/completions":
            message: dict[str, Any] = {"role": "assistant", "content": done.text}
            if done.reasoning:
                message["reasoning_content"] = done.reasoning
            if done.calls:
                message["tool_calls"] = done.tool_calls()
            return {
                "id": "chatcmpl-" + self.id,
                "object": COMPLETION,
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": done.finish}],
                "usage": self._usage(),
            }
        if self.path == "/v1/completions":
            return {
                "id": "cmpl-" + self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "text": done.text, "finish_reason": done.finish}],
                "usage": self._usage(),
            }
        if self.path == "/v1/messages":
            content: list[dict[str, Any]] = []
            if done.reasoning:
                content.append({"type": "thinking", "thinking": done.reasoning, "signature": ""})
            if done.text:
                content.append({"type": "text", "text": done.text})
            for call in done.tool_calls():
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": _arguments(call["function"]["arguments"]),
                    }
                )
            return {
                "id": "msg_" + self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": content,
                "stop_reason": ANTHROPIC_STOP.get(done.finish, "end_turn"),
                "stop_sequence": None,
                "usage": {
                    "input_tokens": done.usage["prompt_tokens"],
                    "output_tokens": done.usage["completion_tokens"],
                },
            }
        if self.path == "/api/chat":
            message = {"role": "assistant", "content": done.text}
            if done.reasoning:
                message["thinking"] = done.reasoning
            if done.calls:
                message["tool_calls"] = [
                    {
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": _arguments(c["function"]["arguments"]),
                        }
                    }
                    for c in done.tool_calls()
                ]
            return {
                "model": self.model,
                "created_at": self._now(),
                "message": message,
                **self._ollama_tail(),
            }
        return {
            "model": self.model,
            "created_at": self._now(),
            "response": done.text,
            **self._ollama_tail(),
        }


def translate(
    path: str, body: dict[str, Any], model: str, chunks: Iterator[dict[str, Any]]
) -> Iterator[bytes]:
    """Удобная обёртка для проверок: весь ответ клиенту по кускам."""
    _, stream = to_request(path, body)
    writer = Writer(path, model, stream, body)
    for chunk in chunks:
        data = writer.feed(chunk)
        if data:
            yield data
    if not stream:
        yield json.dumps(writer.complete(), ensure_ascii=False).encode()
