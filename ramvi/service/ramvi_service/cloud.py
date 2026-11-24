"""7.7 REMOTE-LLM-01: облачные модели по API — RAMVI работает транслятором.

Модель считает облачный сервис; Chat, Agent и Character шлют тот же запрос в формате
OpenAI chat и получают те же куски потока, что от локального движка
(`text`/`reasoning`/`tool_calls`, последний — `end` с `finish_reason` и `usage`).

Три протокола: OpenAI-совместимый (OpenAI, OpenRouter, DeepSeek, Groq, LM Studio/vLLM на
этом компьютере), Anthropic Messages, Google Gemini. Подключение — сервис, адрес и ключ;
ключ лежит только в хранилище секретов ОС и читается при каждом запросе. Только HTTPS
(HTTP — лишь к 127.0.0.1/localhost), переадресации не выполняются: иначе заголовок с
ключом ушёл бы на другой адрес. Видеокарта, драйвер NVIDIA и CUDA для этого не нужны.

Включённая модель подключения — строка таблицы `models` с `provider="cloud"`: чаты
ссылаются на модель внешним ключом, так облачная модель выбирается как локальная.
"""

import base64
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ramvi_storage.repositories import Model, Models, NotFound

from ramvi_service import web
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import ProviderCapabilities
from ramvi_service.redact import mask_secrets

if TYPE_CHECKING:
    from ramvi_service.codex_cli import CodexProvider

CLOUD = "cloud"  # Model.provider и Model.architecture облачной модели
SECRET_NS = "cloud"
# chatgpt — подписка ChatGPT через официальный Codex CLI (codex_cli.py): без ключа, адрес справочный.
KINDS = ("openai", "anthropic", "gemini", "chatgpt")
DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "chatgpt": "https://chatgpt.com",
}
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096  # Anthropic требует max_tokens в каждом запросе
MAX_RESPONSE = 8 * 2**20
MAX_LINE = 2**20
CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
CAPABILITIES = ProviderCapabilities(
    streaming=True, tools=True, image_input=True, json_schema=False, external=True
)
# Поля запроса, которые понимает облако; остальное (enable_thinking, ramvi_*) — только движку.
PASSED = ("messages", "tools", "tool_choice", "max_tokens", "temperature", "top_p", "stop")


def is_cloud(model: Model) -> bool:
    return model.provider == CLOUD


def model_id(connection: str, name: str) -> str:
    return f"{CLOUD}:{connection}:{name}"


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


OPENER = urllib.request.build_opener(_NoRedirects())


# ---------------------------------------------------------------- перевод запроса


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _parts(content: Any) -> list[dict[str, Any]]:
    """Части сообщения OpenAI: текст и картинки data:URL → общий вид {text}|{image}."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    result: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            result.append({"text": part["text"]})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            match = re.fullmatch(r"data:([\w/+.-]+);base64,(.+)", url, re.S)
            if match is None:
                raise ChatError("cloud.image_not_inline")
            result.append({"image": (match.group(1), match.group(2))})
    return result


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_value": value}


def _finish(reason: str | None, tool_calls: bool) -> str:
    if tool_calls:
        return "tool_calls"
    return {
        None: "stop",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "STOP": "stop",
        "max_tokens": "length",
        "MAX_TOKENS": "length",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
    }.get(reason, "content_filter" if reason else "stop")


class OpenAIProtocol:
    """OpenAI chat completions: запрос почти как есть, поток — тот же формат, что у движка."""

    kind = "openai"

    def models_request(self, url: str, key: str) -> tuple[str, dict[str, str]]:
        return url + "/models", self._headers(key)

    def models(self, reply: Any) -> list[str]:
        return sorted(
            str(m["id"]) for m in reply.get("data", []) if isinstance(m, dict) and m.get("id")
        )

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        return {"Authorization": "Bearer " + key} if key else {}

    def chat_request(
        self, url: str, key: str, model: str, request: dict[str, Any], stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        body: dict[str, Any] = {k: request[k] for k in PASSED if k in request}
        body["model"] = model
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return url + "/chat/completions", self._headers(key), body

    def response(self, reply: dict[str, Any]) -> dict[str, Any]:
        choice = (reply.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        **(
                            {"tool_calls": message["tool_calls"]}
                            if message.get("tool_calls")
                            else {}
                        ),
                    },
                    "finish_reason": choice.get("finish_reason") or "stop",
                }
            ],
            "usage": _usage(reply.get("usage")),
        }

    def reader(self) -> "_OpenAIReader":
        return _OpenAIReader()


def _usage(raw: Any) -> dict[str, int]:
    raw = raw if isinstance(raw, dict) else {}
    prompt = raw.get("prompt_tokens", raw.get("input_tokens", raw.get("promptTokenCount", 0)))
    completion = raw.get(
        "completion_tokens", raw.get("output_tokens", raw.get("candidatesTokenCount", 0))
    )
    return {"prompt_tokens": int(prompt or 0), "completion_tokens": int(completion or 0)}


class _OpenAIReader:
    def __init__(self) -> None:
        self.finish: str | None = None
        self.usage: dict[str, int] | None = None
        self.done = False

    def feed(self, event: str, data: str) -> Iterator[dict[str, Any]]:
        if data == "[DONE]":
            self.done = True
            return
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("cloud chunk must be an object")
        if "error" in chunk:
            raise ChatError("cloud.service_error", status=200, detail=str(chunk["error"])[:300])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
            if delta.get("content"):
                yield {"type": "text", "text": delta["content"]}
            if delta.get("tool_calls"):
                yield {"type": "tool_calls", "tool_calls": delta["tool_calls"]}
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        if chunk.get("usage"):
            self.usage = _usage(chunk["usage"])


class AnthropicProtocol:
    kind = "anthropic"

    def _headers(self, key: str) -> dict[str, str]:
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}

    def models_request(self, url: str, key: str) -> tuple[str, dict[str, str]]:
        return url + "/v1/models?limit=1000", self._headers(key)

    def models(self, reply: Any) -> list[str]:
        return sorted(
            str(m["id"]) for m in reply.get("data", []) if isinstance(m, dict) and m.get("id")
        )

    def chat_request(
        self, url: str, key: str, model: str, request: dict[str, Any], stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        system: list[str] = []
        messages: list[dict[str, Any]] = []

        def add(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if messages and messages[-1]["role"] == role:  # роли должны чередоваться
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})

        for m in request.get("messages", []):
            role = m.get("role")
            if role == "system":
                system.append(_text(m.get("content")))
            elif role == "tool":
                add(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(m.get("tool_call_id", "")),
                            "content": _text(m.get("content")),
                        }
                    ],
                )
            else:
                blocks: list[dict[str, Any]] = []
                for part in _parts(m.get("content")):
                    if "text" in part:
                        blocks.append({"type": "text", "text": part["text"]})
                    else:
                        media, data = part["image"]
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": media, "data": data},
                            }
                        )
                for call in m.get("tool_calls") or []:
                    function = call.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(call.get("id", "")),
                            "name": str(function.get("name", "")),
                            "input": _arguments(function.get("arguments")),
                        }
                    )
                add("assistant" if role == "assistant" else "user", blocks)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": int(request.get("max_tokens") or DEFAULT_MAX_TOKENS),
            "messages": messages,
        }
        if system:
            body["system"] = "\n\n".join(s for s in system if s)
        for name in ("temperature", "top_p"):
            if name in request:
                body[name] = request[name]
        if request.get("stop"):
            stop = request["stop"]
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        if request.get("tools"):
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters") or {"type": "object"},
                }
                for t in request["tools"]
                if t.get("type", "function") == "function"
            ]
        if stream:
            body["stream"] = True
        return url + "/v1/messages", self._headers(key), body

    def response(self, reply: dict[str, Any]) -> dict[str, Any]:
        text, calls = [], []
        for block in reply.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
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
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = calls
        return {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": _finish(reply.get("stop_reason"), bool(calls)),
                }
            ],
            "usage": _usage(reply.get("usage")),
        }

    def reader(self) -> "_AnthropicReader":
        return _AnthropicReader()


class _AnthropicReader:
    def __init__(self) -> None:
        self.finish: str | None = None
        self.usage: dict[str, int] | None = None
        self.done = False
        self._tools: dict[int, int] = {}  # индекс блока → индекс вызова инструмента
        self._prompt = 0

    def feed(self, event: str, data: str) -> Iterator[dict[str, Any]]:
        chunk = json.loads(data)
        kind = chunk.get("type", event)
        if kind == "error":
            error = chunk.get("error") or {}
            raise ChatError(
                "cloud.service_error", status=200, detail=str(error.get("message", error))[:300]
            )
        if kind == "message_start":
            self._prompt = int(
                ((chunk.get("message") or {}).get("usage") or {}).get("input_tokens", 0)
            )
        elif kind == "content_block_start":
            block = chunk.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = len(self._tools)
                self._tools[chunk.get("index", 0)] = index
                yield {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "index": index,
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {"name": block.get("name", ""), "arguments": ""},
                        }
                    ],
                }
        elif kind == "content_block_delta":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield {"type": "text", "text": delta["text"]}
            elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                yield {"type": "reasoning", "text": delta["thinking"]}
            elif delta.get("type") == "input_json_delta" and delta.get("partial_json"):
                index = self._tools.get(chunk.get("index", 0), 0)
                yield {
                    "type": "tool_calls",
                    "tool_calls": [
                        {"index": index, "function": {"arguments": delta["partial_json"]}}
                    ],
                }
        elif kind == "message_delta":
            reason = (chunk.get("delta") or {}).get("stop_reason")
            self.finish = _finish(reason, False)
            output = int((chunk.get("usage") or {}).get("output_tokens", 0))
            self.usage = {"prompt_tokens": self._prompt, "completion_tokens": output}
        elif kind == "message_stop":
            self.done = True


# Ключи JSON Schema, которых нет в схеме функций Gemini.
GEMINI_DROP = {
    "additionalProperties",
    "$schema",
    "$defs",
    "$ref",
    "default",
    "examples",
    "title",
    "const",
}


def _gemini_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k not in GEMINI_DROP}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


class GeminiProtocol:
    kind = "gemini"

    def models_request(self, url: str, key: str) -> tuple[str, dict[str, str]]:
        return url + "/v1beta/models?pageSize=1000", {"x-goog-api-key": key}

    def models(self, reply: Any) -> list[str]:
        names = []
        for m in reply.get("models", []):
            if isinstance(m, dict) and "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(str(m.get("name", "")).removeprefix("models/"))
        return sorted(n for n in names if n)

    def chat_request(
        self, url: str, key: str, model: str, request: dict[str, Any], stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        system: list[str] = []
        contents: list[dict[str, Any]] = []
        names: dict[str, str] = {}  # id вызова → имя функции (ответ Gemini ссылается на имя)

        def add(role: str, parts: list[dict[str, Any]]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for m in request.get("messages", []):
            role = m.get("role")
            if role == "system":
                system.append(_text(m.get("content")))
            elif role == "tool":
                call = str(m.get("tool_call_id", ""))
                add(
                    "user",
                    [
                        {
                            "functionResponse": {
                                "name": names.get(call, call),
                                "response": {"content": _text(m.get("content"))},
                            }
                        }
                    ],
                )
            else:
                parts: list[dict[str, Any]] = []
                for part in _parts(m.get("content")):
                    if "text" in part:
                        parts.append({"text": part["text"]})
                    else:
                        media, data = part["image"]
                        parts.append({"inlineData": {"mimeType": media, "data": data}})
                for call in m.get("tool_calls") or []:
                    function = call.get("function") or {}
                    names[str(call.get("id", ""))] = str(function.get("name", ""))
                    parts.append(
                        {
                            "functionCall": {
                                "name": str(function.get("name", "")),
                                "args": _arguments(function.get("arguments")),
                            }
                        }
                    )
                add("model" if role == "assistant" else "user", parts)
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(s for s in system if s)}]}
        config: dict[str, Any] = {}
        for ours, theirs in (
            ("max_tokens", "maxOutputTokens"),
            ("temperature", "temperature"),
            ("top_p", "topP"),
        ):
            if ours in request:
                config[theirs] = request[ours]
        if request.get("stop"):
            stop = request["stop"]
            config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        if config:
            body["generationConfig"] = config
        if request.get("tools"):
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parameters": _gemini_schema(
                                t["function"].get("parameters") or {"type": "object"}
                            ),
                        }
                        for t in request["tools"]
                        if t.get("type", "function") == "function"
                    ]
                }
            ]
        name = urllib.parse.quote(model, safe="-._~")
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{url}/v1beta/models/{name}:{action}", {"x-goog-api-key": key}, body

    def response(self, reply: dict[str, Any]) -> dict[str, Any]:
        reader = _GeminiReader()
        text, calls = [], []
        for chunk in reader.chunks(reply):
            if chunk["type"] == "text":
                text.append(chunk["text"])
            elif chunk["type"] == "tool_calls":
                calls.extend(chunk["tool_calls"])
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": reader.finish or "stop"}],
            "usage": reader.usage or _usage(None),
        }

    def reader(self) -> "_GeminiReader":
        return _GeminiReader()


class _GeminiReader:
    ends_with_stream = True  # у Gemini нет отдельного конца потока: конец — закрытие ответа

    def __init__(self) -> None:
        self.finish: str | None = None
        self.usage: dict[str, int] | None = None
        self.done = False
        self._calls = 0

    def feed(self, event: str, data: str) -> Iterator[dict[str, Any]]:
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("cloud chunk must be an object")
        yield from self.chunks(chunk)

    def chunks(self, chunk: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if "error" in chunk:
            error = chunk["error"]
            raise ChatError(
                "cloud.service_error", status=200, detail=str(error.get("message", error))[:300]
            )
        for candidate in chunk.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if "functionCall" in part:
                    call = part["functionCall"]
                    yield {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "index": self._calls,
                                "id": f"call_{self._calls}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name", ""),
                                    "arguments": json.dumps(
                                        call.get("args") or {}, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    }
                    self._calls += 1
                elif part.get("text"):
                    yield {
                        "type": "reasoning" if part.get("thought") else "text",
                        "text": part["text"],
                    }
            if candidate.get("finishReason"):
                self.finish = _finish(candidate["finishReason"], self._calls > 0)
        if chunk.get("usageMetadata"):
            self.usage = _usage(chunk["usageMetadata"])
            if self.finish is not None:
                self.done = True


PROTOCOLS: dict[str, Any] = {
    "openai": OpenAIProtocol(),
    "anthropic": AnthropicProtocol(),
    "gemini": GeminiProtocol(),
}


class OpenAIResponsesProtocol(OpenAIProtocol):
    """OpenAI Responses API (/v1/responses): только так работают модели Codex, и только здесь
    у новых моделей OpenAI вызываются инструменты. Ответы не хранятся у OpenAI (store=false)."""

    kind = "openai"

    def chat_request(
        self, url: str, key: str, model: str, request: dict[str, Any], stream: bool
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for m in request.get("messages", []):
            role = m.get("role")
            if role == "system":
                instructions.append(_text(m.get("content")))
            elif role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(m.get("tool_call_id", "")),
                        "output": _text(m.get("content")),
                    }
                )
            else:
                kind = "output_text" if role == "assistant" else "input_text"
                content: list[dict[str, Any]] = []
                for part in _parts(m.get("content")):
                    if "text" in part:
                        content.append({"type": kind, "text": part["text"]})
                    else:
                        media, data = part["image"]
                        content.append(
                            {"type": "input_image", "image_url": f"data:{media};base64,{data}"}
                        )
                if content:
                    items.append(
                        {"role": "assistant" if role == "assistant" else "user", "content": content}
                    )
                for call in m.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "")
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call.get("id", "")),
                            "name": str(function.get("name", "")),
                            "arguments": arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments),
                        }
                    )
        body: dict[str, Any] = {"model": model, "input": items, "store": False}
        if instructions:
            body["instructions"] = "\n\n".join(i for i in instructions if i)
        if request.get("max_tokens"):
            body["max_output_tokens"] = request["max_tokens"]
        if request.get("tools"):
            body["tools"] = [
                {
                    "type": "function",
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters") or {"type": "object"},
                }
                for t in request["tools"]
                if t.get("type", "function") == "function"
            ]
        if stream:
            body["stream"] = True
        return url + "/responses", self._headers(key), body

    def response(self, reply: dict[str, Any]) -> dict[str, Any]:
        text, calls = [], []
        for item in reply.get("output") or []:
            if item.get("type") == "message":
                text += [
                    c.get("text", "")
                    for c in item.get("content") or []
                    if c.get("type") == "output_text"
                ]
            elif item.get("type") == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                )
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = calls
        return {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": _responses_finish(reply, bool(calls)),
                }
            ],
            "usage": _usage(reply.get("usage")),
        }

    def reader(self) -> "_ResponsesReader":  # type: ignore[override]
        return _ResponsesReader()


def _responses_finish(reply: dict[str, Any], tool_calls: bool) -> str:
    if tool_calls:
        return "tool_calls"
    if reply.get("status") == "incomplete":
        reason = (reply.get("incomplete_details") or {}).get("reason")
        return "length" if reason == "max_output_tokens" else "content_filter"
    return "stop"


class _ResponsesReader:
    def __init__(self) -> None:
        self.finish: str | None = None
        self.usage: dict[str, int] | None = None
        self.done = False
        self._calls: dict[int, int] = {}  # output_index → индекс вызова инструмента

    def feed(self, event: str, data: str) -> Iterator[dict[str, Any]]:
        chunk = json.loads(data)
        kind = chunk.get("type", event)
        if kind == "response.output_text.delta" and chunk.get("delta"):
            yield {"type": "text", "text": chunk["delta"]}
        elif kind in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ) and chunk.get("delta"):
            yield {"type": "reasoning", "text": chunk["delta"]}
        elif (
            kind == "response.output_item.added"
            and (chunk.get("item") or {}).get("type") == "function_call"
        ):
            item = chunk["item"]
            index = len(self._calls)
            self._calls[chunk.get("output_index", index)] = index
            yield {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "index": index,
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }
                ],
            }
        elif kind == "response.function_call_arguments.delta" and chunk.get("delta"):
            index = self._calls.get(chunk.get("output_index", 0), 0)
            yield {
                "type": "tool_calls",
                "tool_calls": [{"index": index, "function": {"arguments": chunk["delta"]}}],
            }
        elif kind in ("response.completed", "response.incomplete"):
            response = chunk.get("response") or {}
            self.finish = _responses_finish(response, bool(self._calls))
            self.usage = _usage(response.get("usage"))
            self.done = True
        elif kind in ("response.failed", "error"):
            error = (chunk.get("response") or {}).get("error") or chunk.get("error") or chunk
            message = error.get("message", error) if isinstance(error, dict) else error
            raise ChatError("cloud.service_error", status=200, detail=str(message)[:300])


RESPONSES = OpenAIResponsesProtocol()


def protocol_for(kind: str, url: str) -> Any:
    """Сам OpenAI — через Responses API (Codex, инструменты новых моделей); совместимые
    сервисы (OpenRouter, DeepSeek, Groq, LM Studio) — через chat completions."""
    if kind == "openai" and (urllib.parse.urlsplit(url).hostname or "").endswith("openai.com"):
        return RESPONSES
    return PROTOCOLS[kind]


# ---------------------------------------------------------------- сеть и ошибки


def _error(error: Exception, host: str, key: str) -> ChatError:
    def clean(text: str) -> str:
        return mask_secrets(text.replace(key, "***") if key else text)[:300]

    if isinstance(error, urllib.error.HTTPError):
        detail = clean(error.read(2000).decode("utf-8", "replace"))
        after = _retry_after(error.headers.get("Retry-After") if error.headers else None)
        if error.code in (401, 403):
            return ChatError("cloud.key_rejected", host=host, status=error.code, detail=detail)
        if error.code == 429:
            return ChatError("cloud.rate_limited", host=host, detail=detail, retry_after=after)
        if error.code in RETRY_STATUS:  # перегружен/недоступен: запрос не выполнялся
            return ChatError(
                "cloud.service_error",
                host=host,
                status=error.code,
                detail=detail,
                retry_after=after,
            )
        if 300 <= error.code < 400:
            return ChatError("cloud.redirect_refused", host=host, status=error.code)
        return ChatError("cloud.service_error", host=host, status=error.code, detail=detail)
    reason = getattr(error, "reason", error)
    # Соединение не установлено — запрос до сервиса не дошёл, повтор безопасен.
    connect = isinstance(reason, ConnectionRefusedError | socket.gaierror)
    return ChatError("cloud.unreachable", host=host, detail=clean(str(reason)), connect=connect)


RETRY_STATUS = (503, 529)
RETRY_DELAYS = (1.0, 3.0)  # две попытки после первой
MAX_RETRY_AFTER = 10.0


def _retry_after(value: Any) -> float | None:
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER) if value is not None else None
    except (TypeError, ValueError):
        return None


def retryable(error: ChatError) -> bool:
    """Повтор только там, где запрос точно не выполнялся (и не списал деньги)."""
    if error.code == "cloud.rate_limited":
        return True
    if error.code == "cloud.service_error":
        return error.params.get("status") in RETRY_STATUS
    return error.code == "cloud.unreachable" and bool(error.params.get("connect"))


def _open(
    endpoint: str, headers: dict[str, str], body: dict[str, Any] | None, key: str, timeout: float
) -> Any:
    host = urllib.parse.urlsplit(endpoint).hostname or ""
    if not web.allowed(endpoint):
        raise ChatError("cloud.only_https", host=host)
    request = web.request(endpoint, {"Accept": "application/json", **headers})
    if body is not None:
        request.data = json.dumps(body, ensure_ascii=False).encode()
        request.add_header("Content-Type", "application/json")
        request.method = "POST"
    try:
        return OPENER.open(request, timeout=timeout)
    except (urllib.error.URLError, OSError) as error:
        raise _error(error, host, key) from None


def _read_json(response: Any) -> Any:
    data = response.read(MAX_RESPONSE + 1)
    if len(data) > MAX_RESPONSE:
        raise ValueError("cloud response exceeds the limit")
    return json.loads(data)


class CloudProvider:
    """StreamingLLMProvider одной облачной модели; держит только открытый ответ."""

    capabilities = CAPABILITIES

    def __init__(
        self,
        kind: str,
        url: str,
        model: str,
        key: Callable[[], str | None],
        timeout: float = 600.0,
        *,
        before: Callable[[], None] | None = None,
        after: Callable[[dict[str, int] | None, str], None] | None = None,
    ) -> None:
        self.protocol = protocol_for(kind, url)
        self.url, self.model, self.key, self.timeout = url.rstrip("/"), model, key, timeout
        self.host = urllib.parse.urlsplit(self.url).hostname or ""
        # 7.10: before — проверка лимита расходов; after — учёт ответа (usage, done/cut/error).
        self.before, self.after = before, after
        self.retry_delays = RETRY_DELAYS
        self._lock = threading.Lock()
        self._response: Any = None

    def _key(self) -> str:
        key = self.key() or ""
        if not key and urllib.parse.urlsplit(self.url).hostname not in web.LOOPBACK:
            raise ChatError("cloud.no_key", host=self.host)
        return key

    def _request(
        self, request: dict[str, Any], stream: bool, cancel: threading.Event | None = None
    ) -> tuple[Any, str]:
        if self.before is not None:
            self.before()  # лимит исчерпан — запрос не уходит
        key = self._key()
        endpoint, headers, body = self.protocol.chat_request(
            self.url, key, self.model, request, stream
        )
        for delay in (*self.retry_delays, None):
            try:
                return self._send(endpoint, headers, body, key), key
            except ChatError as error:
                if delay is None or not retryable(error):
                    raise
                pause = error.params.get("retry_after")
                if (cancel or threading.Event()).wait(delay if pause is None else pause):
                    raise InterruptedError("request cancelled") from None
        raise AssertionError("unreachable")

    def _send(self, endpoint: str, headers: dict[str, str], body: dict[str, Any], key: str) -> Any:
        try:
            return _open(endpoint, headers, body, key, self.timeout)
        except ChatError as error:
            # Модели-рассуждатели OpenAI отвергают temperature: один повтор без неё, до вывода.
            if (
                error.code == "cloud.service_error"
                and error.params.get("status") == 400
                and "temperature" in str(error.params.get("detail", ""))
                and "temperature" in body
            ):
                body.pop("temperature")
                return _open(endpoint, headers, body, key, self.timeout)
            raise

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if cancel.is_set():
            raise InterruptedError("request cancelled")
        response, key = self._request(request, False, cancel)
        try:
            with self._lock:
                self._response = response
            reply = _read_json(response)
        except (OSError, ValueError) as error:
            self._account(None, "cut" if cancel.is_set() else "error")
            if cancel.is_set():
                raise InterruptedError("request cancelled") from None
            raise _error(error, self.host, key) from None
        finally:
            self._close()
        if not isinstance(reply, dict):
            self._account(None, "error")
            raise ChatError(
                "cloud.service_error", host=self.host, status=200, detail="not an object"
            )
        result: dict[str, Any] = self.protocol.response(reply)
        self._account(result.get("usage"), "done")  # сервис посчитал и списал — учесть всегда
        if cancel.is_set():
            raise InterruptedError("late response cancelled")
        return result

    def _account(self, usage: dict[str, int] | None, state: str) -> None:
        if self.after is not None:
            self.after(usage, state)

    def stream(self, request: dict[str, Any], cancel: threading.Event) -> Iterator[dict[str, Any]]:
        if cancel.is_set():
            raise InterruptedError("request cancelled")
        response, key = self._request(request, True, cancel)
        with self._lock:
            self._response = response
        done = threading.Event()
        accounted = False

        def watch() -> None:
            while not done.wait(0.01):
                if cancel.is_set():
                    self._close()  # разбудить чтение; сервис прекращает ответ при разрыве
                    return

        threading.Thread(target=watch, name="ramvi-cloud-cancel", daemon=True).start()
        reader = self.protocol.reader()
        total, event = 0, ""
        try:
            while not reader.done:
                try:
                    line = response.readline(MAX_LINE + 1)
                except (OSError, ValueError, AttributeError) as error:
                    if cancel.is_set():
                        raise InterruptedError("request cancelled") from None
                    raise _error(error, self.host, key) from None
                if cancel.is_set():
                    raise InterruptedError("request cancelled")
                if not line:
                    if getattr(reader, "ends_with_stream", False) and reader.finish is not None:
                        break
                    raise ChatError("cloud.stream_cut", host=self.host)
                total += len(line)
                if len(line) > MAX_LINE or total > MAX_RESPONSE:
                    raise ValueError("cloud stream exceeds the limit")
                text = line.decode("utf-8").strip()
                if text.startswith("event:"):
                    event = text[6:].strip()
                elif text.startswith("data:"):
                    yield from reader.feed(event, text[5:].strip())
                    event = ""
            accounted = True
            self._account(reader.usage, "done")
            yield {"type": "end", "finish_reason": reader.finish or "stop", "usage": reader.usage}
        finally:
            done.set()
            self._close()
            if not accounted:  # оборван: что успел сообщить сервис (часто — ничего)
                self._account(reader.usage, "cut" if cancel.is_set() else "error")

    def _close(self) -> None:
        with self._lock:
            response, self._response = self._response, None
        if response is not None:
            # close() из другого потока не будит заблокированное чтение: сначала сокет.
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            try:
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
                    if sys.platform == "win32":
                        # На Windows shutdown не будит recv в другом потоке, а close()
                        # откладывается, пока файл ответа держит сокет. Закрыть дескриптор
                        # напрямую; detach — чтобы объект не закрыл его второй раз.
                        socket.close(sock.detach())
            except OSError:
                pass
            try:
                response.close()
            except OSError:
                pass

    def cancel(self) -> None:
        self._close()

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------- подключения


class CloudConnections:
    """Подключения к облачным сервисам: `cloud.json` в папке данных + ключи в хранилище ОС."""

    def __init__(
        self, data_dir: Path, secrets: Any, models: Models, timeout: float = 600.0
    ) -> None:
        from ramvi_service.cloud_budget import Budget

        self.path = Path(data_dir) / "cloud.json"
        self.secrets, self.models, self.timeout = secrets, models, timeout
        self.budget = Budget(Path(data_dir) / "cloud_usage.sqlite3")  # 7.10: учёт ответов
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, **changes: Any) -> None:
        data = {**self._read(), **changes}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.path)

    def _load(self) -> list[dict[str, str]]:
        items = self._read().get("connections")
        return [c for c in items or [] if isinstance(c, dict) and c.get("kind") in KINDS]

    def _save(self, items: list[dict[str, str]]) -> None:
        self._write(connections=items)

    # --- 7.10: цены, лимиты, учёт --------------------------------------------------------

    def _prices(self) -> dict[str, dict[str, Any]]:
        prices = self._read().get("prices")
        return prices if isinstance(prices, dict) else {}

    def _limits(self) -> dict[str, float | None]:
        limits = self._read().get("limits")
        limits = limits if isinstance(limits, dict) else {}
        return {"daily": limits.get("daily"), "monthly": limits.get("monthly")}

    def set_limits(self, body: dict[str, Any]) -> dict[str, Any]:
        """Лимит расходов в долларах за день и за месяц; null — без лимита."""
        if set(body) - {"daily", "monthly"}:
            raise ChatError("api.bad_request", field="limits")
        limits = {}
        for name in ("daily", "monthly"):
            value = body.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0 <= value <= 1e6
            ):
                raise ChatError("api.bad_request", field=name)
            limits[name] = None if value is None else float(value)
        with self._lock:
            self._write(limits=limits)
        return self.spending()

    def set_price(self, connection_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Цена модели за миллион входных/выходных токенов (USD) с источником; дата — сегодня."""
        if set(body) - {"model", "input", "output", "source"}:
            raise ChatError("api.bad_request", field="price")
        name = body.get("model")
        if not isinstance(name, str) or not name:
            raise ChatError("api.bad_request", field="model")
        with self._lock:
            self._find(connection_id)
            identifier = model_id(connection_id, name)
            prices = self._prices()
            if body.get("input") is None and body.get("output") is None:
                prices.pop(identifier, None)  # цену убрали
            else:
                values = [body.get("input"), body.get("output")]
                if any(
                    isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v <= 1e4
                    for v in values
                ):
                    raise ChatError("api.bad_request", field="price")
                source = body.get("source")
                if not isinstance(source, str) or not source.strip() or len(source) > 300:
                    raise ChatError("cloud.price_source_required")
                prices[identifier] = {
                    "input": float(values[0] or 0),  # проверены выше: числа
                    "output": float(values[1] or 0),
                    "source": source.strip(),
                    "checked_on": time.strftime("%Y-%m-%d"),
                }
            self._write(prices=prices)
            return {"model": identifier, "price": prices.get(identifier)}

    def spending(self) -> dict[str, Any]:
        today, unknown_today = self.budget.spent("day")
        month, unknown_month = self.budget.spent("month")
        return {
            "currency": "USD",
            "limits": self._limits(),
            "today": today,
            "month": month,
            "unknown_today": unknown_today,
            "unknown_month": unknown_month,
            "prices": self._prices(),
            "recent": self.budget.recent(),
        }

    def check(self, identifier: str) -> None:
        """Перед запросом: лимит исчерпан — новый запрос не уходит (идущий не обрывается)."""
        limits = self._limits()
        if limits["daily"] is None and limits["monthly"] is None:
            return
        if identifier not in self._prices():
            # Без цены расход не посчитать, и лимит не сработал бы.
            raise ChatError("cloud.price_missing", model=identifier.split(":", 2)[-1])
        for name, period in (("daily", "day"), ("monthly", "month")):
            limit = limits[name]
            spent, _ = self.budget.spent(period)
            if limit is not None and spent >= limit:
                raise ChatError("cloud.limit_reached", period=period, limit=limit, spent=spent)

    def _key(self, connection_id: str) -> str | None:
        if self.secrets is None:
            return None
        try:
            value = self.secrets.get(SECRET_NS, connection_id)
        except Exception:  # хранилище недоступно: как «ключа нет», без текста ошибки
            return None
        return value if isinstance(value, str) else None

    def _find(self, connection_id: str) -> dict[str, str]:
        for item in self._load():
            if item["id"] == connection_id:
                return item
        raise ChatError("cloud.not_found", connection=connection_id)

    def _enabled(self, connection_id: str) -> list[str]:
        return [
            m.path
            for m in self.models.all()
            if is_cloud(m) and m.capabilities.get(CLOUD) == connection_id
        ]

    def view(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kinds": [{"kind": k, "url": DEFAULT_URLS[k]} for k in KINDS],
                "connections": [
                    {
                        **c,
                        "has_key": c["kind"] == "chatgpt" or bool(self._key(c["id"])),
                        "models": self._enabled(c["id"]),
                    }
                    for c in self._load()
                ],
                "spending": self.spending(),
            }

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        """Создать или изменить подключение. Ключ: строка — заменить, "" — удалить, нет — оставить."""
        if set(body) - {"id", "kind", "name", "url", "key"}:
            raise ChatError(
                "api.bad_request",
                field=", ".join(sorted(set(body) - {"id", "kind", "name", "url", "key"})),
            )
        kind, name, key = body.get("kind"), body.get("name"), body.get("key")
        if kind not in KINDS:
            raise ChatError("api.bad_request", field="kind")
        url = str(body.get("url") or DEFAULT_URLS[kind]).strip().rstrip("/")
        if not web.allowed(url) or urllib.parse.urlsplit(url).query:
            raise ChatError("cloud.only_https", host=urllib.parse.urlsplit(url).hostname or url)
        if not isinstance(name, str) or not name.strip() or len(name) > 64:
            raise ChatError("api.bad_request", field="name")
        if key is not None and (not isinstance(key, str) or len(key) > 4096):
            raise ChatError("api.bad_request", field="key")
        if kind == "chatgpt":
            # Вход хранит Codex CLI; ключа и своего адреса у подключения нет.
            if key:
                raise ChatError("api.bad_request", field="key")
            key, url = None, DEFAULT_URLS["chatgpt"]
        with self._lock:
            items = self._load()
            identifier = body.get("id")
            if identifier is None:
                base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or kind
                taken = {c["id"] for c in items}
                identifier, n = base, 2
                while identifier in taken:
                    identifier, n = f"{base}-{n}", n + 1
            elif not isinstance(identifier, str) or not CONNECTION_ID.match(identifier):
                raise ChatError("api.bad_request", field="id")
            elif identifier not in {c["id"] for c in items}:
                raise ChatError("cloud.not_found", connection=identifier)
            if key is not None:
                if self.secrets is None:
                    raise ChatError("cloud.secrets_unavailable")
                try:
                    if key.strip():
                        self.secrets.set(SECRET_NS, identifier, key.strip())
                    else:
                        self.secrets.delete(SECRET_NS, identifier)
                except Exception as error:
                    raise ChatError(
                        "cloud.secrets_unavailable", detail=mask_secrets(str(error))[:200]
                    ) from None
            entry = {"id": identifier, "kind": kind, "name": name.strip(), "url": url}
            items = [entry if c["id"] == identifier else c for c in items]
            if entry not in items:
                items.append(entry)
            self._save(items)
            return {
                **entry,
                "has_key": bool(self._key(identifier)),
                "models": self._enabled(identifier),
            }

    def delete(self, connection_id: str) -> None:
        with self._lock:
            items = self._load()
            self._find(connection_id)
            for name in self._enabled(connection_id):
                self.models.delete(model_id(connection_id, name))  # у чатов модель станет пустой
            if self.secrets is not None:
                try:
                    self.secrets.delete(SECRET_NS, connection_id)
                except Exception:
                    pass
            self._save([c for c in items if c["id"] != connection_id])
            prefix = model_id(connection_id, "")
            self._write(
                prices={k: v for k, v in self._prices().items() if not k.startswith(prefix)}
            )

    def remote_models(self, body: dict[str, Any]) -> list[str]:
        """Список моделей сервиса — проверка ключа. Значения формы или сохранённое подключение."""
        if isinstance(body.get("id"), str):
            saved = self._find(body["id"])
            kind, url = saved["kind"], saved["url"]
            key: str | None = (
                body["key"]
                if isinstance(body.get("key"), str) and body["key"]
                else self._key(saved["id"])
            )
        else:
            kind = str(body.get("kind") or "")
            key = body["key"] if isinstance(body.get("key"), str) else None
            if kind not in KINDS:
                raise ChatError("api.bad_request", field="kind")
            url = str(body.get("url") or DEFAULT_URLS[kind]).strip().rstrip("/")
        if kind == "chatgpt":
            from ramvi_service import codex_cli

            return codex_cli.models()
        secret = key or ""
        endpoint, headers = PROTOCOLS[kind].models_request(url, secret)
        host = urllib.parse.urlsplit(endpoint).hostname or ""
        response = _open(endpoint, headers, None, secret, 30)
        try:
            reply = _read_json(response)
        except (OSError, ValueError) as error:
            raise _error(error, host, secret) from None
        finally:
            response.close()
        if not isinstance(reply, dict):
            raise ChatError("cloud.service_error", host=host, status=200, detail="not an object")
        return list(PROTOCOLS[kind].models(reply))

    def enable(self, connection_id: str, name: str, enabled: bool) -> dict[str, Any]:
        """Модель сервиса появляется (или пропадает) в выборе модели Chat/Agent/Character."""
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ChatError("api.bad_request", field="model")
        with self._lock:
            self._find(connection_id)
            identifier = model_id(connection_id, name.strip())
            if enabled:
                try:
                    self.models.get(identifier)
                except NotFound:
                    self.models.upsert(
                        Model(
                            identifier,
                            CLOUD,
                            CLOUD,
                            name.strip(),
                            time.time(),
                            capabilities={CLOUD: connection_id},
                        )
                    )
            else:
                try:
                    self.models.delete(identifier)
                except NotFound:
                    pass
            return {"id": connection_id, "models": self._enabled(connection_id)}

    def provider(self, model: Model, source: str = "app") -> "CloudProvider | CodexProvider":
        """`source` — кто спросил (приложение или клиент внешнего API), для учёта."""
        connection = self._find(str(model.capabilities.get(CLOUD, "")))

        def after(usage: dict[str, int] | None, state: str) -> None:
            self.budget.record(model.id, source, usage, self._prices().get(model.id), state)

        if connection["kind"] == "chatgpt":
            from ramvi_service.codex_cli import CodexProvider

            return CodexProvider(model.path, before=lambda: self.check(model.id), after=after)
        return CloudProvider(
            connection["kind"],
            connection["url"],
            model.path,
            lambda: self._key(connection["id"]),
            self.timeout,
            before=lambda: self.check(model.id),
            after=after,
        )


def image_part(media: str, data: bytes) -> dict[str, Any]:
    """Часть сообщения OpenAI с картинкой (для проверок и модулей)."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media};base64," + base64.b64encode(data).decode()},
    }
