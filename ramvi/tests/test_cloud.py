"""7.7: облачные модели по API — три протокола на локальном тестовом сервере."""

import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.chat import ChatError
from ramvi_service.cloud import CloudConnections, CloudProvider, model_id
from ramvi_service.engines import EngineManager, SwitchableProvider
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client, text_of

KEY = "sk-test-SECRET-0123456789abcdef"
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "a"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
]
HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Read a.txt"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "hello"},
]


def sse(events: list[tuple[str | None, Any]]) -> bytes:
    out = b""
    for name, data in events:
        if name:
            out += f"event: {name}\n".encode()
        out += (
            b"data: " + (data if isinstance(data, bytes) else json.dumps(data).encode()) + b"\n\n"
        )
    return out


def openai_stream(tool: bool) -> bytes:
    events: list[tuple[str | None, Any]] = [
        (None, {"choices": [{"delta": {"reasoning_content": "hm"}}]}),
        (None, {"choices": [{"delta": {"content": "При"}}]}),
        (None, {"choices": [{"delta": {"content": "вет"}}]}),
    ]
    if tool:
        events.append(
            (
                None,
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "x",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": '{"path"'},
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        )
        events.append(
            (
                None,
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": ': "b"}'}}]
                            }
                        }
                    ]
                },
            )
        )
    events.append(
        (None, {"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool else "stop"}]})
    )
    events.append((None, {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}))
    events.append((None, b"[DONE]"))
    return sse(events)


def anthropic_stream(tool: bool) -> bytes:
    events: list[tuple[str | None, Any]] = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 7}}}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        ("ping", {"type": "ping"}),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "При"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "вет"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    if tool:
        events += [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "x",
                        "name": "read_file",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path"'},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ': "b"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ]
    events += [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if tool else "end_turn"},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return sse(events)


def gemini_stream(tool: bool) -> bytes:
    usage = {"promptTokenCount": 7, "candidatesTokenCount": 3}
    events: list[tuple[str | None, Any]] = [
        (None, {"candidates": [{"content": {"parts": [{"text": "hm", "thought": True}]}}]}),
        (None, {"candidates": [{"content": {"parts": [{"text": "При"}]}}]}),
        (None, {"candidates": [{"content": {"parts": [{"text": "вет"}]}}]}),
    ]
    last: dict[str, Any] = {"content": {"parts": []}, "finishReason": "STOP"}
    if tool:
        last["content"]["parts"].append(
            {"functionCall": {"name": "read_file", "args": {"path": "b"}}}
        )
    events.append((None, {"candidates": [last], "usageMetadata": usage}))
    return sse(events)


STREAMS = {"openai": openai_stream, "anthropic": anthropic_stream, "gemini": gemini_stream}
COMPLETE: dict[str, Callable[[bool], dict[str, Any]]] = {
    "openai": lambda tool: {
        "choices": [
            {
                "message": {
                    "content": "Привет",
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": "x",
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": '{"path": "b"}'},
                                }
                            ]
                        }
                        if tool
                        else {}
                    ),
                },
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    },
    "anthropic": lambda tool: {
        "content": [{"type": "text", "text": "Привет"}]
        + (
            [{"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "b"}}]
            if tool
            else []
        ),
        "stop_reason": "tool_use" if tool else "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    },
    "gemini": lambda tool: {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Привет"}]
                    + (
                        [{"functionCall": {"name": "read_file", "args": {"path": "b"}}}]
                        if tool
                        else []
                    )
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
    },
}
MODELS = {
    "openai": {"data": [{"id": "gpt-b"}, {"id": "gpt-a"}]},
    "anthropic": {"data": [{"id": "claude-x"}]},
    "gemini": {
        "models": [
            {"name": "models/gemini-x", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
        ]
    },
}


class Cloud:
    """Тестовый облачный сервер: /openai/v1, /anthropic, /gemini."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.tool = False
        self.slow: bool | str = False
        self.override: dict[str, Any] | None = None  # готовый ответ без потока
        self.disconnected = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _kind(self) -> str:
                return self.path.split("/")[1]

            def _send(
                self,
                status: int,
                body: bytes,
                kind: str = "application/json",
                extra: dict[str, str] | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _fail(self) -> bool:
                if owner.status == 200:
                    return False
                if owner.status == 302:
                    self._send(302, b"", extra={"Location": "http://127.0.0.1:1/steal"})
                else:
                    self._send(
                        owner.status, json.dumps({"error": {"message": "bad key " + KEY}}).encode()
                    )
                return True

            def do_GET(self) -> None:  # noqa: N802
                owner.requests.append(
                    {
                        "method": "GET",
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                if not self._fail():
                    self._send(200, json.dumps(MODELS[self._kind()]).encode())

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                if self._fail():
                    return
                kind = self._kind()
                stream = body.get("stream") or "alt=sse" in self.path
                if not stream:
                    self._send(
                        200, json.dumps(owner.override or COMPLETE[kind](owner.tool)).encode()
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if owner.slow == "hang":  # один кусок и тишина: чтение будит только закрытие
                    try:
                        self.wfile.write(
                            b'data: {"choices": [{"delta": {"content": "."}}]}\n'
                        )  # без пустой строки
                        self.wfile.flush()
                        time.sleep(2)
                        for _ in range(50):  # после разрыва запись падает не с первого раза
                            self.wfile.write(b": still here\n\n")
                            self.wfile.flush()
                            time.sleep(0.02)
                    except OSError:
                        owner.disconnected.set()
                    return
                if owner.slow:
                    try:
                        for _ in range(200):
                            self.wfile.write(
                                sse([(None, {"choices": [{"delta": {"content": "."}}]})])
                            )
                            self.wfile.flush()
                            time.sleep(0.02)
                    except OSError:
                        owner.disconnected.set()
                    return
                self.wfile.write(STREAMS[kind](owner.tool))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def url(self, kind: str) -> str:
        return self.base + ("/openai/v1" if kind == "openai" else "/" + kind)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Secrets:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self.values.get((namespace, key))

    def set(self, namespace: str, key: str, value: str) -> None:
        self.values[(namespace, key)] = value

    def delete(self, namespace: str, key: str) -> None:
        self.values.pop((namespace, key), None)


@pytest.fixture
def cloud() -> Iterator[Cloud]:
    server = Cloud()
    yield server
    server.close()


def provider(cloud: Cloud, kind: str, key: str | None = KEY) -> CloudProvider:
    return CloudProvider(kind, cloud.url(kind), f"{kind}-model", lambda: key)


def collect(
    chunks: Iterator[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    text = reasoning = ""
    calls: dict[int, dict[str, Any]] = {}
    end: dict[str, Any] = {}
    for chunk in chunks:
        if chunk["type"] == "text":
            text += chunk["text"]
        elif chunk["type"] == "reasoning":
            reasoning += chunk["text"]
        elif chunk["type"] == "tool_calls":
            for delta in chunk["tool_calls"]:
                call = calls.setdefault(delta["index"], {"id": "", "name": "", "arguments": ""})
                call["id"] = delta.get("id") or call["id"]
                function = delta.get("function") or {}
                call["name"] = function.get("name") or call["name"]
                call["arguments"] += function.get("arguments", "")
        elif chunk["type"] == "end":
            end = chunk
    return text, reasoning, [calls[i] for i in sorted(calls)], end


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
def test_stream_gives_the_same_chunks_for_every_protocol(cloud: Cloud, kind: str) -> None:
    request = {
        "messages": [{"role": "user", "content": "Привет"}],
        "max_tokens": 50,
        "temperature": 0,
        "enable_thinking": False,
    }
    text, reasoning, calls, end = collect(provider(cloud, kind).stream(request, threading.Event()))
    assert text == "Привет" and calls == []
    assert reasoning == ("" if kind == "anthropic" else "hm")
    assert end == {
        "type": "end",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }

    cloud.tool = True
    text, _, calls, end = collect(
        provider(cloud, kind).stream({**request, "tools": TOOLS}, threading.Event())
    )
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [
        ("read_file", {"path": "b"})
    ]
    assert calls[0]["id"] and end["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
def test_complete_returns_an_openai_reply_the_agent_understands(cloud: Cloud, kind: str) -> None:
    from ramvi_service.agent.strategy import LLMStrategy

    cloud.tool = True
    strategy = LLMStrategy(provider(cloud, kind).complete)
    reply = strategy.decide(list(HISTORY), TOOLS, threading.Event(), 100)
    assert reply.content == "Привет" and reply.finish_reason == "tool_calls"
    assert [(c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [
        ("read_file", {"path": "b"})
    ]
    assert reply.usage == {"prompt_tokens": 7, "completion_tokens": 3}


def test_requests_are_translated_for_each_service(cloud: Cloud) -> None:
    request = {
        "messages": HISTORY,
        "tools": TOOLS,
        "max_tokens": 50,
        "temperature": 0.3,
        "enable_thinking": False,
        "stop": "END",
    }
    for kind in ("openai", "anthropic", "gemini"):
        provider(cloud, kind).complete(request, threading.Event())
    openai, anthropic, gemini = (r for r in cloud.requests if r["method"] == "POST")

    assert openai["path"] == "/openai/v1/chat/completions"
    assert openai["headers"]["authorization"] == "Bearer " + KEY
    assert openai["body"] == {
        "messages": HISTORY,
        "tools": TOOLS,
        "max_tokens": 50,
        "temperature": 0.3,
        "stop": "END",
        "model": "openai-model",
    }

    assert anthropic["path"] == "/anthropic/v1/messages"
    assert anthropic["headers"]["x-api-key"] == KEY and "authorization" not in anthropic["headers"]
    body = anthropic["body"]
    assert (
        body["system"] == "Be brief."
        and body["max_tokens"] == 50
        and body["stop_sequences"] == ["END"]
    )
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.txt"}}
    ]
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "hello"}
    ]
    assert body["tools"][0]["input_schema"] == TOOLS[0]["function"]["parameters"]

    assert gemini["path"] == "/gemini/v1beta/models/gemini-model:generateContent"
    assert gemini["headers"]["x-goog-api-key"] == KEY
    body = gemini["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][2]["parts"] == [
        {"functionResponse": {"name": "read_file", "response": {"content": "hello"}}}
    ]
    assert body["generationConfig"] == {
        "maxOutputTokens": 50,
        "temperature": 0.3,
        "stopSequences": ["END"],
    }
    parameters = body["tools"][0]["functionDeclarations"][0]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    for r in cloud.requests:
        assert KEY not in json.dumps(r.get("body"))  # ключ только в заголовке


def test_images_go_inline_to_every_service(cloud: Cloud) -> None:
    from ramvi_service.cloud import image_part

    message = {
        "role": "user",
        "content": [{"type": "text", "text": "Что тут?"}, image_part("image/png", b"PNG")],
    }
    for kind in ("anthropic", "gemini"):
        provider(cloud, kind).complete({"messages": [message]}, threading.Event())
    anthropic, gemini = (r["body"] for r in cloud.requests)
    assert anthropic["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "UE5H"},
    }
    assert gemini["contents"][0]["parts"][1] == {
        "inlineData": {"mimeType": "image/png", "data": "UE5H"}
    }
    with pytest.raises(ChatError) as error:  # ссылка на картинку в сети не отправляется
        provider(cloud, "gemini").complete(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                    }
                ]
            },
            threading.Event(),
        )
    assert error.value.code == "cloud.image_not_inline"


@pytest.mark.parametrize("mode", [True, "hang"])
def test_cancel_mid_stream_stops_reading_and_drops_the_connection(
    cloud: Cloud, mode: bool | str
) -> None:
    cloud.slow = mode
    cancel = threading.Event()
    stream = provider(cloud, "openai").stream({"messages": []}, cancel)
    assert next(stream) == {"type": "text", "text": "."}
    started = time.monotonic()
    cancel.set()
    with pytest.raises(InterruptedError):
        list(stream)
    assert time.monotonic() - started < 1
    assert cloud.disconnected.wait(5)  # сервис видит разрыв и перестаёт считать


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "cloud.key_rejected"),
        (403, "cloud.key_rejected"),
        (429, "cloud.rate_limited"),
        (500, "cloud.service_error"),
        (302, "cloud.redirect_refused"),
    ],
)
def test_errors_are_clear_and_never_carry_the_key(cloud: Cloud, status: int, code: str) -> None:
    cloud.status = status
    with pytest.raises(ChatError) as error:
        list(provider(cloud, "openai").stream({"messages": []}, threading.Event()))
    assert error.value.code == code
    assert KEY not in json.dumps(error.value.params) and KEY not in str(error.value)
    assert not [r for r in cloud.requests if r["path"] == "/steal"]  # переадресация не выполнена


def test_https_only_and_key_required_outside_this_computer(cloud: Cloud) -> None:
    with pytest.raises(ChatError) as error:
        CloudProvider("openai", "http://example.com/v1", "m", lambda: KEY).complete(
            {"messages": []}, threading.Event()
        )
    assert error.value.code == "cloud.only_https"
    with pytest.raises(ChatError) as error:
        CloudProvider("openai", "https://example.com/v1", "m", lambda: None).complete(
            {"messages": []}, threading.Event()
        )
    assert error.value.code == "cloud.no_key"
    # LM Studio/vLLM на этом компьютере работают и без ключа.
    reply = provider(cloud, "openai", key=None).complete({"messages": []}, threading.Event())
    assert reply["choices"][0]["message"]["content"] == "Привет"
    assert "authorization" not in cloud.requests[-1]["headers"]


def test_temperature_is_dropped_once_when_the_model_rejects_it(cloud: Cloud) -> None:
    calls = []
    handler: Any = cloud.server.RequestHandlerClass
    original = handler.do_POST

    def picky(handler: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            length = int(handler.headers["Content-Length"])
            handler.rfile.read(length)
            body = b'{"error": {"message": "Unsupported value: temperature"}}'
            handler.send_response(400)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return
        original(handler)

    handler.do_POST = picky
    try:
        reply = provider(cloud, "openai").complete(
            {"messages": [], "temperature": 0}, threading.Event()
        )
    finally:
        handler.do_POST = original
    assert reply["choices"][0]["message"]["content"] == "Привет"
    assert len(calls) == 2 and "temperature" not in cloud.requests[-1]["body"]


def test_connections_keep_the_key_in_the_os_store_only(tmp_path: Path, cloud: Cloud) -> None:
    secrets = Secrets()
    db = open_core(tmp_path)
    connections = CloudConnections(tmp_path, secrets, Models(db))
    saved = connections.save(
        {"kind": "anthropic", "name": "Мой Claude", "url": cloud.url("anthropic"), "key": KEY}
    )
    assert saved["id"] == "claude" or saved["id"]
    assert saved["has_key"] and secrets.values == {("cloud", saved["id"]): KEY}
    assert KEY not in (tmp_path / "cloud.json").read_text() and KEY not in json.dumps(
        connections.view()
    )
    assert connections.remote_models({"id": saved["id"]}) == ["claude-x"]
    assert connections.remote_models(
        {"kind": "gemini", "url": cloud.url("gemini"), "key": KEY}
    ) == ["gemini-x"]
    assert connections.remote_models(
        {"kind": "openai", "url": cloud.url("openai"), "key": KEY}
    ) == ["gpt-a", "gpt-b"]

    connections.enable(saved["id"], "claude-x", True)
    model = Models(db).get(model_id(saved["id"], "claude-x"))
    assert (model.provider, model.path) == ("cloud", "claude-x")
    connections.save(
        {"id": saved["id"], "kind": "anthropic", "name": "Claude", "url": cloud.url("anthropic")}
    )
    assert secrets.values  # без поля key ключ остаётся
    with pytest.raises(ChatError):
        connections.save({"kind": "openai", "name": "x", "url": "http://example.com/v1"})
    connections.delete(saved["id"])
    assert (
        secrets.values == {} and Models(db).all() == [] and connections.view()["connections"] == []
    )
    db.close()


def test_local_engine_is_untouched_by_a_cloud_model(cloud: Cloud) -> None:
    class NoEngine:
        model_id = None

        def ensure(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("engine must not start for a cloud model")

        def provider(self) -> None:
            raise AssertionError("engine must not be used for a cloud model")

    remote = provider(cloud, "openai")
    switch = SwitchableProvider(NoEngine(), lambda m: remote if m == "cloud:c:m" else None)  # type: ignore[arg-type]
    request = {"messages": [], "ramvi_model_id": "cloud:c:m", "ramvi_min_context": 8192}
    assert switch.capabilities_for(request).external and not switch.capabilities_for({}).external
    chunks = list(switch.stream(request, threading.Event()))
    assert "".join(c["text"] for c in chunks if c["type"] == "text") == "Привет"
    assert not [c for c in chunks if c["type"] == "status"]  # «загрузка модели» не показывается
    assert (
        switch.complete(request, threading.Event())["choices"][0]["message"]["content"] == "Привет"
    )
    assert "ramvi_model_id" not in json.dumps(
        cloud.requests
    ) and "ramvi_min_context" not in json.dumps(cloud.requests)


def test_chat_through_the_app_with_only_a_cloud_model(tmp_path: Path, cloud: Cloud) -> None:
    """Без локальных моделей и без движка: подключение, выбор модели, ответ в чате."""
    secrets = Secrets()
    server, engines = build(
        tmp_path, TOKEN, engine_command=ENGINE, voice_options={"secrets": secrets}
    )
    server.start()
    try:
        client = Client(server.port)
        assert client.ok("GET", "/models")["models"] == []
        connection = client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "openai", "name": "Test", "url": cloud.url("openai"), "key": KEY},
        )["connection"]
        assert "key" not in connection and connection["has_key"]
        assert client.ok("POST", "/cloud/models", {"id": connection["id"]})["models"] == [
            "gpt-a",
            "gpt-b",
        ]
        client.ok(
            "POST",
            f"/cloud/connections/{connection['id']}/models",
            {"model": "gpt-a", "enabled": True},
        )
        models = client.ok("GET", "/models")["models"]
        assert [(m["id"], m["provider"], m["compat"]) for m in models] == [
            (f"cloud:{connection['id']}:gpt-a", "cloud", {"status": "cloud"})
        ]
        assert models[0]["cloud"] == {
            "connection": connection["id"],
            "name": "Test",
            "kind": "openai",
            "host": "127.0.0.1",
            "model": "gpt-a",
        }

        chat = client.ok("POST", "/chats", {"model_id": models[0]["id"]})["chat"]
        events = client.events(f"/chats/{chat['id']}/send", {"text": "Привет"})
        assert text_of(events, "text") == "Привет" and events[-1]["error"] is None
        assert not [e for e in events if e["type"] == "status"]
        assert engines.status()["state"] == "stopped" and engines.model_id is None
        sent = [r for r in cloud.requests if r["method"] == "POST"][-1]
        assert (
            sent["body"]["model"] == "gpt-a" and sent["body"]["messages"][-1]["content"] == "Привет"
        )

        loaded = client.ok("POST", "/engine/load", {"model_id": models[0]["id"]})
        assert loaded["started"] is False and loaded["cloud"] is True
        assert engines.status()["state"] == "stopped"
        client.ok("DELETE", f"/cloud/connections/{connection['id']}")
        assert client.ok("GET", "/models")["models"] == [] and secrets.values == {}
    finally:
        server.close()
        engines.stop()


def test_engine_refuses_a_cloud_model(tmp_path: Path) -> None:
    from ramvi_service.engines import EngineError

    db = open_core(tmp_path)
    connections = CloudConnections(tmp_path, Secrets(), Models(db))
    saved = connections.save({"kind": "openai", "name": "x", "url": "https://api.example.com/v1"})
    connections.enable(saved["id"], "gpt-a", True)
    manager = EngineManager(
        Models(db),
        tmp_path / "run",
        make_engine=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no engine")),
    )
    with pytest.raises(EngineError) as error:
        manager.ensure(model_id(saved["id"], "gpt-a"))
    assert error.value.code == "engine.cloud_model"
    db.close()


SCRIPT = """
import sys, json, tempfile
from pathlib import Path
sys.path[:0] = {paths!r}
from test_cloud import Cloud, Secrets, KEY
from test_chat import ENGINE, TOKEN, Client, text_of
from ramvi_service.app import build
cloud = Cloud()
data = Path(tempfile.mkdtemp())
server, engines = build(data, TOKEN, engine_command=ENGINE, voice_options={{"secrets": Secrets()}})
server.start()
client = Client(server.port)
c = client.ok("POST", "/cloud/connections", {{"kind": "gemini", "name": "g", "url": cloud.url("gemini"), "key": KEY}})["connection"]
client.ok("POST", "/cloud/connections/" + c["id"] + "/models", {{"model": "gemini-x", "enabled": True}})
chat = client.ok("POST", "/chats", {{"model_id": "cloud:" + c["id"] + ":gemini-x"}})["chat"]
text = text_of(client.events("/chats/" + chat["id"] + "/send", {{"text": "hi"}}), "text")
server.close(); engines.stop(); cloud.close()
heavy = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "triton", "transformers", "safetensors", "numpy")
               or "cuda" in m.lower() or m.startswith(tuple("ramvi_engine." + p for p in ("kernels", "server", "core", "generation", "cpu", "native", "formats"))))
print(json.dumps({{"text": text, "heavy": heavy}}))
"""


def test_cloud_mode_needs_no_gpu_libraries() -> None:
    """Облачный режим: ни torch, ни CUDA, ни ядра движка в процессе сервиса не загружаются."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    paths = [str(root / p) for p in ("tests", "service", "storage", "engine")]
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(paths=paths)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"text": "Привет", "heavy": []}


def test_module_needs_the_egress_grant_for_a_cloud_model(tmp_path: Path, cloud: Cloud) -> None:
    """Модуль (Character) с облачной моделью: текст уходит наружу только с правом egress.text."""
    from ramvi_service.context import GrantStore
    from ramvi_service.contracts.modules import Authority
    from ramvi_service.events import EventBus
    from ramvi_service.resources import ResourceBroker, SharedLLM
    from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
    from ramvi_service.services import ScopedServices

    class NoEngine:
        model_id = None

    authority = Authority()
    owner = authority.issue("character", "user", "s")
    grants = GrantStore(authority)
    grants.allow(owner, "llm")
    remote = provider(cloud, "openai")
    llm = SharedLLM(
        authority,
        SwitchableProvider(NoEngine(), lambda m: remote if m == "cloud:c:m" else None),  # type: ignore[arg-type]
    )
    scope = ScopedServices(
        owner,
        authority,
        grants,
        ScopedData(tmp_path, owner.module_id),
        ScopedSecrets(MemorySecrets(), owner.module_id),
        ResourceBroker(authority, {}),
        EventBus(authority),
        llm,
    )
    scope.active = True
    try:
        assert (
            scope.call("llm.capabilities", {"model": "cloud:c:m"})["capabilities"]["external"]
            is True
        )
        assert scope.call("llm.capabilities", {})["capabilities"]["external"] is False
        request = {
            "request": {
                "messages": [{"role": "user", "content": "hi"}],
                "ramvi_model_id": "cloud:c:m",
            }
        }
        with pytest.raises(PermissionError):
            scope.call("llm.complete", request)
        assert not [r for r in cloud.requests if r["method"] == "POST"]  # без права ничего не ушло
        grants.allow(owner, "egress.text")
        reply = scope.call("llm.complete", request)
        assert reply["choices"][0]["message"]["content"] == "Привет"
        # Поток, как его читает Character: llm.stream → llm.read до done.
        stream = scope.call("llm.stream", {**request, "timeout": 30})["stream_id"]
        chunks: list[dict[str, Any]] = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            packet = scope.call("llm.read", {"stream_id": stream, "timeout": 0.5})
            chunks += packet["chunks"]
            if packet["done"]:
                break
        assert "".join(c["text"] for c in chunks if c["type"] == "text") == "Привет"
        assert [c["finish_reason"] for c in chunks if c["type"] == "end"] == [
            "stop"
        ]  # как от движка
    finally:
        scope.stop()


def test_agent_task_runs_on_a_cloud_model(tmp_path: Path, cloud: Cloud) -> None:
    """Раздел «Агент» с облачной моделью: инструменты вызывает сервис, движок не нужен."""
    from ramvi_service.agent.desktop import AgentDesktop
    from ramvi_service.agent.gateway import AgentGateway
    from ramvi_service.contracts.modules import Authority, Manifest
    from ramvi_service.modules.host import ModuleHost
    from ramvi_service.resources import SharedLLM
    from ramvi_service.sections import SectionRegistry
    from ramvi_storage.repositories import AgentRuns

    class NoEngine:
        model_id = None

    cloud.override = {
        "content": [
            {
                "type": "tool_use",
                "id": "f1",
                "name": "finish",
                "input": {"summary": "Готово", "success": True},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }
    db = open_core(tmp_path / "data")
    connections = CloudConnections(tmp_path / "data", Secrets(), Models(db))
    saved = connections.save(
        {"kind": "anthropic", "name": "c", "url": cloud.url("anthropic"), "key": KEY}
    )
    connections.enable(saved["id"], "claude-x", True)
    authority = Authority()

    def cloud_for(identifier: str) -> CloudProvider | None:
        model = Models(db).get(identifier)
        return connections.provider(model) if model.provider == "cloud" else None

    llm = SharedLLM(authority, SwitchableProvider(NoEngine(), cloud_for))  # type: ignore[arg-type]
    host = ModuleHost(tmp_path / "host", llm=llm)
    host.agent = AgentGateway(authority, host.grants)
    sections = SectionRegistry(host)
    sections.install(
        Manifest(
            "agent",
            "1.0.0",
            "ramvi_service.agent.module:create",
            capabilities=("agent",),
            ui={"section": {"view": "agent", "title_key": "section.agent"}},
        )
    )
    desktop = AgentDesktop(tmp_path / "data", AgentRuns(db), host, sections, llm)
    project = tmp_path / "project"
    project.mkdir()
    try:
        task = desktop.route(
            "POST",
            "/agent/start",
            {
                "goal": "Проверь проект",
                "workspace": str(project),
                "model_id": model_id(saved["id"], "claude-x"),
            },
        )
        deadline = time.monotonic() + 5
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            state = desktop.route("GET", "/agent/state", {})
            if (state.get("task") or {}).get("state") not in (None, "running", "queued"):
                break
            time.sleep(0.02)
        assert state["task"]["task_id"] == task["task_id"] and state["task"]["state"] == "done", (
            state
        )
        sent = [r for r in cloud.requests if r["method"] == "POST"]
        assert sent and {t["name"] for t in sent[0]["body"]["tools"]} >= {"finish", "read_file"}
        assert sent[0]["body"]["model"] == "claude-x" and "ramvi_model_id" not in json.dumps(
            sent[0]["body"]
        )
    finally:
        desktop.close()
        host.close()
        host.agent.close()
        db.close()


def test_openai_itself_goes_through_the_responses_api() -> None:
    """Модели Codex у OpenAI есть только в /v1/responses; совместимые сервисы — chat completions."""
    from ramvi_service.cloud import (
        OpenAIProtocol,
        OpenAIResponsesProtocol,
        image_part,
        protocol_for,
    )

    assert isinstance(protocol_for("openai", "https://api.openai.com/v1"), OpenAIResponsesProtocol)
    assert type(protocol_for("openai", "https://openrouter.ai/api/v1")) is OpenAIProtocol
    history = [
        *HISTORY[:1],
        {
            "role": "user",
            "content": [{"type": "text", "text": "Что тут?"}, image_part("image/png", b"PNG")],
        },
        *HISTORY[2:],
    ]
    endpoint, headers, body = OpenAIResponsesProtocol().chat_request(
        "https://api.openai.com/v1",
        KEY,
        "gpt-codex",
        {"messages": history, "tools": TOOLS, "max_tokens": 9, "temperature": 0},
        True,
    )
    assert endpoint == "https://api.openai.com/v1/responses" and headers == {
        "Authorization": "Bearer " + KEY
    }
    assert body == {
        "model": "gpt-codex",
        "store": False,
        "stream": True,
        "instructions": "Be brief.",
        "max_output_tokens": 9,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Что тут?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,UE5H"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "read_file",
                "arguments": '{"path": "a.txt"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "hello"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": TOOLS[0]["function"]["parameters"],
            }
        ],
    }


def test_responses_stream_and_reply_give_the_common_chunks() -> None:
    from ramvi_service.cloud import OpenAIResponsesProtocol

    protocol = OpenAIResponsesProtocol()
    reader = protocol.reader()
    events = [
        ("response.created", {"type": "response.created"}),
        (
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "hm"},
        ),
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "При"}),
        ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "вет"}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "call_id": "x",
                    "name": "read_file",
                    "arguments": "",
                },
            },
        ),
        (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": '{"path"',
            },
        ),
        (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 2,
                "delta": ': "b"}',
            },
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                },
            },
        ),
    ]
    chunks = [c for name, data in events for c in reader.feed(name, json.dumps(data))]
    text, reasoning, calls, _ = collect(iter(chunks))
    assert (text, reasoning) == ("Привет", "hm") and reader.done
    assert [(c["id"], c["name"], json.loads(c["arguments"])) for c in calls] == [
        ("x", "read_file", {"path": "b"})
    ]
    assert reader.finish == "tool_calls" and reader.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }
    with pytest.raises(ChatError):
        list(
            protocol.reader().feed(
                "response.failed",
                json.dumps({"type": "response.failed", "response": {"error": {"message": "boom"}}}),
            )
        )

    reply = protocol.response(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": "Привет"}]},
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    )
    assert reply["choices"][0] == {
        "index": 0,
        "message": {"role": "assistant", "content": "Привет"},
        "finish_reason": "length",
    }
    reply = protocol.response(
        {
            "status": "completed",
            "output": [
                {"type": "function_call", "call_id": "x", "name": "finish", "arguments": "{}"}
            ],
        }
    )
    assert reply["choices"][0]["message"]["tool_calls"] == [
        {"id": "x", "type": "function", "function": {"name": "finish", "arguments": "{}"}}
    ]
    assert reply["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_gets_alternating_roles(cloud: Cloud) -> None:
    messages = [
        {"role": "user", "content": "раз"},
        {"role": "user", "content": "два"},
        {"role": "assistant", "content": "ок"},
    ]
    provider(cloud, "anthropic").complete({"messages": messages}, threading.Event())
    sent = cloud.requests[-1]["body"]["messages"]
    assert sent == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "раз"}, {"type": "text", "text": "два"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ок"}]},
    ]


def test_agent_on_screen_with_a_cloud_model_counts_as_external(
    tmp_path: Path, cloud: Cloud, monkeypatch: Any
) -> None:
    """Агент с управлением окном и облачной моделью: контекст окна уходит наружу только с egress-правом."""
    import ramvi_service.agent.desktop as desktop_module

    seen: list[bool] = []

    class Spy:
        def __init__(self, *args: Any, external: bool = False, **kwargs: Any) -> None:
            seen.append(external)

        def register(self, tools: Any) -> None:
            pass

    monkeypatch.setattr(desktop_module, "UIControl", Spy)
    from ramvi_service.agent.desktop import AgentDesktop
    from ramvi_service.agent.gateway import AgentGateway
    from ramvi_service.contracts.modules import Authority, Manifest
    from ramvi_service.modules.host import ModuleHost
    from ramvi_service.resources import SharedLLM
    from ramvi_service.sections import SectionRegistry
    from ramvi_storage.repositories import AgentRuns

    class NoEngine:
        model_id = None

    db = open_core(tmp_path / "data")
    connections = CloudConnections(tmp_path / "data", Secrets(), Models(db))
    saved = connections.save(
        {"kind": "openai", "name": "o", "url": cloud.url("openai"), "key": KEY}
    )
    connections.enable(saved["id"], "gpt-a", True)
    authority = Authority()
    llm = SharedLLM(authority, SwitchableProvider(NoEngine(), lambda m: None))  # type: ignore[arg-type]
    host = ModuleHost(tmp_path / "host", llm=llm)
    host.agent = AgentGateway(authority, host.grants)
    host.context = type("Context", (), {"adapter": object()})()
    sections = SectionRegistry(host)
    sections.install(
        Manifest(
            "agent",
            "1.0.0",
            "ramvi_service.agent.module:create",
            capabilities=("agent",),
            ui={"section": {"view": "agent", "title_key": "section.agent"}},
        )
    )
    desktop = AgentDesktop(tmp_path / "data", AgentRuns(db), host, sections, llm)
    desktop._target = {"application": "app", "window_id": "w"}
    project = tmp_path / "project"
    project.mkdir()
    owner = authority.issue("agent", "user", "t")
    try:
        desktop._bind(owner, str(project), model_id(saved["id"], "gpt-a"), True, "", 1)
        assert seen and seen[0] is True
    finally:
        host.agent.release_owner(owner)
        host.agent.close()
        db.close()


def test_responses_keeps_earlier_assistant_replies_as_output_text() -> None:
    from ramvi_service.cloud import OpenAIResponsesProtocol

    messages = [{"role": "user", "content": "раз"}, {"role": "assistant", "content": "ок"}]
    _, _, body = OpenAIResponsesProtocol().chat_request(
        "https://api.openai.com/v1", KEY, "m", {"messages": messages}, False
    )
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "раз"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "ок"}]},
    ]
