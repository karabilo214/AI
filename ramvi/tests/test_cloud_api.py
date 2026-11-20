"""7.9: транслятор — внешние клиенты получают облачные модели через локальный API RAMVI."""

import http.client
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.api_server import ApiOptions
from ramvi_service.app import build
from ramvi_service.cloud_api import to_request, translate
from test_api_server import Api
from test_chat import ENGINE, TOKEN
from test_cloud import HISTORY, KEY, TOOLS, Cloud, Secrets

MODEL = "cloud:test:gpt-a"


@pytest.fixture
def cloud() -> Iterator[Cloud]:
    server = Cloud()
    yield server
    server.close()


@pytest.fixture
def svc(tmp_path: Path, cloud: Cloud) -> Iterator[tuple[Any, str, str]]:
    server, engines = build(
        tmp_path, TOKEN, engine_command=ENGINE, voice_options={"secrets": Secrets()}
    )
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    connection = server.cloud.save(
        {"kind": "openai", "name": "Test", "url": cloud.url("openai"), "key": KEY}
    )
    assert connection["id"] == "test"
    server.cloud.enable("test", "gpt-a", True)
    _, key = server.api.keys.create("ide", {"chat", "models", "cloud"})
    _, plain = server.api.keys.create("plain", {"chat", "models"})
    try:
        yield server, key, plain
    finally:
        server.close()
        engines.stop()


def sent(cloud: Cloud) -> list[dict[str, Any]]:
    return [r for r in cloud.requests if r["method"] == "POST"]


def test_every_client_format_becomes_one_request() -> None:
    anthropic, stream = to_request(
        "/v1/messages",
        {
            "model": "m",
            "system": [{"type": "text", "text": "Be brief."}],
            "max_tokens": 50,
            "stop_sequences": ["END"],
            "stream": True,
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": TOOLS[0]["function"]["parameters"],
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read a.txt"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "UE5H"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "c1",
                            "name": "read_file",
                            "input": {"path": "a.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c1",
                            "content": [{"type": "text", "text": "hello"}],
                        }
                    ],
                },
            ],
        },
    )
    assert stream
    assert anthropic == {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read a.txt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,UE5H"}},
                ],
            },
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
        ],
        "max_tokens": 50,
        "tools": TOOLS,
        "stop": ["END"],
    }

    ollama, stream = to_request(
        "/api/chat",
        {
            "model": "m",
            "options": {"num_predict": 9, "temperature": 0},
            "messages": [
                {"role": "user", "content": "Что тут?", "images": ["UE5H"]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                    ],
                },
                {"role": "tool", "content": "hello"},
            ],
        },
    )
    assert stream  # у Ollama поток по умолчанию
    assert ollama["max_tokens"] == 9 and ollama["temperature"] == 0
    assert ollama["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,UE5H"},
    }
    assert (
        ollama["messages"][1]["tool_calls"][0]["id"]
        == ollama["messages"][2]["tool_call_id"]
        == "call_0"
    )
    assert json.loads(ollama["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "a.txt"
    }

    generate, stream = to_request("/api/generate", {"prompt": "hi", "system": "s", "stream": False})
    assert not stream and generate["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]
    openai, _ = to_request(
        "/v1/chat/completions",
        {
            "messages": [{"role": "developer", "content": "x"}],
            "max_completion_tokens": 5,
            "model": "m",
        },
    )
    assert openai == {"messages": [{"role": "system", "content": "x"}], "max_tokens": 5}
    completions, _ = to_request("/v1/completions", {"prompt": ["a", "b"], "max_tokens": 3})
    assert completions == {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 3}


CHUNKS: list[dict[str, Any]] = [
    {"type": "reasoning", "text": "hm"},
    {"type": "text", "text": "При"},
    {"type": "text", "text": "вет"},
    {
        "type": "tool_calls",
        "tool_calls": [
            {
                "index": 0,
                "id": "x",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path"'},
            }
        ],
    },
    {"type": "tool_calls", "tool_calls": [{"index": 0, "function": {"arguments": ': "b"}'}}]},
    {
        "type": "end",
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    },
]


def test_answers_come_back_in_the_client_format() -> None:
    raw = b"".join(translate("/v1/messages", {"stream": True}, MODEL, iter(CHUNKS))).decode()
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    assert [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"] == [
        "thinking",
        "text",
        "tool_use",
    ]
    assert kinds.count("content_block_start") == kinds.count("content_block_stop") == 3
    assert (
        "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
        == "Привет"
    )
    assert (
        "".join(
            e["delta"].get("partial_json", "") for e in events if e["type"] == "content_block_delta"
        )
        == '{"path": "b"}'
    )
    assert events[-2]["delta"]["stop_reason"] == "tool_use" and events[-2]["usage"] == {
        "output_tokens": 3
    }

    whole = json.loads(b"".join(translate("/v1/messages", {}, MODEL, iter(CHUNKS))))
    assert [b["type"] for b in whole["content"]] == ["thinking", "text", "tool_use"]
    assert whole["content"][2]["input"] == {"path": "b"} and whole["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
    }

    lines = [
        json.loads(x)
        for x in b"".join(translate("/api/chat", {}, MODEL, iter(CHUNKS))).splitlines()
    ]
    assert "".join(x["message"]["content"] for x in lines) == "Привет"
    assert lines[-2]["message"]["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "b"}}}
    ]
    assert (
        lines[-1]["done"] and lines[-1]["eval_count"] == 3 and lines[-1]["prompt_eval_count"] == 7
    )

    whole = json.loads(b"".join(translate("/v1/chat/completions", {}, MODEL, iter(CHUNKS))))
    assert whole["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path": "b"}',
    }
    assert (
        whole["choices"][0]["finish_reason"] == "tool_calls"
        and whole["usage"]["total_tokens"] == 10
    )

    raw = b"".join(translate("/v1/completions", {"stream": True}, MODEL, iter(CHUNKS))).decode()
    texts = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert "".join(t["choices"][0]["text"] for t in texts) == "Привет" and raw.rstrip().endswith(
        "data: [DONE]"
    )
    generated = json.loads(
        b"".join(translate("/api/generate", {"stream": False}, MODEL, iter(CHUNKS)))
    )
    assert generated["response"] == "Привет" and generated["done"]


def test_official_openai_client_gets_the_cloud_model(
    svc: tuple[Any, str, str], cloud: Cloud
) -> None:
    openai = pytest.importorskip("openai")
    server, key, _ = svc
    client = openai.OpenAI(base_url=server.api.url + "/v1", api_key=key)
    assert MODEL in [m.id for m in client.models.list()]
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Привет"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    parts, usage = [], None
    for chunk in stream:
        parts += [c.delta.content or "" for c in chunk.choices]
        usage = chunk.usage or usage
    assert "".join(parts) == "Привет" and usage is not None and usage.total_tokens == 10
    cloud.tool = True
    reply = client.chat.completions.create(
        model="gpt-a", messages=HISTORY, tools=TOOLS
    )  # короткое имя тоже
    call = reply.choices[0].message.tool_calls[0]
    assert (call.function.name, json.loads(call.function.arguments)) == ("read_file", {"path": "b"})
    # Облако видит облачный ключ RAMVI, а не ключ клиента; клиент облачного ключа не видит.
    assert {r["headers"]["authorization"] for r in sent(cloud)} == {"Bearer " + KEY}
    assert all(key not in json.dumps(r) for r in cloud.requests)
    assert sent(cloud)[-1]["body"]["model"] == "gpt-a" and sent(cloud)[-1]["body"]["tools"] == TOOLS


def test_official_anthropic_client_gets_the_cloud_model(
    svc: tuple[Any, str, str], cloud: Cloud
) -> None:
    anthropic = pytest.importorskip("anthropic")
    server, key, _ = svc
    client = anthropic.Anthropic(base_url=server.api.url, api_key=key)
    reply = client.messages.create(
        model=MODEL, max_tokens=50, messages=[{"role": "user", "content": "Привет"}]
    )
    assert [b.type for b in reply.content] == [
        "thinking",
        "text",
    ]  # рассуждения облака — отдельным блоком
    assert reply.content[1].text == "Привет" and reply.stop_reason == "end_turn"
    with client.messages.stream(
        model=MODEL, max_tokens=50, messages=[{"role": "user", "content": "Привет"}]
    ) as stream:
        assert "".join(stream.text_stream) == "Привет"
    cloud.tool = True
    reply = client.messages.create(
        model=MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": "Read b"}],
        tools=[{"name": "read_file", "input_schema": TOOLS[0]["function"]["parameters"]}],
    )
    tool = [b for b in reply.content if b.type == "tool_use"][0]
    assert (tool.name, tool.input) == (
        "read_file",
        {"path": "b"},
    ) and reply.stop_reason == "tool_use"


def test_official_ollama_client_gets_the_cloud_model(svc: tuple[Any, str, str]) -> None:
    ollama = pytest.importorskip("ollama")
    server, key, _ = svc
    client = ollama.Client(host=server.api.url, headers={"Authorization": f"Bearer {key}"})
    assert MODEL in [m.model for m in client.list().models]
    text = "".join(
        part.message.content
        for part in client.chat(
            model=MODEL, messages=[{"role": "user", "content": "Привет"}], stream=True
        )
    )
    assert text == "Привет"
    assert client.generate(model=MODEL, prompt="Привет").response == "Привет"


def test_raw_completions_and_ollama_lines(svc: tuple[Any, str, str]) -> None:
    server, key, _ = svc
    api = Api(server.api.port, key)
    status, body = api.call("POST", "/v1/completions", {"model": MODEL, "prompt": "Привет"})
    assert status == 200 and body["choices"][0]["text"] == "Привет"
    connection = http.client.HTTPConnection("127.0.0.1", server.api.port, timeout=30)
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})
    connection.request(
        "POST",
        "/api/chat",
        body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    response = connection.getresponse()
    lines = [json.loads(line) for line in response.read().splitlines()]
    connection.close()
    assert response.status == 200 and response.getheader("Content-Type") == "application/x-ndjson"
    assert lines[-1]["done"] and "".join(x["message"]["content"] for x in lines) == "Привет"


def test_cloud_models_need_their_own_key_permission(
    svc: tuple[Any, str, str], cloud: Cloud
) -> None:
    server, _, plain = svc
    api = Api(server.api.port, plain)
    status, body = api.call(
        "POST",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 403 and body["error"]["type"] == "api.cloud_forbidden"
    assert not sent(cloud)  # без права ни один запрос не ушёл в облако
    status, body = api.call("GET", "/v1/models")
    assert status == 503 or MODEL not in [m["id"] for m in body["data"]]  # облачные модели не видны
    status, body = api.call("GET", "/api/tags")
    assert status == 503 or all(m["name"] != MODEL for m in body["models"])


def test_usage_is_counted_for_the_client(svc: tuple[Any, str, str]) -> None:
    server, key, _ = svc
    Api(server.api.port, key).call(
        "POST",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    row = next(c for c in server.api.clients() if c["name"] == "ide")
    assert (row["requests"], row["prompt_tokens"], row["completion_tokens"], row["errors"]) == (
        1,
        7,
        3,
        0,
    )
    assert (
        server.cloud.spending()["recent"][0]["source"] == "api:ide"
    )  # 7.10: расход видно по программе


def test_cloud_errors_are_clear_and_keep_the_cloud_key_hidden(
    svc: tuple[Any, str, str], cloud: Cloud
) -> None:
    server, key, _ = svc
    cloud.status = 401
    status, body = Api(server.api.port, key).call(
        "POST",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 502 and body["error"]["params"]["cloud_code"] == "cloud.key_rejected"
    assert body["error"]["params"]["cloud"]["status"] == 401
    assert KEY not in json.dumps(body)


def test_revoking_the_key_stops_its_cloud_stream(svc: tuple[Any, str, str], cloud: Cloud) -> None:
    server, key, _ = svc
    cloud.slow = True
    received: list[bytes] = []
    finished = threading.Event()

    def read() -> None:
        connection = http.client.HTTPConnection("127.0.0.1", server.api.port, timeout=30)
        body = json.dumps(
            {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            body,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        while chunk := response.read1(4096):
            received.append(chunk)
        connection.close()
        finished.set()

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + 5
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    assert received
    record = next(c for c in server.api.clients() if c["name"] == "ide")
    started = time.monotonic()
    assert server.api.revoke(record["id"], cancel_running=True) == 1
    assert finished.wait(3) and time.monotonic() - started < 2  # поток оборвался сразу
    assert cloud.disconnected.wait(5)  # облако перестало считать
    status, _ = Api(server.api.port, key).call(
        "POST", "/v1/chat/completions", {"model": MODEL, "messages": []}
    )
    assert status == 401


def test_names_prefer_the_local_model_and_ambiguous_short_names_are_refused(
    tmp_path: Path, cloud: Cloud
) -> None:
    from test_chat import add_models

    add_models(tmp_path, "only-one")
    server, engines = build(
        tmp_path, TOKEN, engine_command=ENGINE, voice_options={"secrets": Secrets()}
    )
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    hi = [{"role": "user", "content": "hi"}]
    try:
        engines.ensure("only-one")
        for name in ("one", "two"):
            server.cloud.save(
                {"kind": "openai", "name": name, "url": cloud.url("openai"), "key": KEY}
            )
            server.cloud.enable(name, "gpt-a", True)
        server.cloud.enable("one", "only-one", True)
        _, key = server.api.keys.create("ide", {"chat", "models", "cloud"})
        api = Api(server.api.port, key)
        status, body = api.call(
            "POST", "/v1/chat/completions", {"model": "only-one", "messages": hi, "temperature": 0}
        )
        assert status == 200 and "echo" in json.dumps(
            body, ensure_ascii=False
        )  # то же имя — локальная модель
        assert not sent(cloud)
        engines.stop()
        status, body = api.call(
            "POST", "/v1/chat/completions", {"model": "only-one", "messages": hi}
        )
        assert (
            status == 200 and body["choices"][0]["message"]["content"] == "Привет"
        )  # имя одно — облако
        status, body = api.call("POST", "/v1/chat/completions", {"model": "gpt-a", "messages": hi})
        assert (
            status == 503 and len(sent(cloud)) == 1
        )  # «gpt-a» есть у двух подключений — не угадываем
        status, body = api.call(
            "POST", "/v1/chat/completions", {"model": "cloud:two:gpt-a", "messages": hi}
        )
        assert status == 200 and len(sent(cloud)) == 2
    finally:
        server.close()
        engines.stop()


def test_client_leaving_mid_stream_stops_the_cloud_answer(
    svc: tuple[Any, str, str], cloud: Cloud
) -> None:
    server, key, _ = svc
    cloud.slow = True
    connection = http.client.HTTPConnection("127.0.0.1", server.api.port, timeout=30)
    body = json.dumps(
        {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    )
    connection.request(
        "POST",
        "/v1/chat/completions",
        body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    response = connection.getresponse()
    assert response.read1(4096)
    response.close()  # клиент ушёл посреди ответа (ответ держит сокет — закрыть и его)
    connection.close()
    assert cloud.disconnected.wait(5)
    deadline = time.monotonic() + 5
    while (
        server.api.running(next(c for c in server.api.clients() if c["name"] == "ide")["id"])
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    row = next(c for c in server.api.clients() if c["name"] == "ide")
    assert row["running"] == 0 and row["errors"] == 1
