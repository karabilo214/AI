"""Поток ответа модели до модуля: первые куски до конца генерации, отмена, backpressure."""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import services
from ramvi_service.context import GrantStore
from ramvi_service.contracts.modules import Authority, Owner, ProviderCapabilities
from ramvi_service.events import EventBus
from ramvi_service.providers import LocalEngineProvider, stream_chunks
from ramvi_service.resources import ResourceBroker, SharedLLM
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.services import ScopedServices, ServiceClient, ServiceGateway

WORDS = [f"слово{i} " for i in range(20)]
DELAY = 0.05  # движок выдаёт слово раз в 50 мс, весь ответ ~1 с


class FakeEngine:
    """SSE в формате ramvi-engine /v1/chat/completions; отмена как в настоящем движке."""

    def __init__(self) -> None:
        self.cancelled: dict[str, float] = {}
        self.started: list[float] = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/engine/cancel":
                    engine.cancelled.setdefault(request["request_id"], time.monotonic())
                    raw = b'{"requested": true}'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                assert request["stream"] is True
                request_id = self.headers["X-Request-ID"]
                engine.started.append(time.monotonic())
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def sse(obj: Any) -> None:
                    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()

                def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

                try:
                    sse(chunk({"role": "assistant", "content": ""}))
                    sse(chunk({"reasoning_content": "думаю"}))
                    for word in WORDS:
                        time.sleep(DELAY)
                        if request_id in engine.cancelled:
                            sse({"error": {"message": "отменено", "type": "cancelled"}})
                            sse("[DONE]")
                            return
                        sse(chunk({"content": word}))
                    sse(chunk({}, "stop"))
                    sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 20}})
                    sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(4)


@pytest.fixture
def engine() -> Iterator[FakeEngine]:
    fake = FakeEngine()
    yield fake
    fake.close()


def make_scope(
    tmp_path: Path, provider: Any, owner_name: str = "character"
) -> tuple[ScopedServices, Authority, Owner]:
    authority = Authority()
    owner = authority.issue(owner_name, "user", "a")
    grants = GrantStore(authority)
    grants.allow(owner, "llm")
    scope = ScopedServices(
        owner,
        authority,
        grants,
        ScopedData(tmp_path, owner.module_id),
        ScopedSecrets(MemorySecrets(), owner.module_id),
        ResourceBroker(authority, {}),
        EventBus(authority),
        SharedLLM(authority, provider),
    )
    scope.active = True
    return scope, authority, owner


def client_for(scope: ScopedServices, authority: Authority) -> tuple[ServiceGateway, ServiceClient]:
    gateway = ServiceGateway(authority)
    gateway.bind(scope)
    return gateway, ServiceClient(gateway.start(), scope.owner.token, timeout=5)


def test_engine_chunks_are_normalized() -> None:
    def parse(event: dict[str, Any]) -> list[dict[str, Any]]:
        return list(stream_chunks(event))

    delta = {"choices": [{"delta": {"reasoning_content": "r", "content": "t"}}]}
    assert parse(delta) == [
        {"type": "reasoning", "text": "r"},
        {"type": "text", "text": "t"},
    ]
    calls = {"choices": [{"delta": {"tool_calls": [{"index": 0}]}, "finish_reason": "tool_calls"}]}
    assert parse(calls) == [
        {"type": "tool_calls", "tool_calls": [{"index": 0}]},
        {"type": "finish", "finish_reason": "tool_calls"},
    ]
    with pytest.raises(InterruptedError):
        parse({"error": {"message": "x", "type": "cancelled"}})
    with pytest.raises(RuntimeError):
        parse({"error": {"message": "x", "type": "server_error"}})


def test_module_gets_first_words_before_generation_ends(tmp_path: Path, engine: FakeEngine) -> None:
    scope, authority, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    gateway, client = client_for(scope, authority)
    try:
        start = time.monotonic()
        arrivals: list[tuple[float, dict[str, Any]]] = []
        for chunk in client.stream({"messages": [{"role": "user", "content": "привет"}]}):
            arrivals.append((time.monotonic() - start, chunk))
        texts = [c["text"] for _, c in arrivals if c["type"] == "text"]
        first_text = next(t for t, c in arrivals if c["type"] == "text")
        total = arrivals[-1][0]
        assert "".join(texts) == "".join(WORDS)
        assert arrivals[0][1] == {"type": "reasoning", "text": "думаю"}
        assert arrivals[-1][1] == {
            "type": "end",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 20},
        }
        # Первое слово — через одну задержку движка, а не после всего ответа.
        assert first_text < DELAY + 0.15 < total
        assert total >= DELAY * len(WORDS)
        assert scope.llm is not None and not scope.llm.status()["active"]
    finally:
        scope.stop()
        gateway.close()


def test_interrupt_stops_engine_and_frees_model_quickly(tmp_path: Path, engine: FakeEngine) -> None:
    scope, authority, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    gateway, client = client_for(scope, authority)
    try:
        seen = 0
        for chunk in client.stream({"messages": []}):
            seen += chunk["type"] == "text"
            if seen == 3:
                break  # перебивание: выход из цикла отменяет генерацию
        interrupted = time.monotonic()
        # Новый ответ после перебивания: модель свободна, старые слова не приходят.
        first = next(c for c in client.stream({"messages": []}) if c["type"] == "text")
        reaction = time.monotonic() - interrupted
        assert first == {"type": "text", "text": WORDS[0]}
        assert len(engine.cancelled) == 1
        cancel_at = next(iter(engine.cancelled.values()))
        assert cancel_at - interrupted < 0.1
        # Время до первого слова нового ответа включает одну задержку движка.
        assert reaction < 0.25 + DELAY
    finally:
        scope.stop()
        gateway.close()


def test_cancelled_stream_delivers_no_old_chunks(tmp_path: Path, engine: FakeEngine) -> None:
    scope, _, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    try:
        stream_id = scope.call("llm.stream", {"request": {"messages": []}})["stream_id"]
        time.sleep(DELAY * 4)
        assert scope.call("llm.cancel", {"stream_id": stream_id}) == {"ok": True}
        with pytest.raises(ValueError, match="unknown stream"):
            scope.call("llm.read", {"stream_id": stream_id})
        assert scope.call("llm.cancel", {"stream_id": stream_id}) == {"ok": True}
    finally:
        scope.stop()


def test_backpressure_holds_generation_until_module_reads(
    tmp_path: Path, engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "STREAM_BUFFER", 2)
    scope, _, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    try:
        stream_id = scope.call("llm.stream", {"request": {"messages": []}})["stream_id"]
        time.sleep(DELAY * 8)
        buffered = scope._streams[stream_id]
        assert len(buffered.chunks) == 2 and not buffered.finished
        chunks: list[dict[str, Any]] = []
        while True:
            result = scope.call("llm.read", {"stream_id": stream_id, "timeout": 0.5})
            chunks += result["chunks"]
            if result["done"]:
                break
        assert "".join(c["text"] for c in chunks if c["type"] == "text") == "".join(WORDS)
        assert chunks[-1]["type"] == "end"
    finally:
        scope.stop()


def test_module_that_stops_reading_releases_model(
    tmp_path: Path, engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "STREAM_BUFFER", 1)
    monkeypatch.setattr(services, "STREAM_IDLE", 0.3)
    scope, _, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    try:
        stream_id = scope.call("llm.stream", {"request": {"messages": []}})["stream_id"]
        deadline = time.monotonic() + 3
        while not engine.cancelled and time.monotonic() < deadline:
            time.sleep(0.02)
        assert engine.cancelled
        with pytest.raises(TimeoutError, match="does not read"):
            while True:
                scope.call("llm.read", {"stream_id": stream_id, "timeout": 0.5})
        assert scope.llm is not None and not scope.llm.status()["active"]
    finally:
        scope.stop()


def test_stop_cancels_running_stream(tmp_path: Path, engine: FakeEngine) -> None:
    scope, _, _ = make_scope(tmp_path, LocalEngineProvider(engine.url, "m", timeout=5))
    stream_id = scope.call("llm.stream", {"request": {"messages": []}})["stream_id"]
    time.sleep(DELAY * 3)
    started = time.monotonic()
    scope.stop()
    assert time.monotonic() - started < 0.5
    assert engine.cancelled
    with pytest.raises(PermissionError):
        scope.call("llm.read", {"stream_id": stream_id})


def test_provider_without_streaming_is_rejected(tmp_path: Path) -> None:
    class Plain:
        capabilities = ProviderCapabilities()

        def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
            return {}

        def cancel(self) -> None:
            pass

        def unload(self) -> None:
            pass

    scope, _, _ = make_scope(tmp_path, Plain())
    with pytest.raises(NotImplementedError, match="streaming"):
        scope.call("llm.stream", {"request": {"messages": []}})
    with pytest.raises(NotImplementedError, match="llm.stream"):
        scope.call("llm.complete", {"request": {"messages": [], "stream": True}})
    scope.stop()


def test_complete_only_provider_has_cancellable_async_job_without_claiming_streaming(
    tmp_path: Path,
) -> None:
    started = threading.Event()

    class Plain:
        capabilities = ProviderCapabilities()

        def complete(self, request, cancel):
            started.set()
            assert cancel.wait(3)
            raise InterruptedError("cancelled single completion")

        def cancel(self):
            pass

        def unload(self):
            pass

    scope, _, _ = make_scope(tmp_path, Plain())
    try:
        identifier = scope.call("llm.stream", {"request": {"messages": []}, "single": True})[
            "stream_id"
        ]
        stream = scope._streams[identifier]
        assert started.wait(2)
        scope.call("llm.cancel", {"stream_id": identifier})
        stream.thread.join(2)
        assert not stream.thread.is_alive() and stream.finished
        assert not scope.call("llm.capabilities", {})["capabilities"]["streaming"]
    finally:
        scope.stop()


def test_zero_chunk_keepalive_preserves_buffer_and_finished_result(tmp_path: Path) -> None:
    from ramvi_service.services import LLMStream

    class Plain:
        capabilities = ProviderCapabilities()

    scope, _, _ = make_scope(tmp_path, Plain())
    stream = LLMStream(threading.Event())
    stream.chunks.append({"type": "text", "text": "Fresh 524287"})
    stream.finished = True
    stream.last_read = 0
    scope._streams["retained"] = stream
    try:
        result = scope.call("llm.read", {"stream_id": "retained", "max_chunks": 0, "timeout": 0})
        assert result == {"chunks": [], "done": False} and stream.last_read > 0
        result = scope.call("llm.read", {"stream_id": "retained", "timeout": 0})
        assert result["chunks"][0]["text"] == "Fresh 524287"
        assert scope.call("llm.read", {"stream_id": "retained", "max_chunks": 0, "timeout": 0})[
            "done"
        ]
        assert "retained" in scope._streams
        assert scope.call("llm.read", {"stream_id": "retained", "timeout": 0})["done"]
        assert "retained" not in scope._streams
    finally:
        scope.stop()


def test_complete_only_job_delivers_one_result_and_finish(tmp_path: Path) -> None:
    class Plain:
        capabilities = ProviderCapabilities()

        def complete(self, request, cancel):
            return {"choices": [{"message": {"content": "Fresh 104729"}, "finish_reason": "stop"}]}

        def cancel(self):
            pass

        def unload(self):
            pass

    scope, _, _ = make_scope(tmp_path, Plain())
    chunks = []
    try:
        identifier = scope.call("llm.stream", {"request": {"messages": []}, "single": True})[
            "stream_id"
        ]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            packet = scope.call("llm.read", {"stream_id": identifier, "timeout": 0.1})
            chunks.extend(packet["chunks"])
            if packet["done"]:
                break
        else:
            raise AssertionError("completion did not finish")
        assert chunks == [
            {"type": "text", "text": "Fresh 104729"},
            {"type": "finish", "finish_reason": "stop"},
        ]
    finally:
        scope.stop()
