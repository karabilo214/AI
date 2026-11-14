"""Метрики: ручные формулы, контрольный backend и HTTP без GPU."""

import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_engine import metrics
from ramvi_engine.lifecycle import EngineSession, GenerationCancelled, ModelBundle
from ramvi_engine.server.app import App, make_handler


class Tokenizer:
    def encode(self, text: Any, **kwargs: Any) -> list[int]:
        return [99]

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return "input"

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join({0: "O", 1: "K"}[token] for token in tokens)


class Engine:
    ck = SimpleNamespace(eos={-1})
    maxt = 2048
    pos = None

    def __init__(self, fail: bool = False) -> None:
        self.place = ["gpu", "stream"]
        self.fail = fail

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        st = kwargs["stats"]
        st.update(
            prompt_n=1,
            prompt_ms=12.0,
            first_token_ms=12.0,
            gen_ms=28.0,
            drafted=4,
            accepted=2,
            draft_calls=2,
            draft_skips=1,
            draft_probes=0,
            passes=2,
            steps1=1,
        )
        for index, token in enumerate([0, 1, -1]):
            self.place = ["gpu", "cpu"]  # final placement differs from startup plan
            st["gen_n"] = index + 1
            yield token
            if self.fail:
                raise RuntimeError("private failure text")


def test_manual_formula_and_ttft_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.2, 10.5])
    monkeypatch.setattr("ramvi_engine.metrics.time.perf_counter", lambda: next(times))
    meter = metrics.GenerationMetrics("test", started=10.0)
    meter.observe("text", "visible")
    result = meter.finish(
        {
            "first_token_ms": 100.0,
            "prompt_ms": 100.0,
            "gen_ms": 1000.0,
            "prompt_n": 7,
            "drafted": 10,
            "accepted": 6,
        },
        {"gen_n": 5, "prompt_n": 13},
        {},
        {},
        "completed",
    )
    assert result["ttft_ms"] == pytest.approx(200.0)
    assert result["engine_first_token_ms"] == 100.0
    assert result["worker_total_ms"] == 500.0
    assert result["decode_tokens_per_second"] == 4.0
    assert result["draft"]["acceptance_rate"] == 0.6
    assert result["input_tokens"] == 13 and result["prefill_tokens"] == 7


def test_zero_and_unavailable_metrics_are_not_fabricated() -> None:
    meter = metrics.GenerationMetrics("zero")
    result = meter.finish(
        {"first_token_ms": 0.0, "gen_ms": 0.0, "drafted": 0, "accepted": 0},
        {"gen_n": 0, "prompt_n": 1},
        {},
        {},
        "completed",
    )
    assert result["ttft_ms"] is result["engine_first_token_ms"] is None
    assert result["decode_tokens_per_second"] is result["draft"]["acceptance_rate"] is None
    assert metrics.number({"x": float("nan")}, "x") is None
    assert metrics.number({"x": float("inf")}, "x") is None
    assert metrics.number({"x": -1}, "x") is None
    assert metrics.number({"x": True}, "x") is None
    json.dumps(result, allow_nan=False)


def test_memory_probe_uses_device_counters_and_stops_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rss = iter([100, 180])
    allocated = iter([200, 240])
    resets = []
    monkeypatch.setattr(metrics, "process_rss", lambda: next(rss))
    monkeypatch.setattr(metrics.MemoryMonitor, "INTERVAL", 100.0)
    cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda dev: resets.append(dev),
        memory_allocated=lambda dev: next(allocated),
        max_memory_allocated=lambda dev: 260,
        max_memory_reserved=lambda dev: 400,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    probe = metrics.MemoryMonitor()
    probe.start(SimpleNamespace(pos=SimpleNamespace(device=Device())))
    data = probe.stop()
    assert resets == ["cuda:0"]
    assert data["peak_vram_allocated_bytes"] == 260
    assert data["peak_vram_reserved_bytes"] == 400
    assert data["peak_ram_bytes"] == 180
    assert data["ram_samples"] == 2
    assert data["driver_memory_included"] is False
    assert probe._thread is not None and not probe._thread.is_alive()
    assert probe.stop() == data


def test_cpu_or_unloaded_probe_never_initializes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "process_rss", lambda: None)
    probe = metrics.MemoryMonitor()
    probe.start(object())
    data = probe.stop()
    assert data["peak_ram_bytes"] is data["peak_vram_allocated_bytes"] is None
    assert probe._thread is None


def test_app_records_actual_plan_and_private_snapshot() -> None:
    app = App(Engine(), Tokenizer(), "test")
    events = list(app.run([99]))
    record = events[-1][1]["metrics"]
    assert record["outcome"] == "completed"
    assert record["model"] == "test"
    assert record["started_at_unix_s"] <= record["finished_at_unix_s"]
    assert record["output_tokens"] == 3  # includes EOS, as standard usage already does
    assert record["decode_tokens"] == 2
    assert record["decode_tokens_per_second"] == pytest.approx(2 * 1000 / 28)
    assert record["draft"]["acceptance_rate"] == 0.5
    assert record["plan"]["layers"] == ["gpu", "cpu"]
    snapshot = app.metrics.snapshot()
    assert snapshot["counts"] == {"completed": 1} and snapshot["active"] == 0
    record["plan"]["layers"].clear()
    assert app.metrics.snapshot()["last"]["plan"]["layers"] == ["gpu", "cpu"]
    assert not any(t.name == "ramvi-rss" for t in threading.enumerate())


def test_error_abort_and_cancel_have_partial_metrics() -> None:
    app = App(Engine(fail=True), Tokenizer(), "test")
    with pytest.raises(RuntimeError) as failure:
        list(app.run([99]))
    assert failure.value.ramvi_metrics["outcome"] == "error"  # type: ignore[attr-defined]
    assert "private failure" not in json.dumps(app.metrics.snapshot())
    raw = app._run([99])
    next(raw)
    raw.close()
    assert app.metrics.snapshot()["last"]["outcome"] == "aborted"
    session = EngineSession(lambda: ModelBundle(Engine(), Tokenizer()))
    session.load()
    managed = App(session, session.tokenizer, "test")
    raw = managed._run([99])
    next(raw)
    session.cancel()
    with pytest.raises(GenerationCancelled):
        next(raw)
    assert managed.metrics.snapshot()["last"]["outcome"] == "cancelled"
    assert managed.metrics.snapshot()["active"] == 0
    session.close()
    assert not any(t.name == "ramvi-rss" for t in threading.enumerate())


@pytest.mark.parametrize("stream", [False, True])
def test_every_api_response_and_status(stream: bool) -> None:
    app = App(Engine(), Tokenizer(), "test", api_key="secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def request(method: str, path: str, body: Any = None, auth: bool = True) -> tuple[int, str]:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.request(
                method,
                path,
                None if body is None else json.dumps(body),
                {"Authorization": "Bearer secret"} if auth else {},
            )
            response = conn.getresponse()
            return response.status, response.read().decode()
        finally:
            conn.close()

    routes = [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/api/chat",
        "/api/generate",
    ]
    try:
        assert request("GET", "/engine/metrics", auth=False)[0] == 401
        for index, path in enumerate(routes):
            body = {
                "model": "test",
                "stream": stream,
                "max_tokens": 3,
                "temperature": 0,
                "prompt": "SECRET_INPUT",
                "raw": True,
                "messages": [{"role": "user", "content": "SECRET_INPUT"}],
            }
            status, text = request("POST", path, body)
            assert status == 200, text
            if not stream:
                found = [json.loads(text)["ramvi_metrics"]]
            elif path.startswith("/api/"):
                found = [
                    row["ramvi_metrics"]
                    for row in map(json.loads, text.splitlines())
                    if "ramvi_metrics" in row
                ]
            else:
                rows = [
                    json.loads(line[6:])
                    for line in text.splitlines()
                    if line.startswith("data: ") and line[6:] != "[DONE]"
                ]
                found = [row["ramvi_metrics"] for row in rows if "ramvi_metrics" in row]
            assert len(found) == 1 and found[0]["outcome"] == "completed"
            status, text = request("GET", "/engine/metrics")
            snapshot = json.loads(text)
            assert status == 200
            assert snapshot["metrics"]["last"]["generation_id"] == found[0]["generation_id"]
            assert snapshot["metrics"]["counts"]["completed"] == index + 1
            assert snapshot["metrics"]["active"] == 0
            assert "SECRET_INPUT" not in text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def test_cuda_probe_failure_does_not_break_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "process_rss", lambda: None)

    def fail(_: Any) -> None:
        raise RuntimeError("CUDA metrics unavailable")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=fail))
    )
    probe = metrics.MemoryMonitor()
    probe.start(SimpleNamespace(pos=SimpleNamespace(device=SimpleNamespace(type="cuda"))))
    result = probe.stop()
    assert result["vram_probe_error"]
    assert result["peak_vram_allocated_bytes"] is None


def test_repeated_metrics_do_not_change_output_or_keep_prompt() -> None:
    app = App(Engine(), Tokenizer(), "test")
    ids = set()
    for prompt in ([1], [99991, 100003], [17, 19, 23]):
        events = list(app.run(prompt))
        assert "".join(value for kind, value in events if kind == "text") == "OK"
        final = events[-1][1]
        assert final["metrics"]["input_tokens"] == len(prompt)
        ids.add(final["metrics"]["generation_id"])
    assert len(ids) == 3
    assert app.metrics.snapshot()["counts"] == {"completed": 3}


def test_generation_error_response_contains_its_own_metrics() -> None:
    app = App(Engine(fail=True), Tokenizer(), "test")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.request(
                "POST", "/v1/completions", json.dumps({"prompt": "secret", "stream": False})
            )
            response = conn.getresponse()
            payload = json.loads(response.read())
            assert response.status == 400
            assert payload["ramvi_metrics"]["outcome"] == "error"
            assert (
                payload["ramvi_metrics"]["generation_id"]
                == app.metrics.snapshot()["last"]["generation_id"]
            )
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
    assert not any(t.name == "ramvi-rss" for t in threading.enumerate())
