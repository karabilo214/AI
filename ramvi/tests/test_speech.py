"""Output contracts, independent egress/secrets, owner isolation and bounded playback."""

import base64
import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from ramvi_service.contracts.speech import AudioPiece, SpeechRequest
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.cloud import CloudSpeech, validate_alignment
from ramvi_service.speech.service import SpeechService


def wait(predicate: Callable[[], bool]) -> None:
    until = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < until
        time.sleep(0.01)


class Provider:
    def __init__(self):
        self.requests = []
        self.closed = False
        self.stopped = threading.Event()
        self.block = False

    def voices(self):
        return [{"id": "local", "name": "Local", "languages": ["ru"]}]

    def synthesize(self, request, cancel):
        self.requests.append(request)
        yield AudioPiece(b"\0" * (640 if self.block else 3), 16000)
        if self.block:
            self.stopped.wait(3)
        if not cancel.is_set():
            yield AudioPiece(b"\0" * 5, 16000)

    def cancel(self):
        self.stopped.set()

    def close(self):
        self.closed = True
        self.cancel()


class Output:
    def __init__(self, rate):
        self.rate, self.pcm, self.closed, self.aborted = rate, b"", False, False

    def start(self):
        self.started = True

    def write(self, pcm):
        assert len(pcm) % 2 == 0
        self.pcm += pcm

    def close(self):
        self.closed = True

    def cancel(self):
        self.aborted = True


@pytest.fixture
def rig(tmp_path):
    providers, outputs = [], []

    def factory(info, key):
        p = Provider()
        providers.append(p)
        return p

    def output(rate):
        o = Output(rate)
        outputs.append(o)
        return o

    service = SpeechService(provider_factory=factory, output_factory=output)
    data = ScopedData(tmp_path, "character")
    secrets = ScopedSecrets(MemorySecrets(), "character")
    yield service, data, secrets, providers, outputs
    service.close()


def start(service, data, secrets, **extra):
    return service.start(
        "owner",
        data,
        secrets,
        {
            "request": {"text": "Свежий ответ 900731", "emotion": "joy", "style": "cheerful"},
            **extra,
        },
    )["job_id"]


def test_defaults_and_remote_egress_before_credentials_or_provider(rig):
    s, d, k, p, o = rig
    with pytest.raises(RuntimeError, match="disabled"):
        start(s, d, k)
    s.configure("owner", d, {"enabled": True, "provider": "openai"})
    with pytest.raises(PermissionError, match="remote_text_disabled"):
        start(s, d, k)
    assert not p and not o
    s.configure("owner", d, {"remote_text_enabled": True})
    with pytest.raises(RuntimeError, match="key_missing"):
        start(s, d, k)
    k.backend.set("voice", "api_key", "STT-only")
    with pytest.raises(RuntimeError, match="key_missing"):
        start(s, d, k)
    assert s.key(k, "openai", "fake-tts-key") == {"key_set": True}
    assert k.backend.get("voice", "api_key") == "STT-only"
    assert k.get("tts_openai") == "fake-tts-key"
    assert "fake-tts-key" not in json.dumps(s.catalog(d))
    assert "fake-tts-key" not in (d.root / "speech.json").read_text()
    s.key(k, "openai", "")
    assert k.get("tts_openai") is None


def test_local_fallback_pcm_alignment_owner_and_cleanup(rig):
    s, d, k, p, o = rig
    s.configure("owner", d, {"enabled": True})
    identifier = start(s, d, k)
    wait(lambda: s.status("owner", identifier)["job"]["settled"])
    job = s.status("owner", identifier)["job"]
    assert job["state"] == "completed" and job["bytes"] == 8
    assert job["metadata"]["fallback"] and job["metadata"]["emotion_control"] == "provider_default"
    assert p[0].requests[0].style == "" and p[0].requests[0].emotion == "joy"
    assert o[0].pcm == b"\0" * 8 and o[0].closed and p[0].closed
    with pytest.raises(PermissionError):
        s.status("stranger", identifier)
    with pytest.raises(PermissionError):
        s.cancel("stranger", identifier)
    s.release("owner")
    assert s.status("owner")["job"] is None


def test_preview_without_enabling_and_cancel_own_job(rig):
    s, d, k, p, o = rig
    original = s.provider_factory

    def blocked(info, key):
        obj = original(info, key)
        obj.block = True
        return obj

    s.provider_factory = blocked
    identifier = start(s, d, k, preview=True)
    wait(lambda: len(o) == 1)
    s.cancel("owner", identifier)
    wait(lambda: s.status("owner", identifier)["job"]["settled"])
    assert s.status("owner", identifier)["job"]["state"] == "cancelled"
    assert o[0].aborted and o[0].closed and not s.settings(d)["enabled"]


@pytest.mark.parametrize(
    "fields",
    [
        {"text": ""},
        {"text": "x" * 2049},
        {"text": "x", "intensity": float("nan")},
        {"text": "x", "intensity": True},
    ],
)
def test_invalid_request(fields):
    with pytest.raises(ValueError):
        SpeechRequest(**fields)


@pytest.mark.parametrize("provider", ["openai", "gemini", "elevenlabs"])
def test_official_adapter_protocol_on_loopback(provider):
    observed = []
    pcm = b"\0\0" * 120
    alignment = {
        "characters": ["H", "i"],
        "character_start_times_seconds": [0, 0.1],
        "character_end_times_seconds": [0.1, 0.2],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append((self.path, dict(self.headers), body))
            if provider == "gemini":
                # Поток SSE, как у настоящего API: звук кусками в step.delta.
                events = [{"event_type": "interaction.created"}, {"event_type": "step.start"}]
                events += [
                    {
                        "event_type": "step.delta",
                        "delta": {
                            "type": "audio",
                            "sample_rate": 24000,
                            "channels": 1,
                            "data": base64.b64encode(pcm[i : i + 3]).decode(),
                        },
                    }
                    for i in range(0, len(pcm), 3)
                ]
                result = "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                result += b"data: [DONE]\n\n"
            else:
                result = (
                    pcm
                    if provider == "openai"
                    else json.dumps(
                        {"audio_base64": base64.b64encode(pcm).decode(), "alignment": alignment}
                    ).encode()
                )
            self.send_response(200)
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    adapter = CloudSpeech(
        provider, "chosen-model", "fake-key", test_origin=f"http://127.0.0.1:{server.server_port}"
    )
    try:
        result = list(
            adapter.synthesize(
                SpeechRequest("Hi", voice="voice", style="cheerful"), threading.Event()
            )
        )
        assert b"".join(x.pcm for x in result) == pcm and all(x.rate == 24000 for x in result)
        path, headers, body = observed[0]
        if provider == "openai":
            assert (
                path == "/v1/audio/speech"
                and body["instructions"] == "cheerful"
                and headers["Authorization"] == "Bearer fake-key"
            )
        elif provider == "gemini":
            assert (
                path == "/v1beta/interactions?alt=sse" and headers["x-goog-api-key"] == "fake-key"
            )
            assert body["stream"] is True and len(result) > 1  # звук пришёл кусками
            assert body["response_format"]["sample_rate"] == 24000
            assert body["input"][0]["content"][0]["annotations"][0]["style"] == "cheerful"
        else:
            assert "/voice/with-timestamps?" in path and headers["xi-api-key"] == "fake-key"
            assert adapter.timings["source"] == "provider" and adapter.timings["starts"] == [0, 0.1]
        assert body.get("model", body.get("model_id")) == "chosen-model"
    finally:
        adapter.close()
        server.shutdown()
        thread.join(2)
        server.server_close()
    assert not thread.is_alive() and adapter.key == ""


@pytest.mark.parametrize(
    "value",
    [
        {},
        {
            "characters": ["a"],
            "character_start_times_seconds": [float("nan")],
            "character_end_times_seconds": [1],
        },
        {
            "characters": ["a"],
            "character_start_times_seconds": [2],
            "character_end_times_seconds": [1],
        },
    ],
)
def test_bad_timings_rejected(value):
    with pytest.raises(ValueError):
        validate_alignment(value)


def test_real_module_worker_scoped_preview_and_shutdown(tmp_path):
    from pathlib import Path

    from ramvi_service.modules.host import ModuleHost
    from ramvi_service.sections import SectionRegistry

    providers, outputs = [], []

    def factory(info, key):
        p = Provider()
        providers.append(p)
        return p

    def output(rate):
        o = Output(rate)
        outputs.append(o)
        return o

    host = ModuleHost(tmp_path)
    host.speech = SpeechService(host.resources, provider_factory=factory, output_factory=output)
    try:
        registry = SectionRegistry(host)
        registry.discover(Path(__file__).parents[1] / "modules")
        registry.open("character")
        catalog = host.invoke("character", "speech", {"operation": "catalog"})
        assert catalog["available"] and not catalog["settings"]["enabled"]
        host.invoke("character", "speech", {"operation": "preview"})
        wait(lambda: bool(outputs) and outputs[0].closed)
        result = host.invoke("character", "speech", {"operation": "status"})
        assert result["error"] is None
        host.suspend("character")
        assert host.resources.status()["leases"] == 0
        assert all(p.closed for p in providers)
        assert not host.speech.jobs
    finally:
        host.close()
        host.speech.close()


def test_alignment_fits_module_rpc_with_worst_case_unicode_and_timestamps():
    alignment = validate_alignment(
        {
            "characters": ["😀"] * 2048,
            "character_start_times_seconds": [119.123456789] * 2048,
            "character_end_times_seconds": [119.987654321] * 2048,
        }
    )
    assert len(json.dumps({"metadata": {"alignment": alignment}}).encode()) < 60000
    assert alignment["kind"] == "none" and alignment["reason"] == "alignment_size_limit"


def test_cloud_rejects_redirect_without_leaking_response_or_following():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.send_response(307)
            self.send_header("Location", "https://example.org/steal")
            self.end_headers()
            self.wfile.write(b"private text and fake-key")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    adapter = CloudSpeech(
        "openai", "model", "fake-key", test_origin=f"http://127.0.0.1:{server.server_port}"
    )
    try:
        with pytest.raises(RuntimeError, match="^speech.api_http_307$"):
            list(adapter.synthesize(SpeechRequest("private text"), threading.Event()))
        assert calls == ["/v1/audio/speech"]
    finally:
        adapter.close()
        server.shutdown()
        thread.join(2)
        server.server_close()


def test_cancelled_cloud_request_never_posts():
    cancel = threading.Event()
    cancel.set()

    # TCP connection can be opened, but credentials/text must not be sent after cancellation.
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raise AssertionError("cancelled request sent")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    adapter = CloudSpeech(
        "openai", "model", "fake", test_origin=f"http://127.0.0.1:{server.server_port}"
    )
    try:
        with pytest.raises(InterruptedError):
            list(adapter.synthesize(SpeechRequest("fresh"), cancel))
    finally:
        adapter.close()
        server.shutdown()
        thread.join(2)
        server.server_close()


def test_cloud_cancel_interrupts_a_stalled_response_socket():
    started, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(bytes(8192))
            self.wfile.flush()
            started.set()
            release.wait(4)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    adapter = CloudSpeech(
        "openai", "model", "fake", test_origin=f"http://127.0.0.1:{server.server_port}"
    )
    cancel = threading.Event()
    errors = []

    def read():
        try:
            list(adapter.synthesize(SpeechRequest("fresh"), cancel))
        except Exception as error:
            errors.append(type(error).__name__)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert started.wait(2)
        cancel.set()
        adapter.cancel()
        reader.join(2)
        assert not reader.is_alive()
    finally:
        release.set()
        adapter.close()
        reader.join(2)
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_output_gates_only_its_own_microphone_session_and_releases(rig):
    s, d, k, _, outputs = rig

    class Input:
        calls = []

        def session_playback(self, active, *, owner):
            self.calls.append((active, owner))

    s.voice = Input()
    s.configure("owner", d, {"enabled": True})
    identifier = start(s, d, k)
    wait(lambda: s.status("owner", identifier)["job"]["settled"])
    assert s.voice.calls == [(True, "owner"), (False, "owner")]
    assert outputs[0].closed


def test_sse_events_join_lines_across_chunks_and_stop_at_done():
    from ramvi_service.speech.cloud import sse_events

    raw = b'event: a\r\ndata: {"event_type": "step.delta", "n": 1}\r\n\r\ndata: {"n": 2}\n\ndata: [DONE]\n\ndata: {"n": 3}\n'
    chunks = [raw[i : i + 7] for i in range(0, len(raw), 7)]
    assert [e.get("n") for e in sse_events(iter(chunks))] == [1, 2]


def test_next_segment_is_synthesized_ahead_and_plays_after_the_current(tmp_path):
    """Пауза между фразами — не время синтеза: следующая готовится, пока звучит текущая."""
    gate, order, providers = threading.Event(), [], []

    class Slow(Provider):
        def synthesize(self, request, cancel):
            self.requests.append(request)
            order.append(("synth", request.text))
            yield AudioPiece(b"\0" * 640, 16000)
            if request.text == "первая":
                gate.wait(3)  # первая «звучит», пока не отпустим
            yield AudioPiece(b"\0" * 2, 16000)

    class Logged(Output):
        def start(self):
            order.append(("play", len(order)))

    def factory(info, key):
        providers.append(Slow())
        return providers[-1]

    service = SpeechService(provider_factory=factory, output_factory=Logged)
    data = ScopedData(tmp_path, "character")
    secrets = ScopedSecrets(MemorySecrets(), "character")
    try:
        service.configure("owner", data, {"enabled": True})
        service.advance("owner", 1, "r")

        def go(segment, text):
            return service.start(
                "owner",
                data,
                secrets,
                {
                    "generation": 1,
                    "response_id": "r",
                    "segment_id": segment,
                    "request": {"text": text},
                },
            )["job_id"]

        first = go("s1", "первая")
        wait(lambda: service.status("owner", first)["job"]["state"] == "playing")
        second = go("s2", "вторая")  # не speech.busy: одна заранее
        wait(lambda: ("synth", "вторая") in order)
        wait(lambda: service.status("owner", second)["job"]["state"] == "waiting")
        with pytest.raises(RuntimeError, match="busy"):
            go("s3", "третья")  # больше одной в очереди нельзя
        assert service.status("owner")["job"]["id"] == first  # без id — звучащая
        assert sum(1 for x in order if x[0] == "play") == 1
        gate.set()
        wait(lambda: service.status("owner", second)["job"]["settled"])
        assert service.status("owner", second)["job"]["state"] == "completed"
        plays = [i for i, x in enumerate(order) if x[0] == "play"]
        assert len(plays) == 2 and order.index(("synth", "вторая")) < plays[1]
        # Новый ответ отменяет и звучащую, и заранее готовую.
        gate.clear()
        service.advance("owner", 2, "r2")
        a = service.start(
            "owner",
            data,
            secrets,
            {
                "generation": 2,
                "response_id": "r2",
                "segment_id": "s1",
                "request": {"text": "первая"},
            },
        )["job_id"]
        wait(lambda: service.status("owner", a)["job"]["state"] == "playing")
        b = service.start(
            "owner",
            data,
            secrets,
            {
                "generation": 2,
                "response_id": "r2",
                "segment_id": "s2",
                "request": {"text": "вторая"},
            },
        )["job_id"]
        service.advance("owner", 3, "r3")
        wait(
            lambda: (
                service.status("owner", b)["job"]["settled"]
                and service.status("owner", a)["job"]["settled"]
            )
        )
        assert service.status("owner", b)["job"]["state"] == "cancelled"
    finally:
        gate.set()
        service.close()
