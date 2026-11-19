"""Adversarial ordering at generation, PCM and device boundaries. No paid providers."""

import threading

import pytest
from ramvi_service.contracts.speech import AudioPiece
from ramvi_service.speech.buffer import PCM_BLOCKS
from ramvi_service.speech.service import SpeechService
from test_speech import Output, Provider, rig, wait

__all__ = ["rig"]


def fenced_start(s, d, k, generation, response, segment="s1", text="Fresh 900731"):
    return s.start(
        "owner",
        d,
        k,
        {
            "generation": generation,
            "response_id": response,
            "segment_id": segment,
            "request": {"text": text},
        },
    )["job_id"]


def test_generation_rejects_delayed_start_and_duplicate_never_replays(rig):
    s, d, k, providers, outputs = rig
    s.configure("owner", d, {"enabled": True})
    s.advance("owner", 1, "first")
    first = fenced_start(s, d, k, 1, "first")
    wait(lambda: s.status("owner", first)["job"]["settled"])
    duplicate = fenced_start(s, d, k, 1, "first")
    assert duplicate == first and len(outputs) == len(providers) == 1
    with pytest.raises(ValueError, match="segment_conflict"):
        fenced_start(s, d, k, 1, "first", text="different")
    s.advance("owner", 2, "new")
    with pytest.raises(RuntimeError, match="stale_generation"):
        fenced_start(s, d, k, 1, "first")
    with pytest.raises(ValueError, match="generation_required"):
        s.start("owner", d, k, {"request": {"text": "unfenced"}})
    with pytest.raises(RuntimeError, match="stale_generation"):
        s.advance("owner", 1, "late")
    current = fenced_start(s, d, k, 2, "new", text="New 700021")
    wait(lambda: s.status("owner", current)["job"]["settled"])
    assert len(outputs) == 2 and len(s.jobs) == 1


def test_fence_while_output_constructor_is_late_cannot_start_old_device(rig):
    s, d, k, _, outputs = rig
    entered, release = threading.Event(), threading.Event()

    def delayed(rate):
        entered.set()
        assert release.wait(3)
        output = Output(rate)
        outputs.append(output)
        return output

    s.output_factory = delayed
    s.configure("owner", d, {"enabled": True})
    s.advance("owner", 1, "old")
    old = fenced_start(s, d, k, 1, "old")
    try:
        assert entered.wait(2)
        s.advance("owner", 2, "new")
        release.set()
        wait(lambda: s.status("owner", old)["job"]["settled"])
        assert not getattr(outputs[0], "started", False) and outputs[0].pcm == b""
        new = fenced_start(s, d, k, 2, "new")
        wait(lambda: s.status("owner", new)["job"]["settled"])
        assert outputs[1].started and outputs[1].pcm
    finally:
        release.set()


def test_slow_device_applies_pcm_backpressure_and_fence_discards_buffer(rig):
    s, d, k, providers, outputs = rig
    blocked, release = threading.Event(), threading.Event()

    class Streaming(Provider):
        count = 0

        def synthesize(self, request, cancel):
            for _ in range(500):
                self.count += 1
                yield AudioPiece(bytes(640), 16000)

    class Slow(Output):
        def write(self, pcm):
            assert len(pcm) <= 640
            blocked.set()
            release.wait(0.2)
            super().write(pcm)

    def provider(info, key):
        p = Streaming()
        providers.append(p)
        return p

    def output(rate):
        o = Slow(rate)
        outputs.append(o)
        return o

    s.provider_factory = provider
    s.output_factory = output
    s.configure("owner", d, {"enabled": True})
    s.advance("owner", 1, "old")
    old = fenced_start(s, d, k, 1, "old")
    try:
        assert blocked.wait(2)
        wait(lambda: s.status("owner", old)["job"]["pcm_buffer"]["high_water"] == PCM_BLOCKS)
        assert providers[0].count <= PCM_BLOCKS + 2  # queued + one write + producer waiting
        s.advance("owner", 2, "new")
        at_fence = len(outputs[0].pcm)
        release.set()
        wait(lambda: s.status("owner", old)["job"]["settled"])
        assert len(outputs[0].pcm) == at_fence and outputs[0].aborted
        assert s.status("owner", old)["job"]["pcm_buffer"]["high_water"] <= PCM_BLOCKS
    finally:
        release.set()


def test_fence_is_scoped_and_wakes_resource_waiter(tmp_path):
    from ramvi_service.contracts.modules import Authority
    from ramvi_service.contracts.speech import SPEAKER
    from ramvi_service.resources import ResourceBroker
    from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets

    authority = Authority()
    a = authority.issue("first", "user", "a")
    b = authority.issue("second", "user", "b")
    broker = ResourceBroker(authority, {SPEAKER: 1})
    lease = broker.acquire(b, {SPEAKER: 1})
    outputs = []

    def output(rate):
        o = Output(rate)
        outputs.append(o)
        return o

    s = SpeechService(broker, provider_factory=lambda *_: Provider(), output_factory=output)
    data = ScopedData(tmp_path, "first")
    secrets = ScopedSecrets(MemorySecrets(), "first")
    try:
        s.configure(a, data, {"enabled": True})
        s.advance(a, 1, "old")
        job = s.start(
            a,
            data,
            secrets,
            {"generation": 1, "response_id": "old", "request": {"text": "waiting"}},
        )["job_id"]
        wait(lambda: broker.status()["queued"] == 1)
        s.advance(a, 2, None)
        wait(lambda: s.status(a, job)["job"]["settled"])
        assert not outputs and broker.status()["leases"] == 1
        broker.release(b, lease)
    finally:
        s.close()


def test_dialogue_backpressure_is_cancellable_and_next_response_gets_first_slot(rig):
    import json

    from ramvi_character.speech import SEGMENT_QUEUE
    from test_character_dialogue import Services, active, finish
    from test_character_response import records

    s, d, k, providers, outputs = rig

    class Blocking(Provider):
        def synthesize(self, request, cancel):
            self.stopped.wait(3)
            if not cancel.is_set():
                yield AudioPiece(bytes(640), 16000)

    original = s.provider_factory
    first = True

    def factory(info, key):
        nonlocal first
        if first:
            first = False
            p = Blocking()
            providers.append(p)
            return p
        return original(info, key)

    s.provider_factory = factory

    class Connected(Services):
        def call(self, method, payload):
            if method == "speech.catalog":
                return s.catalog(d)
            if method == "speech.configure":
                return s.configure("owner", d, payload["settings"])
            if method == "speech.advance":
                return s.advance("owner", payload["generation"], payload["response_id"])
            if method == "speech.start":
                return s.start("owner", d, k, payload)
            if method == "speech.status":
                return s.status("owner", payload.get("job_id"))
            if method == "speech.cancel":
                return s.cancel("owner", payload.get("job_id"))
            return super().call(method, payload)

    svc = Connected()
    rows = records()
    segments = [
        {**rows[1], "id": f"s{i}", "seq": i, "text": f"Old segment {i} 783041"} for i in range(1, 9)
    ]
    svc.output = "\n".join(json.dumps(x) for x in [rows[0], *segments, {**rows[-1], "last_seq": 8}])
    m = active(svc)
    try:
        m.speech("configure", {"settings": {"enabled": True}})
        m.dialogue("start", {"text": "Old", "model": "one"})
        wait(lambda: m._speech.backpressure_waits > 0)
        assert len(m._speech.queue) == SEGMENT_QUEUE
        assert m.dialogue("poll")["job"]["state"] == "running"
        result = m.dialogue("cancel")
        assert "error" not in result
        finish(m)
        assert not m._speech.queue and not outputs
        svc.output = "\n".join(
            json.dumps(x) for x in [rows[0], {**rows[1], "text": "Fresh 131071"}, rows[-1]]
        )
        m.dialogue("start", {"text": "New", "model": "one"})
        finish(m)
        wait(lambda: len(outputs) == 1 and outputs[0].closed)
        assert s.status("owner")["job"]["state"] == "completed"
        assert svc.cancelled
    finally:
        m.dispose()


def test_first_validated_segment_plays_before_llm_finishes(rig):
    import json

    from test_character_dialogue import Services, active, finish
    from test_character_response import records

    s, d, k, _, outputs = rig
    release = threading.Event()
    rows = records()

    class Connected(Services):
        reads = 0

        def call(self, method, payload):
            if method == "speech.catalog":
                return s.catalog(d)
            if method == "speech.configure":
                return s.configure("owner", d, payload["settings"])
            if method == "speech.advance":
                return s.advance("owner", payload["generation"], payload["response_id"])
            if method == "speech.start":
                return s.start("owner", d, k, payload)
            if method == "speech.status":
                return s.status("owner", payload.get("job_id"))
            if method == "speech.cancel":
                return s.cancel("owner", payload.get("job_id"))
            if method == "llm.read":
                self.reads += 1
                if self.reads == 1:
                    return {
                        "done": False,
                        "chunks": [
                            {
                                "type": "text",
                                "text": "\n".join(json.dumps(r) for r in rows[:2]) + "\n",
                            }
                        ],
                    }
                if self.reads == 2:
                    assert release.wait(3)
                    return {
                        "done": False,
                        "chunks": [
                            {"type": "text", "text": json.dumps(rows[-1]) + "\n"},
                            {"type": "finish", "finish_reason": "stop"},
                        ],
                    }
                return {"done": True, "chunks": []}
            return super().call(method, payload)

    svc = Connected()
    m = active(svc)
    try:
        m.speech("configure", {"settings": {"enabled": True}})
        m.dialogue("start", {"text": "Fresh 104729", "model": "one"})
        wait(lambda: outputs and outputs[0].closed)
        assert outputs[0].pcm and m.dialogue("poll")["job"]["state"] == "running"
        release.set()
        assert finish(m)["job"]["state"] == "completed"
    finally:
        release.set()
        m.dispose()


def test_sound_device_construction_cannot_start_playback(monkeypatch):
    import sys
    from types import SimpleNamespace

    from ramvi_service.speech.device_worker import DeviceOutput

    calls = []

    class Stream:
        def __init__(self, **kwargs):
            calls.append("open")

        def start(self):
            calls.append("start")

        def write(self, pcm):
            calls.append("write")

        def stop(self):
            calls.append("stop")

        def close(self):
            calls.append("close")

        def abort(self):
            calls.append("abort")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawOutputStream=Stream))
    monkeypatch.setattr("ramvi_service.voice.audio._com_init", lambda: None)
    output = DeviceOutput(16000)
    assert calls == ["open"]
    output.start()
    output.write(bytes(640))
    output.cancel()
    output.close()
    assert calls == ["open", "start", "write", "abort", "stop", "close"]
