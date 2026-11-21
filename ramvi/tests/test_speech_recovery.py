"""Failure paths must release resources, preserve text, and recover only explicitly."""

import json
import threading
from typing import Any

import pytest
from ramvi_service.contracts.modules import Authority
from ramvi_service.contracts.speech import SPEAKER, AudioPiece
from ramvi_service.resources import ResourceBroker
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.service import SpeechService
from test_character_dialogue import Services, active, finish
from test_character_response import records
from test_speech import Output, Provider, wait


@pytest.mark.parametrize(
    "fault,code",
    [
        ("synth", "speech.provider_failed"),
        ("first_timeout", "speech.provider_timeout"),
        ("idle_timeout", "speech.audio_timeout"),
        ("open", "speech.device_unavailable"),
        ("start", "speech.device_unavailable"),
        ("write", "speech.device_lost"),
        ("close", "speech.cleanup_failed"),
        ("provider_close", "speech.cleanup_failed"),
        ("gate", "speech.device_unavailable"),
        ("gate_close", "speech.cleanup_failed"),
        ("http", "speech.api_http_503"),
    ],
)
def test_fault_cleanup_and_next_job(fault: Any, code: Any, tmp_path: Any) -> None:
    authority = Authority()
    owner = authority.issue("character", "user", "recovery")
    broker = ResourceBroker(authority, {SPEAKER: 1})
    outputs = []
    providers = []
    gates = []

    class Backend(Provider):
        bad = True

        def synthesize(self, request: Any, cancel: Any) -> Any:
            if self.bad and fault == "synth":
                raise OSError("private input or credential must not escape")
            if self.bad and fault == "http":
                raise RuntimeError("speech.api_http_503")
            if self.bad and fault == "first_timeout":
                self.stopped.wait(2)
            if not cancel.is_set():
                yield AudioPiece(b"\1\0" * 320, 16000)
            if self.bad and fault == "idle_timeout":
                self.stopped.wait(2)

        def close(self) -> Any:
            super().close()
            if self.bad and fault == "provider_close":
                raise OSError("private cleanup text")

    class Device(Output):
        bad = True

        def start(self) -> Any:
            if self.bad and fault == "start":
                raise OSError("driver name private")
            super().start()

        def write(self, pcm: Any) -> Any:
            if self.bad and fault == "write":
                raise OSError("driver lost")
            super().write(pcm)

        def close(self) -> Any:
            super().close()
            if self.bad and fault == "close":
                raise OSError("close failed")

    class Input:
        bad = True

        def session_playback(self, active: Any, *, owner: Any) -> Any:
            gates.append(active)
            if self.bad and (
                (fault == "gate" and active) or (fault == "gate_close" and not active)
            ):
                raise OSError("gate failed")

    healthy = False
    input = Input()

    def backend(info: Any, key: Any) -> Any:
        p = Backend()
        p.bad = not healthy
        providers.append(p)
        return p

    def output(rate: Any) -> Any:
        if not healthy and fault == "open":
            raise OSError("no output")
        o = Device(rate)
        o.bad = not healthy
        outputs.append(o)
        return o

    s = SpeechService(broker, input, backend, output, first_audio_timeout=0.06, audio_timeout=0.06)
    data = ScopedData(tmp_path, "character")
    secret = ScopedSecrets(MemorySecrets(), "character")
    try:
        s.configure(owner, data, {"enabled": True})
        s.advance(owner, 1, "bad")
        bad = s.start(
            owner,
            data,
            secret,
            {"generation": 1, "response_id": "bad", "request": {"text": "Fresh 104729"}},
        )["job_id"]
        wait(lambda: s.status(owner, bad)["job"]["settled"])
        packet = s.status(owner, bad)["job"]
        assert packet["state"] == "error" and packet["error"] == code
        assert broker.status()["leases"] == 0 and providers[0].closed
        assert all(o.closed for o in outputs)
        assert not any(j["producer"].is_alive() for j in s.jobs.values())
        healthy = True
        input.bad = False
        s.advance(owner, 2, "good")
        good = s.start(
            owner,
            data,
            secret,
            {"generation": 2, "response_id": "good", "request": {"text": "Fresh 393241"}},
        )["job_id"]
        wait(lambda: s.status(owner, good)["job"]["settled"])
        assert s.status(owner, good)["job"]["state"] == "completed"
        assert outputs[-1].pcm and outputs[-1].closed and broker.status()["leases"] == 0
    finally:
        s.close()


class Connected(Services):
    def __init__(self, core: Any, data: Any, secret: Any) -> None:
        super().__init__()
        self.core, self.disk, self.secret = core, data, secret

    def call(self, method: Any, payload: Any) -> Any:
        s = self.core
        if method == "speech.catalog":
            return s.catalog(self.disk, payload.get("refresh", False))
        if method == "speech.configure":
            return s.configure("owner", self.disk, payload["settings"])
        if method == "speech.advance":
            return s.advance("owner", payload["generation"], payload["response_id"])
        if method == "speech.start":
            return s.start("owner", self.disk, self.secret, payload)
        if method == "speech.status":
            return s.status("owner", payload.get("job_id"))
        if method == "speech.cancel":
            return s.cancel("owner", payload.get("job_id"))
        return super().call(method, payload)


def test_failed_audio_preserves_entire_text_and_resume_never_replays_current(tmp_path: Any) -> None:
    providers = []
    outputs = []
    healthy = False

    class Backend(Provider):
        def synthesize(self, request: Any, cancel: Any) -> Any:
            self.requests.append(request)
            if not healthy:
                raise OSError("private text")
            yield AudioPiece(b"\1\0" * 320, 16000)

    def factory(info: Any, key: Any) -> Any:
        p = Backend()
        providers.append(p)
        return p

    def device(rate: Any) -> Any:
        o = Output(rate)
        outputs.append(o)
        return o

    core = SpeechService(provider_factory=factory, output_factory=device)
    svc = Connected(
        core, ScopedData(tmp_path, "character"), ScopedSecrets(MemorySecrets(), "character")
    )
    m = active(svc)
    rows = records()
    parts = [
        {**rows[1], "id": f"s{i}", "seq": i, "text": f"Fresh text {i} 104729"} for i in range(1, 9)
    ]
    svc.output = "\n".join(json.dumps(r) for r in [rows[0], *parts, {**rows[-1], "last_seq": 8}])
    try:
        m.speech("configure", {"settings": {"enabled": True}})
        reply = m.dialogue("start", {"text": "First", "model": "one"})
        result = finish(m)
        wait(lambda: m._speech.view()["mode"] == "text")
        assert result["job"]["state"] == "completed" and all(
            r["text"] in result["history"][-1]["assistant"] for r in parts
        )
        assert m._speech.current(reply["response_id"], reply["generation"])
        count = sum(len(p.requests) for p in providers)
        assert count == 1 and not m._speech.queue and not outputs
        healthy = True
        m.speech("resume")
        assert m._speech.current(reply["response_id"], reply["generation"])
        assert not m._speech.offer(reply["response_id"], parts[0], reply["generation"])
        assert sum(len(p.requests) for p in providers) == count
        svc.output = "\n".join(
            json.dumps(r) for r in [rows[0], {**rows[1], "text": "Recovered 393241"}, rows[-1]]
        )
        m.dialogue("start", {"text": "Second", "model": "one"})
        finish(m)
        wait(lambda: bool(outputs) and outputs[-1].closed)
        assert m._speech.thread is not None
        assert m._speech.thread.is_alive() and core.status("owner")["job"]["state"] == "completed"
        m.speech("configure", {"settings": {"enabled": False}})
        assert not m._speech.armed and core.status("owner")["job"]["settled"]
    finally:
        m.dispose()
        core.close()
    assert m._speech.thread is not None
    assert not m._speech.thread.is_alive()


def test_inventory_failure_does_not_hold_job_lock_and_refresh_recovers(tmp_path: Any) -> None:
    release = threading.Event()
    entered = threading.Event()
    attempts = 0

    class Inventory(Provider):
        def voices(self) -> Any:
            nonlocal attempts
            attempts += 1
            entered.set()
            release.wait(2)
            if attempts == 1:
                raise OSError("private inventory data")
            return super().voices()

    def factory(info: Any, key: Any) -> Any:
        return Inventory()

    core = SpeechService(provider_factory=factory)
    data = ScopedData(tmp_path, "character")
    try:
        result = core.catalog(data)
        assert result["local_voices"]["state"] == "loading"
        assert entered.wait(1)
        # These calls complete while voice discovery is still blocked.
        assert core.status("someone")["job"] is None
        core.advance("someone", 1, None)
        release.set()
        wait(lambda: core.catalog(data)["local_voices"]["state"] == "error")
        core.catalog(data, refresh=True)
        wait(lambda: core.catalog(data)["local_voices"]["state"] == "ready")
        assert core.catalog(data)["providers"][0]["voices"][0]["id"] == "local"
    finally:
        release.set()
        core.close()
    assert core.inventory_thread is not None
    assert not core.inventory_thread.is_alive()


def test_output_timeout_terminates_owned_worker() -> None:
    from ramvi_service.speech.output import SoundOutput

    class Worker:
        stopped = False

        def start(self) -> Any:
            pass

        def request(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("private driver detail")

        def stop(self, **kwargs: Any) -> Any:
            self.stopped = True

    worker = Worker()
    output = SoundOutput(16000, worker=worker, timeout=0.01)
    with pytest.raises(TimeoutError, match="^speech.device_timeout$"):
        output.start()
    assert worker.stopped
    with pytest.raises(InterruptedError):
        output.write(b"\1\0")
    output.close()


def test_old_failure_cannot_disable_new_voice_selection() -> None:
    from ramvi_character.speech import SpeechOutput
    from test_character_speech import SpeechServices

    svc = SpeechServices()
    output = SpeechOutput(svc)
    try:
        output.configure({"enabled": True})
        old = output.audio_generation
        output.configure({"enabled": True})
        new = output.audio_generation
        output._degrade("speech.device_lost", old)
        assert output.armed and output.error is None and output.audio_generation == new
    finally:
        output.close()


def test_voice_switch_and_preview_keep_text_generation_valid(tmp_path: Any) -> None:
    providers = []

    def factory(info: Any, key: Any) -> Any:
        p = Provider()
        providers.append(p)
        return p

    core = SpeechService(provider_factory=factory, output_factory=Output)
    svc = Connected(
        core, ScopedData(tmp_path, "character"), ScopedSecrets(MemorySecrets(), "character")
    )
    m = active(svc)
    try:
        m.speech("configure", {"settings": {"enabled": True, "voice": "first"}})
        reply = m.dialogue("start", {"text": "Fresh 104729", "model": "one"})
        finish(m)
        m.speech("configure", {"settings": {"voice": "second"}})
        assert m._speech.current(reply["response_id"], reply["generation"])
        m.speech("preview")
        wait(lambda: any(p.requests and p.requests[0].voice == "second" for p in providers))
        assert m._speech.current(reply["response_id"], reply["generation"])
        m.speech("stop")
        assert m._speech.current(reply["response_id"], reply["generation"]) and not m._speech.armed
    finally:
        m.dispose()
        core.close()


def test_failed_owner_does_not_interrupt_another_module(tmp_path: Any) -> None:
    import time

    authority = Authority()
    other = authority.issue("other", "user", "a")
    failed = authority.issue("character", "user", "b")
    broker = ResourceBroker(authority, {SPEAKER: 1})
    outputs = []

    class Backend(Provider):
        def synthesize(self, request: Any, cancel: Any) -> Any:
            if request.text == "bad":
                raise OSError("provider failed")
            for _ in range(100):
                if cancel.is_set():
                    return
                yield AudioPiece(b"\1\0" * 320, 16000)

    class Playing(Output):
        writes = 0

        def write(self, pcm: Any) -> Any:
            time.sleep(0.01)
            self.writes += 1
            super().write(pcm)

    def device(rate: Any) -> Any:
        o = Playing(rate)
        outputs.append(o)
        return o

    core = SpeechService(broker, provider_factory=lambda *_: Backend(), output_factory=device)
    a = ScopedData(tmp_path, "other")
    b = ScopedData(tmp_path, "character")
    secret = ScopedSecrets(MemorySecrets(), "other")
    try:
        core.configure(other, a, {"enabled": True})
        core.configure(failed, b, {"enabled": True})
        running = core.start(other, a, secret, {"request": {"text": "healthy"}})["job_id"]
        wait(lambda: bool(outputs) and outputs[0].writes >= 2)
        bad = core.start(failed, b, secret, {"request": {"text": "bad"}})["job_id"]
        wait(lambda: core.status(failed, bad)["job"]["settled"])
        count = outputs[0].writes
        wait(lambda: outputs[0].writes > count)
        assert not outputs[0].aborted and broker.status()["leases"] == 1
        core.release(failed)
        assert core.status(other, running)["job"]["state"] == "playing"
        core.cancel(other, running)
        wait(lambda: core.status(other, running)["job"]["settled"])
        assert broker.status()["leases"] == 0
    finally:
        core.close()


def test_provider_switch_requires_explicit_remote_text_permission_again(tmp_path: Any) -> None:
    core = SpeechService(provider_factory=lambda *_: Provider(), output_factory=Output)
    data = ScopedData(tmp_path, "character")
    secret = ScopedSecrets(MemorySecrets(), "character")
    try:
        core.configure(
            "owner", data, {"enabled": True, "provider": "openai", "remote_text_enabled": True}
        )
        result = core.configure("owner", data, {"provider": "gemini"})
        assert not result["settings"]["remote_text_enabled"]
        with pytest.raises(PermissionError, match="speech.remote_text_disabled"):
            core.start("owner", data, secret, {"request": {"text": "must stay local"}})
        assert not core.jobs
    finally:
        core.close()


def test_late_failed_job_settlement_cannot_disarm_new_preview() -> None:
    from ramvi_character.speech import SpeechOutput
    from test_character_speech import SpeechServices

    class Svc(SpeechServices):
        def call(self, method: Any, payload: Any) -> Any:
            result = super().call(method, payload)
            if method == "speech.status" and payload["job_id"] == "1":
                result["job"].update(state="error", error="speech.device_lost")
            return result

    svc = Svc()
    speech = SpeechOutput(svc)
    degrade = speech._degrade

    def race(error: BaseException | str, expected_generation: int | None = None) -> None:
        degrade(error, expected_generation)
        speech.preview()  # explicit UI action between failure and old settlement

    speech._degrade = race  # type: ignore[method-assign]
    try:
        speech.preview()
        wait(lambda: len(svc.starts) == 2)
        assert svc.starts[-1]["preview"]
        wait(lambda: not speech.armed)
    finally:
        speech.close()
