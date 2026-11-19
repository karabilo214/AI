"""Character voice policy: explicit start, partials are advisory, late finals discarded."""

import threading
from typing import Any

from ramvi_character.module import CharacterModule
from test_character_dialogue import Services, active, finish
from test_voice import wait


class VoiceServices(Services):
    def __init__(self) -> None:
        super().__init__()
        self.starts = self.stops = 0
        self.events: list[dict[str, Any]] = []
        self.guard = threading.Lock()
        self.capture = "capture-one"

    def push(self, kind: str, text: str = "", identifier: str = "utterance-one") -> None:
        with self.guard:
            self.events.append(
                {
                    "seq": len(self.events) + 1,
                    "session_id": self.capture,
                    "type": kind,
                    "utterance_id": identifier,
                    "text": text,
                    "language": "ru",
                }
            )

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "voice.capabilities":
            return {
                "capabilities": {"languages": ["auto", "ru", "en"], "session": {"supported": True}}
            }
        if method == "voice.session.start":
            self.starts += 1
            return {"session_id": self.capture, "state": "listening"}
        if method == "voice.session.cancel":
            self.stops += 1
            return {"state": "stopped"}
        if method == "voice.session.read":
            with self.guard:
                return {
                    "session_id": self.capture,
                    "state": "listening",
                    "cursor": len(self.events),
                    "events": [e.copy() for e in self.events if e["seq"] > payload["cursor"]],
                }
        return super().call(method, payload)


def stop(module: CharacterModule) -> None:
    module.voice("stop")
    assert module._voice.thread is not None
    module._voice.thread.join(timeout=3)
    assert not module._voice.thread.is_alive()


def test_voice_does_not_activate_with_text_or_checkpoint_and_partials_never_send() -> None:
    svc = VoiceServices()
    module = active(svc)
    module.configure("", "one")
    assert svc.starts == 0 and module.voice()["state"] == "inactive"
    assert module.voice("capabilities")["available"]
    assert svc.starts == 0
    module.voice("start")
    wait(lambda: svc.starts == 1)
    svc.push("speech_start")
    svc.push("partial", "Это ещё не готовый текст")
    wait(lambda: module.voice()["partial"] != "")
    assert svc.requests == []
    svc.push("speech_end")
    svc.push("final", "Привет")
    wait(lambda: module.voice()["reply"] is not None)
    result = finish(module)
    assert result["history"][0]["user"] == "Привет" and len(svc.requests) == 1
    stop(module)
    assert svc.stops == 1 and not module.status()["devices_open"]
    checkpoint = module.checkpoint()
    module.dispose()
    restored = CharacterModule()
    restored.initialize({"services": svc, "checkpoint": checkpoint})
    restored.activate()
    assert restored.voice()["state"] == "inactive" and svc.starts == 1
    restored.dispose()


def test_new_speech_cancels_text_generation_and_waits_before_sending_final() -> None:
    svc = VoiceServices()
    module = active(svc)
    svc.block.clear()
    module.dialogue("start", {"text": "old prompt", "model": "one"})
    wait(lambda: len(svc.requests) == 1)
    module.voice("start")
    wait(lambda: svc.starts == 1)
    svc.push("speech_start")
    wait(lambda: module._dialogue.cancelled.is_set())
    svc.block.set()
    finish(module)
    svc.push("speech_end")
    svc.push("final", "новая просьба")
    wait(lambda: len(svc.requests) == 2)
    result = finish(module)
    assert result["history"][0]["status"] == "cancelled"
    assert result["history"][1]["user"] == "новая просьба"
    stop(module)
    module.dispose()


def test_stop_ignores_queued_final_and_suspend_releases_only_own_session() -> None:
    svc = VoiceServices()
    module = active(svc)
    module.configure("", "one")
    module.voice("start")
    wait(lambda: svc.starts == 1)
    module.voice("stop")
    svc.push("final", "late speech")
    stop(module)
    assert not svc.requests
    module.voice("start")
    wait(lambda: svc.starts == 2)
    module.suspend()
    assert module.voice()["code"] == "character.voice_error"
    assert not module.status()["devices_open"]
    assert svc.stops == 2
    module.dispose()


def test_real_character_worker_session_uses_scoped_voice_and_suspend_releases_mic(
    tmp_path: Any,
) -> None:
    from pathlib import Path

    from ramvi_service.modules.host import ModuleHost
    from ramvi_service.sections import SectionRegistry
    from test_voice import broker_env, tone
    from test_voice_session import Detector

    env, authority, broker, chat_owner = broker_env(tmp_path)
    host = ModuleHost(tmp_path / "data", authority=authority)
    env.voice.resources = host.resources
    env.voice.make_detector = Detector
    host.voice = env.voice
    registry = SectionRegistry(host)
    registry.discover(Path(__file__).resolve().parents[1] / "modules")
    try:
        assert not env.mic.opened
        assert registry.open("character")["state"] == "active"
        assert not env.mic.opened
        host.invoke("character", "configure", {"draft": "", "model": "test-model"})
        assert registry.command("character", "voice", {"operation": "start"})["state"] == "starting"
        wait(lambda: env.voice.status()["state"] == "session")
        assert env.mic.stream is not None and not env.mic.stream.closed
        env.mic.say(tone(0.5))
        wait(
            lambda: (
                registry.command("character", "voice", {"operation": "status"})["state"]
                == "hearing"
            )
        )
        host.suspend("character")
        assert env.voice.status()["state"] == "idle" and env.mic.stream.closed
        assert host.resources.status()["leases"] == 0
        host.resume("character")
        assert (
            registry.command("character", "voice", {"operation": "status"})["state"] == "inactive"
        )
        assert len(env.mic.opened) == 1  # no hidden resumption
        env.voice.start()
        env.mic.say(tone(0.5))
        assert env.voice.stop()["text"] == "привет мир"  # ordinary Chat is still independent
    finally:
        host.close()
        env.voice.close()


def test_lost_owner_heartbeat_ends_microphone_session() -> None:
    import time

    svc = VoiceServices()
    module = active(svc)
    module.configure("", "one")
    module.voice("start")
    wait(lambda: svc.starts == 1)
    module._voice.last_contact = time.monotonic() - 6
    wait(lambda: module._voice.state == "error")
    assert svc.stops == 1
    assert module._voice.error == "voice.session_heartbeat_lost"
    assert not module.status()["devices_open"]
    module.dispose()


def test_new_utterance_discards_late_final_from_previous_speech() -> None:
    svc = VoiceServices()
    module = active(svc)
    module.configure("", "one")
    try:
        module.voice("start")
        wait(lambda: svc.starts == 1)
        svc.push("speech_start", identifier="old")
        svc.push("speech_end", identifier="old")
        svc.push("speech_start", identifier="new")
        svc.push("final", "Late 783041", identifier="old")
        svc.push("speech_end", identifier="new")
        svc.push("final", "Fresh 131071", identifier="new")
        wait(lambda: module.voice()["reply"] is not None)
        result = finish(module)
        assert result["history"][0]["user"] == "Fresh 131071"
        assert len(svc.requests) == 1
    finally:
        module.dispose()
