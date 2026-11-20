"""Character policy feeds only validated dialogue segments into optional output."""

import threading
from typing import Any

from ramvi_character.speech import SEGMENT_QUEUE, SpeechOutput
from test_character_dialogue import Services, active, finish
from test_speech import wait


class SpeechServices(Services):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self.starts: list[dict[str, Any]] = []
        self.cancels: list[dict[str, Any]] = []

    def call(self, method, payload):
        if method == "speech.catalog":
            return {"settings": {"enabled": self.enabled}}
        if method == "speech.configure":
            self.enabled = payload["settings"]["enabled"]
            return {"settings": {"enabled": self.enabled}}
        if method == "speech.advance":
            self.cancels.append(payload)
            return {"generation": payload["generation"]}
        if method == "speech.start":
            self.starts.append(payload)
            return {"job_id": str(len(self.starts))}
        if method == "speech.status":
            return {
                "job": {
                    "state": "completed",
                    "settled": True,
                    "metadata": {"fallback": True},
                    "error": None,
                }
            }
        if method == "speech.cancel":
            self.cancels.append(payload)
            return {"ok": True}
        return super().call(method, payload)


def test_disabled_output_and_validated_segments_only_then_no_restart_audio():
    svc = SpeechServices()
    m = active(svc)
    m.dialogue("start", {"text": "Hello", "model": "one"})
    finish(m)
    assert not svc.starts
    m.speech("configure", {"settings": {"enabled": True}})
    m.dialogue("start", {"text": "Fresh 103137", "model": "one"})
    finish(m)
    wait(lambda: len(svc.starts) == 1)
    assert svc.starts[0]["request"]["text"] == "Привет"
    assert svc.starts[0]["request"]["emotion"] == "neutral"
    assert svc.starts[0]["segment_id"] and not svc.starts[0]["preview"]
    checkpoint = m.checkpoint()
    assert not m._speech.thread.is_alive()
    restored = type(m)()
    restored.initialize({"services": svc, "checkpoint": checkpoint})
    restored.activate()
    assert not restored._speech.armed
    m.dispose()
    restored.dispose()


def test_preview_is_tracked_even_when_output_disabled_and_heartbeat_stops():
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    try:
        speech.preview()
        wait(lambda: len(svc.starts) == 1)
        assert svc.starts[0]["preview"] and svc.starts[0]["response_id"].startswith("preview-")
        wait(lambda: not speech.armed)
        assert not speech.enabled
        speech.configure({"enabled": True})
        speech.last_contact -= 6
        wait(lambda: not speech.armed)
        assert not speech.queue
    finally:
        speech.close()
    assert not speech.thread.is_alive()


def test_late_segment_wrong_response_queue_backpressure_and_cancel_wakes():
    svc = SpeechServices()
    speech = SpeechOutput(svc)
    speech.enabled = speech.armed = True
    speech.begin("new")
    speech.offer("old", {"id": "late", "text": "late"})
    assert not speech.queue
    for i in range(SEGMENT_QUEUE):
        speech.offer("new", {"id": str(i), "text": "bounded"})
    result = []
    producer = threading.Thread(
        target=lambda: result.append(speech.offer("new", {"id": "next", "text": "blocked"}))
    )
    producer.start()
    try:
        wait(lambda: speech.backpressure_waits > 0)
        assert len(speech.queue) == SEGMENT_QUEUE
        speech.cancel()
        producer.join(2)
        assert result == [False] and not speech.queue and not producer.is_alive()
    finally:
        speech.close()
        producer.join(2)
