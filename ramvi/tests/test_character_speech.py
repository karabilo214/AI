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

    def call(self, method: Any, payload: Any) -> Any:
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


def test_disabled_output_and_validated_segments_only_then_no_restart_audio() -> None:
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
    assert m._speech.thread is not None
    assert not m._speech.thread.is_alive()
    restored = type(m)()
    restored.initialize({"services": svc, "checkpoint": checkpoint})
    restored.activate()
    assert not restored._speech.armed
    m.dispose()
    restored.dispose()


def test_preview_is_tracked_even_when_output_disabled_and_heartbeat_stops() -> None:
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
    assert speech.thread is not None
    assert not speech.thread.is_alive()


def test_late_segment_wrong_response_queue_backpressure_and_cancel_wakes() -> None:
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


def test_next_segment_starts_synthesis_while_the_current_one_plays() -> None:
    """Пока звучит первая фраза, вторая уже отдана в синтез; после первой — играет вторая."""
    playing = threading.Event()

    class Playing(SpeechServices):
        def call(self, method: Any, payload: Any) -> Any:
            if method == "speech.status":
                done = not playing.is_set() and payload["job_id"] == "1" or payload["job_id"] != "1"
                return {
                    "job": {
                        "state": "completed" if done else "playing",
                        "settled": done,
                        "metadata": {},
                        "error": None,
                    }
                }
            return super().call(method, payload)

    svc = Playing()
    speech = SpeechOutput(svc)
    try:
        speech.configure({"enabled": True})  # запускает поток вывода
        playing.set()
        generation = speech.begin("r")
        for i in (1, 2):
            speech.offer(
                "r",
                {
                    "id": f"s{i}",
                    "text": f"фраза {i}",
                    "emotion": "joy",
                    "intensity": 0.5,
                    "voice_style": ("neutral", "irritated")[i - 1],  # разная подача — две фразы
                },
                generation,
            )
        speech.finish("r", generation)
        wait(lambda: len(svc.starts) == 2)  # вторая — пока первая звучит
        assert [s["segment_id"] for s in svc.starts] == ["s1", "s2"]
        playing.clear()
        wait(lambda: not speech.queue)
        assert len(svc.starts) == 2  # вторую не запускают повторно
    finally:
        speech.close()
