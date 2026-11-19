"""Explicit session contract, bounds, ownership and no capture outside a session."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.chat import ChatError
from ramvi_service.voice.audio import peak
from ramvi_service.voice.session import VoiceSession
from test_voice import FakeMicrophone, FakeTranscriber, broker_env, tone, wait


class Detector:
    def __init__(self, path: Path | None = None) -> None:
        self.closed = False

    def prepare(self) -> None:
        pass

    def speech(self, pcm: bytes, rate: int) -> bool:
        return peak(pcm[-rate // 10 * 2 :]) > 500

    def close(self) -> None:
        self.closed = True


def feed(mic: FakeMicrophone, session: VoiceSession, pcm: bytes) -> None:
    step = mic.rate // 10 * 2
    for offset in range(0, len(pcm), step):
        wait(lambda: session.audio.qsize() < 8)
        assert mic.on_block is not None
        mic.on_block(pcm[offset : offset + step])
    wait(lambda: session.audio.empty())
    time.sleep(0.01)


def test_partial_and_final_have_one_utterance_and_auto_language() -> None:
    mic = FakeMicrophone()
    recognizer = FakeTranscriber(Path("test"))
    ended = threading.Event()
    session = VoiceSession(
        mic, Detector(), recognizer, on_close=ended.set, partial_s=0.3, silence_s=0.3
    )
    assert not mic.opened
    session.start()
    try:
        feed(mic, session, tone(0.8) + bytes(16000))
        wait(lambda: any(e["type"] == "final" for e in session.read()["events"]))
        events = session.read()["events"]
        assert [e["type"] for e in events].count("speech_start") == 1
        assert any(e["type"] == "partial" and e["provisional"] for e in events)
        assert sum(e["type"] == "final" for e in events) == 1
        ids = {e["utterance_id"] for e in events if "utterance_id" in e}
        assert len(ids) == 1
        assert all(call[2] == "auto" for call in recognizer.calls)
    finally:
        session.stop()
        session.join()
    assert mic.stream is not None and mic.stream.closed and ended.is_set() and recognizer.closed
    assert all(not thread.is_alive() for thread in session.threads)
    previous = session.sequence
    mic.say(tone(0.5))
    assert session.sequence == previous and session.audio.empty()


def test_silence_and_playback_do_not_create_transcription_or_utterance() -> None:
    mic = FakeMicrophone()
    recognizer = FakeTranscriber(Path("test"))
    session = VoiceSession(mic, Detector(), recognizer, on_close=lambda: None)
    session.start()
    try:
        feed(mic, session, bytes(16000))
        session.playback(True)
        mic.say(tone(4))
        session.playback(False)
        mic.say(tone(0.5))  # gate tail, no echo accepted
        time.sleep(0.1)
        assert not recognizer.calls
        assert not any(e["type"] == "speech_start" for e in session.read()["events"])
        assert session.read()["barge_in_during_playback"] is False
    finally:
        session.stop()
        session.join()


def test_cancel_discards_late_transcription() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Slow(FakeTranscriber):
        def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
            entered.set()
            release.wait(2)
            return {"text": "late text", "language": "en"}

        def close(self) -> None:
            super().close()
            release.set()

    mic = FakeMicrophone()
    recognizer = Slow(Path("test"))
    session = VoiceSession(mic, Detector(), recognizer, on_close=lambda: None, silence_s=0.2)
    session.start()
    feed(mic, session, tone(0.4) + bytes(16000))
    assert entered.wait(1)
    session.stop()
    session.join()
    assert not any(e["type"] in {"partial", "final"} for e in session.read()["events"])


def test_overflow_stops_and_releases_instead_of_silent_loss() -> None:
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class SlowDetector(Detector):
        def speech(self, pcm: bytes, rate: int) -> bool:
            entered.set()
            release.wait(2)
            return True

        def close(self) -> None:
            release.set()

    mic = FakeMicrophone()
    session = VoiceSession(mic, SlowDetector(), FakeTranscriber(Path("test")), on_close=closed.set)
    session.start()
    mic.say(tone(0.1))
    assert entered.wait(1)
    mic.say(tone(4))
    release.set()
    wait(closed.is_set)
    session.join()
    assert (
        session.error == "voice.session_overflow" and mic.stream is not None and mic.stream.closed
    )
    assert any(e["type"] == "error" for e in session.read()["events"])


def test_session_resource_ownership_and_chat_recording_remain_separate(tmp_path: Path) -> None:
    env, authority, broker, chat_owner = broker_env(tmp_path)
    env.voice.make_detector = Detector
    character = authority.issue("character", "local_user", "voice-session")
    other = authority.issue("other", "local_user", "other")
    try:
        result = env.voice.start_session(owner=character)
        assert broker.status()["leases"] == 1
        with pytest.raises(ChatError):
            env.voice.read_session(owner=other)
        with pytest.raises(ChatError):
            env.voice.cancel(owner=other)
        with pytest.raises(ChatError):
            env.voice.start()
        env.voice.cancel_session(owner=character, capture_id=result["session_id"])
        assert broker.status()["leases"] == 0
        env.voice.start()  # ordinary Chat push-to-record still works
        env.mic.say(tone(0.5))
        assert env.voice.stop()["text"] == "привет мир"
        assert broker.status()["leases"] == 0
        second = env.voice.start_session(owner=character)
        with pytest.raises(ChatError):
            env.voice.cancel_session(owner=character, capture_id=result["session_id"])
        assert env.voice.read_session(owner=character)["session_id"] == second["session_id"]
    finally:
        env.voice.close()


def test_detector_failure_does_not_open_microphone_or_leak_lease(tmp_path: Path) -> None:
    class Broken(Detector):
        def prepare(self) -> None:
            raise RuntimeError("broken VAD")

    env, authority, broker, chat_owner = broker_env(tmp_path)
    env.voice.make_detector = Broken
    try:
        with pytest.raises(RuntimeError):
            env.voice.start_session()
        assert broker.status()["leases"] == 0 and not env.mic.opened
        assert env.voice.status()["state"] == "idle"
    finally:
        env.voice.close()


def test_language_switch_inside_one_turn_uses_independent_auto_windows() -> None:
    class Bilingual(FakeTranscriber):
        def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
            assert language == "auto"
            detected = "ru" if peak(pcm) < 1500 else "en"
            return {"text": "Привет" if detected == "ru" else "Hello", "language": detected}

    mic = FakeMicrophone()
    session = VoiceSession(mic, Detector(), Bilingual(Path("test")), on_close=lambda: None)
    session.start()
    try:
        feed(
            mic,
            session,
            tone(2.5, amplitude=1000) + bytes(6400) + tone(2.5, amplitude=2000) + bytes(32000),
        )
        wait(lambda: any(e["type"] == "final" for e in session.read()["events"]))
        events = session.read()["events"]
        assert sum(e["type"] == "speech_start" for e in events) == 1
        final = next(e for e in events if e["type"] == "final")
        assert final["languages"] == ["ru", "en"] and final["language"] == "mixed"
        assert "Привет" in final["text"] and "Hello" in final["text"]
    finally:
        session.stop()
        session.join()


def test_empty_final_is_rejected_without_a_stuck_recognizing_turn() -> None:
    mic = FakeMicrophone()
    session = VoiceSession(
        mic, Detector(), FakeTranscriber(Path("test"), ""), on_close=lambda: None, silence_s=0.2
    )
    session.start()
    try:
        feed(mic, session, tone(0.5) + bytes(16000))
        wait(lambda: any(e["type"] == "rejected" for e in session.read()["events"]))
        assert not any(e["type"] == "final" for e in session.read()["events"])
        assert session.error is None and not session.stopped.is_set()
    finally:
        session.stop()
        session.join()


def test_all_owned_cleanup_runs_even_if_microphone_and_detector_close_fail() -> None:
    mic = FakeMicrophone()
    recognizer = FakeTranscriber(Path("test"))
    calls = []

    class Broken(Detector):
        def close(self):
            calls.append("detector")
            raise OSError("driver disappeared")

    session = VoiceSession(mic, Broken(), recognizer, on_close=lambda: calls.append("release"))
    session.start()
    assert mic.stream is not None

    def fail():
        calls.append("microphone")
        raise OSError("close failed")

    mic.stream.close = fail
    session.stop()
    session.join()
    assert calls == ["microphone", "release", "detector"] and recognizer.closed
    assert session.error == "voice.cleanup_failed" and not any(
        t.is_alive() for t in session.threads
    )
