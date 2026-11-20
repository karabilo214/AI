"""DAC time, not submission time: solvable silence/tone/gap/cancel controls."""

import json
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_service.speech.clock import cue_at, playback_view
from ramvi_service.speech.device_worker import DeviceOutput


def callback_device(rate: int = 16000) -> DeviceOutput:
    device = DeviceOutput.__new__(DeviceOutput)
    device.rate, device.frames = rate, rate // 50
    device.silence = bytes(device.frames * 2)
    device.windows = deque(maxlen=128)
    device.pending = deque()
    device.last_dac = None
    device.offset = 0
    device.valid = True
    device.stream = SimpleNamespace(time=100.0)
    return device


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 44100, 48000])
def test_callback_anchors_audio_to_future_dac_and_preserves_gaps_and_sample_position(
    rate: int,
) -> None:
    device = callback_device(rate)
    data = bytearray(device.frames * 2)
    timing = SimpleNamespace(outputBufferDacTime=100.1)
    device._callback(data, device.frames, timing, False)
    assert data == device.silence and not device.windows
    device.pending.append((b"\xff\x1f" * device.frames, 0.25))
    device._callback(data, device.frames, timing, False)
    raw = device.position()
    assert data != device.silence and not device.pending
    view = playback_view(raw, 0)
    assert len(view["windows"]) == 1
    assert view["windows"][0]["start_ms"] == pytest.approx(100)
    assert view["windows"][0]["position_s"] == 0
    assert playback_view(raw, 0.2)["windows"] == []  # Audio has already passed.
    device.pending.append((device.silence, 0.0))
    device._callback(data, device.frames, SimpleNamespace(outputBufferDacTime=100.5), False)
    view = playback_view(device.position(), 0.5)
    assert view["windows"][0]["amplitude"] == 0
    assert view["windows"][0]["position_s"] == pytest.approx(0.02)
    assert playback_view(device.position(), 0.3)["windows"][0]["start_ms"] == pytest.approx(200)


def test_short_final_block_has_no_invented_audio_and_underflow_invalidates_old_windows() -> None:
    device = callback_device()
    device.pending.append((b"\x01\0" * 17, 0.01))
    data = bytearray(640)
    device._callback(data, 320, SimpleNamespace(outputBufferDacTime=100.02), False)
    assert data[:34] == b"\x01\0" * 17 and not any(data[34:])
    assert device.windows[-1][1] == 17 / 16000
    device._callback(data, 320, SimpleNamespace(outputBufferDacTime=100.04), True)
    assert playback_view(device.position(), 0)["clock"] == "unavailable"
    device._callback(data, 320, SimpleNamespace(outputBufferDacTime=100.06), False)
    assert playback_view(device.position(), 0)["windows"] == []


def test_bounded_telemetry_and_character_phoneme_viseme_cues() -> None:
    device = callback_device()
    for index in range(1000):
        device.pending.append((device.silence, 0))
        device._callback(
            bytearray(640), 320, SimpleNamespace(outputBufferDacTime=100 + index * 0.02), False
        )
    assert len(device.windows) == 128
    assert len(json.dumps(device.position()).encode()) < 60000
    for kind, value in [("characters", "а"), ("phonemes", "aa"), ("visemes", "talk_a")]:
        alignment = {"kind": kind, kind: [value], "starts": [0.1], "ends": [0.2]}
        assert cue_at(alignment, 0.15) == {"kind": kind, "value": value}
        assert cue_at(alignment, 0.2) is None
    assert cue_at({"kind": []}, 0) is None
    assert (
        cue_at(
            {"kind": "characters", "characters": ["а"], "starts": [float("nan")], "ends": [1]}, 0
        )
        is None
    )


def test_slow_synthesis_does_not_hold_status_lock_and_cancellation_clears_clock(
    tmp_path: Path,
) -> None:
    import threading

    from ramvi_service.contracts.speech import AudioPiece
    from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
    from ramvi_service.speech.service import SpeechService

    release = threading.Event()

    class Source:
        def synthesize(self, request: Any, cancel: threading.Event) -> Iterator[AudioPiece]:
            yield AudioPiece(b"\0\x20" * 320, 16000)
            release.wait(2)

        def close(self) -> None:
            release.set()

    class Device:
        def __init__(self, rate: int) -> None:
            self.rate = rate

        def start(self) -> None:
            pass

        def write(self, pcm: bytes) -> None:
            pass

        def close(self) -> None:
            pass

        def cancel(self) -> None:
            pass

        def playback_view(self, alignment: Any = None) -> dict[str, Any]:
            return {
                "clock": "portaudio_dac",
                "windows": [
                    {"start_ms": 0, "end_ms": 20, "position_s": 0, "amplitude": 0.25, "cue": None}
                ],
            }

    service = SpeechService(provider_factory=lambda *_: Source(), output_factory=Device)
    data, secrets = ScopedData(tmp_path, "character"), ScopedSecrets(MemorySecrets(), "character")
    try:
        service.configure("owner", data, {"enabled": True})
        service.advance("owner", 1, "first")
        identifier = service.start(
            "owner",
            data,
            secrets,
            {
                "request": {"text": "Fresh 65539", "emotion": "sad"},
                "generation": 1,
                "response_id": "first",
                "segment_id": "s1",
            },
        )["job_id"]
        deadline = time.monotonic() + 3
        while not service.status("owner", identifier)["job"]["bytes"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        packet = service.status("owner", identifier)["job"]
        assert packet["metadata"]["requested_emotion"] == "sad"
        assert packet["playback"]["windows"]
        service.advance("owner", 2, "second")
        assert service.status("owner", identifier)["job"]["playback"]["windows"] == []
    finally:
        release.set()
        service.close()


def test_queued_audio_forecast_is_anchored_to_callback_and_disappears_on_expiry() -> None:
    device = callback_device()
    device.pending.extend([(device.silence, 0.25)] * 4)
    assert playback_view(device.position(), 0)["windows"] == []
    device._callback(bytearray(640), 320, SimpleNamespace(outputBufferDacTime=100.02), False)
    packet = playback_view(device.position(), 0)
    assert len(packet["windows"]) == 4
    assert [w["start_ms"] for w in packet["windows"]] == [20, 40, 60, 80]
    assert [w["position_s"] for w in packet["windows"]] == [0, 0.02, 0.04, 0.06]
    assert playback_view(device.position(), 0.2)["windows"] == []


@pytest.mark.parametrize("split", [1, 3, 511, 8192, 32767])
def test_provider_chunk_boundaries_do_not_insert_silence(split: int) -> None:
    import threading

    from ramvi_service.contracts.speech import AudioPiece
    from ramvi_service.speech.buffer import PCMBuffer

    pcm = bytes(range(256)) * 101

    class Source:
        def synthesize(self, request: Any, cancel: threading.Event) -> Iterator[AudioPiece]:
            for offset in range(0, len(pcm), split):
                yield AudioPiece(pcm[offset : offset + split], 16000)

    buffer = PCMBuffer(threading.Event())
    producer = threading.Thread(target=buffer.produce, args=(Source(), None))
    producer.start()
    blocks = []
    try:
        while (piece := buffer.get(time.monotonic() + 5)) is not None:
            blocks.append(piece.pcm)
    finally:
        buffer.cancel.set()
        producer.join(2)
    assert not producer.is_alive()
    assert b"".join(blocks) == pcm
    assert all(len(b) == 640 for b in blocks[:-1])
    assert 0 < len(blocks[-1]) <= 640


@pytest.mark.parametrize("method", ["write", "drain"])
def test_stuck_device_write_or_drain_terminates_worker(method: str) -> None:
    from ramvi_service.speech.output import SoundOutput

    class Worker:
        stopped = False

        def start(self) -> None:
            pass

        def request(self, operation: str, payload: Any, **kwargs: Any) -> dict[str, bool]:
            if operation == method:
                raise TimeoutError("driver blocked")
            return {"ok": True}

        def stop(self, **kwargs: Any) -> None:
            self.stopped = True

    worker = Worker()
    output = SoundOutput(16000, worker=worker)
    output.start()
    with pytest.raises(TimeoutError, match="speech.device_timeout"):
        output.write(bytes(640)) if method == "write" else output.drain()
    assert worker.stopped
    output.close()
