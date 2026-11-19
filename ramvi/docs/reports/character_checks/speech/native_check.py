"""Run from installed wheels: native synth, actual output, cancel and lease cleanup."""

import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path

from ramvi_service.contracts.modules import Authority
from ramvi_service.contracts.speech import SPEAKER, SpeechRequest
from ramvi_service.resources import ResourceBroker
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.native import SystemSpeech
from ramvi_service.speech.service import SpeechService

result = {}
provider = SystemSpeech()
service = None
try:
    voices = provider.voices()
    result["voice_count"] = len(voices)
    result["synthesis"] = []
    for text, voice in [
        ("Проверка нового голоса. Число сто три тысячи сто тридцать семь.", "Milena"),
        ("A fresh voice check. Seven thousand two hundred forty nine.", "Samantha"),
        ("Новый ответ. Hello, welcome back.", "Milena"),
    ]:
        pieces = list(provider.synthesize(SpeechRequest(text, voice=voice), threading.Event()))
        pcm = b"".join(piece.pcm for piece in pieces)
        assert any(pcm), "silent synthesis"
        result["synthesis"].append(
            {
                "voice": voice,
                "sha256": hashlib.sha256(pcm).hexdigest(),
                "non_silent": True,
                "rate": pieces[0].rate,
                "audio_seconds": round(sum(len(p.pcm) for p in pieces) / 2 / pieces[0].rate, 3),
            }
        )
    assert len({item["sha256"] for item in result["synthesis"]}) == 3
    provider.close()
    with tempfile.TemporaryDirectory() as directory:
        auth = Authority()
        owner = auth.issue("character", "user", "tts-native")
        broker = ResourceBroker(auth, {SPEAKER: 1})
        data = ScopedData(Path(directory), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")
        service = SpeechService(broker)
        service.configure(owner, data, {"enabled": True, "voice": "Milena"})

        def settle(identifier):
            end = time.monotonic() + 60
            while not service.status(owner, identifier)["job"]["settled"]:
                assert time.monotonic() < end
                time.sleep(0.05)
            return service.status(owner, identifier)["job"]

        result["playback"] = settle(
            service.start(
                owner,
                data,
                secrets,
                {
                    "request": {
                        "text": "Проверка голоса завершена.",
                        "emotion": "joy",
                        "style": "cheerful",
                    }
                },
            )["job_id"]
        )
        result["cancellation"] = []
        for _ in range(3):
            job = service.start(
                owner, data, secrets, {"request": {"text": "Длинная проверка отмены. " * 25}}
            )["job_id"]
            time.sleep(0.1)
            service.cancel(owner, job)
            packet = settle(job)
            result["cancellation"].append({"state": packet["state"], "settled": packet["settled"]})
        identifier = service.start(
            owner, data, secrets, {"request": {"text": "Проверка остановки звучащего ответа. " * 5}}
        )["job_id"]
        deadline = time.monotonic() + 45
        while service.status(owner, identifier)["job"]["state"] != "playing":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        service.cancel(owner, identifier)
        result["playback_cancel"] = settle(identifier)
        assert result["playback_cancel"]["state"] == "cancelled"
        service.close()
        result["leases"] = broker.status()["leases"]
        assert result["playback"]["state"] == "completed" and result["leases"] == 0
finally:
    provider.close()
    if service:
        service.close()
Path("native_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
