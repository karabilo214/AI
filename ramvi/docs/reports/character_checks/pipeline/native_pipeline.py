"""Physical local output: three generation fences, fresh replies and duplicate delivery."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from ramvi_service.contracts.modules import Authority
from ramvi_service.contracts.speech import SPEAKER
from ramvi_service.resources import ResourceBroker
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.native import SystemSpeech
from ramvi_service.speech.output import SoundOutput
from ramvi_service.speech.service import SpeechService

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
recorders = []
current = 0


class RecordedOutput:
    def __init__(self, rate):
        self.generation = current
        self.backend = SoundOutput(rate)
        self.started = None
        self.last_write = None
        self.non_silent = False
        self.hash = hashlib.sha256()
        self.writes = 0
        self.closed = False
        self.max_block = 0
        recorders.append(self)

    def start(self):
        self.backend.start()
        self.started = time.monotonic()

    def write(self, pcm):
        self.backend.write(pcm)
        self.last_write = time.monotonic()
        self.hash.update(pcm)
        self.non_silent |= any(pcm)
        self.writes += 1
        self.max_block = max(self.max_block, len(pcm))

    def cancel(self):
        self.backend.cancel()

    def close(self):
        self.backend.close()
        self.closed = True


results = {"runs": []}
service = None
native = SystemSpeech()
try:
    voices = native.voices()
    voice = next((v["id"] for v in voices if "ru" in v["languages"]), "")
    native.close()
    with tempfile.TemporaryDirectory() as directory:
        authority = Authority()
        owner = authority.issue("character", "user", "pipeline-check")
        broker = ResourceBroker(authority, {SPEAKER: 1})
        data = ScopedData(Path(directory), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")
        service = SpeechService(broker, output_factory=RecordedOutput)
        service.configure(owner, data, {"enabled": True, "voice": voice})

        def wait(predicate):
            until = time.monotonic() + 60
            while not predicate():
                assert time.monotonic() < until, "deadline"
                time.sleep(0.01)

        for index, code in enumerate(["131071", "262147", "524287"]):
            old_generation = index * 2 + 1
            new_generation = old_generation + 1
            service.advance(owner, old_generation, f"old-{index}")
            current = old_generation
            old_payload = {
                "generation": old_generation,
                "response_id": f"old-{index}",
                "segment_id": "s1",
                "request": {"text": "Сейчас звучит старый длинный ответ. " * 8},
            }
            old = service.start(owner, data, secrets, old_payload)["job_id"]
            wait(
                lambda identifier=old: (
                    service.status(owner, identifier)["job"]["state"] == "playing"
                    and recorders[-1].writes > 0
                )
            )
            old_output = recorders[-1]
            started = time.monotonic()
            service.advance(owner, new_generation, f"new-{index}")
            ack = time.monotonic()
            try:
                service.start(owner, data, secrets, old_payload)
            except RuntimeError as error:
                assert str(error) == "speech.stale_generation"
            else:
                raise AssertionError("old request accepted")
            wait(lambda identifier=old: service.status(owner, identifier)["job"]["settled"])
            old_packet = service.status(owner, old)["job"]
            old_count = old_output.writes
            current = new_generation
            payload = {
                "generation": new_generation,
                "response_id": f"new-{index}",
                "segment_id": "s1",
                "request": {"text": f"Новый ответ. Проверочный код {code}."},
            }
            new = service.start(owner, data, secrets, payload)["job_id"]
            wait(lambda identifier=new: service.status(owner, identifier)["job"]["settled"])
            packet = service.status(owner, new)["job"]
            new_output = recorders[-1]
            assert packet["state"] == "completed" and new_output.non_silent and new_output.closed
            count = len(recorders)
            duplicate = service.start(owner, data, secrets, payload)
            assert duplicate["job_id"] == new and duplicate["duplicate"] and count == len(recorders)
            assert old_output.writes == old_count and old_output.last_write <= ack
            results["runs"].append(
                {
                    "code": code,
                    "old_state": old_packet["state"],
                    "new_state": packet["state"],
                    "fence_ms": round((ack - started) * 1000, 3),
                    "old_audio_after_ack": False,
                    "new_non_silent": new_output.non_silent,
                    "new_sha256": new_output.hash.hexdigest(),
                    "duplicate_replayed": False,
                    "max_pcm_block_bytes": new_output.max_block,
                    "pcm_high_water": packet["pcm_buffer"]["high_water"],
                    "leases_after": broker.status()["leases"],
                }
            )
        assert len({r["new_sha256"] for r in results["runs"]}) == 3
        service.close()
        results["leases_after_close"] = broker.status()["leases"]
        results["all_outputs_closed"] = all(r.closed for r in recorders)
finally:
    native.close()
    if service:
        service.close()
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False), flush=True)
