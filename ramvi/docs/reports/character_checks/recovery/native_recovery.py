"""Local recovery: missing voice, hung audio child, silent provider; real healthy output."""

import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from ramvi_engine.processes import ProcessSupervisor
from ramvi_service.contracts.modules import Authority
from ramvi_service.contracts.speech import SPEAKER
from ramvi_service.resources import ResourceBroker
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.speech.native import SystemSpeech
from ramvi_service.speech.output import SoundOutput
from ramvi_service.speech.service import SpeechService

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
phase = {"fault": None}
devices = []
providers = []
hung = []


class Silent:
    def __init__(self):
        self.closed = threading.Event()

    def synthesize(self, request, cancel):
        self.closed.wait(5)
        yield from ()

    def close(self):
        self.closed.set()

    cancel = close


class Device:
    def __init__(self, rate):
        worker = None
        if phase["fault"] == "device":
            worker = ProcessSupervisor(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time\nfor line in sys.stdin: time.sleep(60)",
                ],
                timeout=0.15,
            )
            hung.append(worker)
        self.backend = SoundOutput(rate, worker=worker, timeout=0.15 if worker else 2)
        self.non_silent = False
        self.hash = hashlib.sha256()
        self.starts = 0
        self.closed = False
        devices.append(self)

    def start(self):
        self.backend.start()
        self.starts += 1

    def write(self, pcm):
        self.backend.write(pcm)
        self.non_silent |= any(pcm)
        self.hash.update(pcm)

    def interrupt(self):
        self.backend.interrupt()

    def cancel(self):
        self.backend.cancel()

    def close(self):
        try:
            self.backend.close()
        finally:
            self.closed = self.backend.worker.status()["state"] != "running"


def backend(info, key):
    p = Silent() if phase["fault"] == "provider" else SystemSpeech()
    providers.append(p)
    return p


result = {"runs": []}
service = None
native = SystemSpeech()
try:
    voices = native.voices()
    native.close()
    ru = next((v["id"] for v in voices if "ru" in v["languages"]), "")
    en = next((v["id"] for v in voices if "en" in v["languages"]), ru)
    with tempfile.TemporaryDirectory() as directory:
        authority = Authority()
        owner = authority.issue("character", "user", "recovery-check")
        broker = ResourceBroker(authority, {SPEAKER: 1})
        data = ScopedData(Path(directory), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")
        service = SpeechService(
            broker, provider_factory=backend, output_factory=Device, audio_timeout=3
        )

        def wait(identifier):
            until = time.monotonic() + 30
            while not service.status(owner, identifier)["job"]["settled"]:
                assert time.monotonic() < until, "recovery deadline"
                time.sleep(0.01)
            return service.status(owner, identifier)["job"]

        for index, (fault, code) in enumerate(
            [("voice", "104729"), ("device", "393241"), ("provider", "10000019")]
        ):
            phase["fault"] = fault
            service.first_audio_timeout = 0.15 if fault == "provider" else 8
            service.configure(
                owner,
                data,
                {
                    "enabled": True,
                    "voice": "RAMVI unavailable voice 104729" if fault == "voice" else ru,
                },
            )
            service.advance(owner, index * 2 + 1, f"bad-{index}")
            started = time.monotonic()
            bad = wait(
                service.start(
                    owner,
                    data,
                    secrets,
                    {
                        "generation": index * 2 + 1,
                        "response_id": f"bad-{index}",
                        "request": {"text": "Проверка ошибки голоса."},
                    },
                )["job_id"]
            )
            error_ms = round((time.monotonic() - started) * 1000, 3)
            expected = {
                "voice": "speech.voice_missing",
                "device": "speech.device_timeout",
                "provider": "speech.provider_timeout",
            }[fault]
            assert bad["state"] == "error" and bad["error"] == expected, bad
            assert broker.status()["leases"] == 0
            phase["fault"] = None
            service.first_audio_timeout = 8
            voice = en if index == 1 else ru
            service.configure(owner, data, {"enabled": True, "voice": voice})
            service.advance(owner, index * 2 + 2, f"good-{index}")
            text = (
                f"Recovered voice. Code {code}."
                if index == 1
                else f"Голос восстановлен. Проверочный код {code}."
            )
            good = wait(
                service.start(
                    owner,
                    data,
                    secrets,
                    {
                        "generation": index * 2 + 2,
                        "response_id": f"good-{index}",
                        "request": {"text": text},
                    },
                )["job_id"]
            )
            device = devices[-1]
            assert (
                good["state"] == "completed"
                and device.non_silent
                and device.closed
                and device.starts == 1
            ), good
            result["runs"].append(
                {
                    "fault": fault,
                    "error": bad["error"],
                    "error_ms": error_ms,
                    "code": code,
                    "recovered_voice": voice,
                    "recovered_state": good["state"],
                    "non_silent": device.non_silent,
                    "sha256": device.hash.hexdigest(),
                    "starts": device.starts,
                    "leases_after": broker.status()["leases"],
                }
            )
        result["hung_children_stopped"] = all(p.status()["state"] != "running" for p in hung)
        result["all_device_children_stopped"] = all(
            d.backend.worker.status()["state"] != "running" for d in devices
        )
        result["distinct_recovered_audio"] = len({r["sha256"] for r in result["runs"]})
        service.close()
        result["leases_after_close"] = broker.status()["leases"]
        if sys.platform == "win32":
            from ramvi_service.voice.capture import ProcessMicrophone

            received = {"bytes": 0, "blocks": 0}

            def capture(pcm):
                received["bytes"] += len(pcm)
                received["blocks"] += 1

            stream = ProcessMicrophone().open(None, capture)
            try:
                deadline = time.monotonic() + 5
                while received["blocks"] < 10 and time.monotonic() < deadline:
                    time.sleep(0.02)
            finally:
                stream.close()
            result["microphone"] = {
                **received,
                "rate": stream.rate,
                "child_stopped": stream.worker.status()["state"] != "running",
                "reader_stopped": not stream.thread.is_alive(),
                "error": stream.error,
            }
            assert (
                received["blocks"] > 0
                and result["microphone"]["child_stopped"]
                and result["microphone"]["reader_stopped"]
                and stream.error is None
            )

        assert (
            result["hung_children_stopped"]
            and result["all_device_children_stopped"]
            and result["distinct_recovered_audio"] == 3
        )
finally:
    native.close()
    if service:
        service.close()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
