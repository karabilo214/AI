"""Isolated Windows voice check using an existing portable runtime and STT weights.

Requires owner permission, a free shared machine and an external absolute stop task.
Pass --physical-microphone only for an authorized two-second microphone check.
Otherwise only the supplied prerecorded synthetic WAV is used.
"""
# ruff: noqa: E402

import argparse
import json
import os
import sys
import time
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--bundle", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--work", type=Path, required=True)
p.add_argument("--models", type=Path, required=True)
p.add_argument("--physical-microphone", action="store_true")
args = p.parse_args()
root = args.work.resolve()
source = args.source.resolve()
bundle = args.bundle.resolve()
deps = root / "deps"
deps.mkdir(exist_ok=True)
for wheel in (root / "wheels").glob("*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(deps)
sys.path[:0] = [str(source / x) for x in ["service", "storage", "character"]] + [str(deps)]
os.environ.update(PYTHONUTF8="1", PYTHONPATH=str(deps))
os.environ["PATH"] = str(bundle / "runtime/python") + os.pathsep + os.environ.get("PATH", "")
from ramvi_service.contracts.modules import Authority
from ramvi_service.contracts.voice import MICROPHONE
from ramvi_service.resources import ResourceBroker
from ramvi_service.voice.models import DEFAULT_MODEL, MODELS
from ramvi_service.voice.service import VoiceService
from ramvi_service.voice.session import VoiceSession
from ramvi_service.voice.stt import WhisperProcess
from ramvi_service.voice.vad import VADProcess

model = MODELS[DEFAULT_MODEL].path(args.models)
vad_path = source / "service/ramvi_service/voice/assets/silero_v5.1.2.onnx"
lock = root.parent / "ramvi_gpu_owner.json"
fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
with os.fdopen(fd, "w") as f:
    json.dump({"owner": "character-voice", "root": str(root), "pid": os.getpid()}, f)
results = {}
voice = None
session = None
vad = None
try:
    vad = VADProcess(vad_path)
    start = time.monotonic()
    vad.prepare()
    results["vad_start_s"] = round(time.monotonic() - start, 3)
    results["silence_speech"] = vad.speech(bytes(32000), 16000)
    vad.close()
    vad = None
    if args.physical_microphone:
        authority = Authority()
        owner = authority.issue("voice", "local_user", "native-mic-check")
        broker = ResourceBroker(authority, {MICROPHONE: 1})
        settings = SimpleNamespace(
            voice_device="",
            voice_language="auto",
            voice_model=DEFAULT_MODEL,
            voice_backend="local",
            voice_api_url="",
            voice_api_model="whisper-1",
        )
        voice = VoiceService(
            lambda: args.models,
            lambda: settings,
            resources=broker,
            owner=owner,
            has_nvidia=lambda: False,
            memory=lambda: (None, None),
        )
        devices = voice.devices()["devices"]
        results["input_device_count"] = len(devices)
        voice.start_session()
        time.sleep(2)
        status = voice.read_session()
        voice.cancel_session()
        results["microphone"] = {
            "state_before_stop": status["state"],
            "state_after_stop": voice.status()["state"],
            "leases_after_stop": broker.status()["leases"],
        }
        voice.close()
        voice = None

    class FileMicrophone:
        rate = 16000

        def open(self, device, callback):
            self.callback = callback
            self.closed = False
            return self

        def close(self):
            self.closed = True

        def devices(self):
            return []

    mic = FileMicrophone()
    recognizer = WhisperProcess(model, threads=4, timeout=60)
    session = VoiceSession(mic, VADProcess(vad_path), recognizer, on_close=lambda: None)
    session.start()
    with wave.open(str(root / "fresh103137.wav"), "rb") as f:
        pcm = f.readframes(f.getnframes())
    pcm = bytes(16000) + pcm + bytes(40000)
    for offset in range(0, len(pcm), 3200):
        if session.stopped.is_set():
            break
        mic.callback(pcm[offset : offset + 3200])
        time.sleep(0.1)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if any(e["type"] == "final" for e in session.read()["events"]) or session.error:
            break
        time.sleep(0.1)
    results["recorded_audio"] = {"events": session.read()["events"], "error": session.error}
    session.stop()
    session.join()
    session = None
    results["file_stream_closed"] = mic.closed
    (root / "windows_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False), flush=True)
finally:
    if vad:
        vad.close()
    if voice:
        voice.close()
    if session:
        session.stop()
        session.join()
    try:
        saved = json.loads(lock.read_text())
        if saved.get("pid") == os.getpid():
            lock.unlink()
    except FileNotFoundError:
        pass
