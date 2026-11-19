"""Reproduce CPU VAD/STT sessions using recorded/synthetic audio, never a live mic.

Example: python voice_native.py --source <ramvi> --model <ggml-model> --work <out>
         --cases voice/cases.json --generate-synthetic
Requires the optional voice dependencies; --generate-synthetic uses macOS Milena.
"""
# ruff: noqa: E402

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--audio-dir", type=Path)
parser.add_argument("--generate-synthetic", action="store_true")
args = parser.parse_args()
source, work = args.source.resolve(), args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
audio_dir = (args.audio_dir or work / "audio").resolve()
audio_dir.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(source / part) for part in ("service", "storage", "engine")]
from ramvi_service.voice.session import VoiceSession
from ramvi_service.voice.stt import WhisperProcess
from ramvi_service.voice.vad import VADProcess


class FileMicrophone:
    rate = 16000

    def open(self, device, callback):
        self.callback, self.closed = callback, False
        return self

    def close(self):
        self.closed = True

    def devices(self):
        return []


results = []
for case in json.loads(args.cases.read_text(encoding="utf-8")):
    filename = case["id"] + ".wav"
    if args.generate_synthetic:
        subprocess.run(
            [
                "say",
                "-v",
                "Milena",
                "-o",
                str(audio_dir / filename),
                "--file-format=WAVE",
                "--data-format=LEI16@16000",
                case["reference"],
            ],
            check=True,
            timeout=30,
        )
    with wave.open(str(audio_dir / filename), "rb") as reader:
        if (
            reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getnchannels() != 1
        ):
            raise ValueError("expected 16 kHz mono PCM16 WAV")
        pcm = reader.readframes(reader.getnframes())
    mic = FileMicrophone()
    detector = VADProcess(source / "service/ramvi_service/voice/assets/silero_v5.1.2.onnx")
    recognizer = WhisperProcess(args.model, threads=5, timeout=60)
    session = VoiceSession(mic, detector, recognizer, on_close=lambda: None)
    started = time.monotonic()
    session.start()
    try:
        padded = bytes(16000) + pcm + bytes(40000)
        for offset in range(0, len(padded), 3200):
            if session.stopped.is_set():
                break
            mic.callback(padded[offset : offset + 3200])
            time.sleep(0.1)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            events = session.read()["events"]
            owner_ids = {e["utterance_id"] for e in events if e["type"] == "speech_start"}
            final_ids = {e["utterance_id"] for e in events if e["type"] == "final"}
            if (owner_ids and owner_ids == final_ids) or session.error:
                break
            time.sleep(0.1)
        text = " ".join(e["text"] for e in session.read()["events"] if e["type"] == "final")
        row = {
            "id": case["id"],
            "events": session.read()["events"],
            "error": session.error,
            "wall_s": round(time.monotonic() - started, 3),
            "bilingual_fragments_present": all(
                part.lower() in text.lower() for part in case["fragments"]
            ),
            "number_exact": case["number"] in text,
        }
        results.append(row)
        (work / "voice_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        session.stop()
        session.join()
    assert mic.closed and all(not thread.is_alive() for thread in session.threads)
