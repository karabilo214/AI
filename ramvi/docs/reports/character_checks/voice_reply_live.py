"""9.11: ответ персонажа целиком — части с разной подачей голоса (облако OpenAI + Gemini TTS).

Персонаж (OpenAI) отвечает на реплики; каждая часть ответа озвучивается Gemini с пометкой
подачи так же, как это делает модуль (`speech.voice_style`); ответ склеивается в один WAV
для прослушивания. Печатаются части: текст, эмоция, сила, подача.

usage: voice_reply_live.py <файл ключей> <токенизатор Qwen3> <модель> <профиль> <мат> <папка WAV>
       <отчёт.json> <реплика> [реплика ...]
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "character_checks" / "roast"))

from live_mac import Secrets, read_keys  # noqa: E402
from ramvi_character.speech import voice_style  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from ramvi_service.contracts.speech import SpeechRequest  # noqa: E402
from ramvi_service.speech.cloud import CloudSpeech  # noqa: E402
from roast_live import LAST_JOB, pick, say  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

PAUSE = b"\x00\x00" * (24000 // 5)  # 0,2 с между частями


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    os.environ["RAMVI_TOKENIZER"] = sys.argv[2]
    choice, profile, level = sys.argv[3], sys.argv[4], int(sys.argv[5])
    folder, out, lines = Path(sys.argv[6]), Path(sys.argv[7]), sys.argv[8:]
    folder.mkdir(parents=True, exist_ok=True)
    providers = json.loads((ROOT / "service/ramvi_service/speech/providers.json").read_text())
    tts_model = next(p["model"] for p in providers["providers"] if p["id"] == "gemini")
    app = Path(tempfile.mkdtemp(prefix="ramvi-voice-"))
    shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    report: list[dict[str, Any]] = []
    try:
        model = pick(client, keys, choice)
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        client.ok("POST", path, {"operation": "profile", "arguments": {"id": profile}})
        client.ok(
            "POST",
            path,
            {"operation": "boundary", "arguments": {"name": "profanity", "allowed": level > 0}},
        )
        if level:
            client.ok("POST", path, {"operation": "profanity_level", "arguments": {"level": level}})
        for number, text in enumerate(lines, 1):
            reply, seconds, error = say(client, path, text, model)
            parts = [e for e in LAST_JOB.get("events", []) if e.get("type") == "segment"]
            audio = b""
            shown = []
            for part in parts:
                style = voice_style(part["emotion"], part["intensity"], part["voice_style"])
                request = SpeechRequest(part["text"], voice="Kore", language="ru", style=style)
                for attempt in range(3):
                    try:
                        pcm = b"".join(
                            p.pcm
                            for p in CloudSpeech("gemini", tts_model, keys["google"]).synthesize(
                                request, threading.Event()
                            )
                        )
                        break
                    except RuntimeError:  # 429 — подождать и повторить
                        time.sleep(5 * (attempt + 1))
                else:
                    pcm = b""
                audio += pcm + PAUSE
                shown.append(
                    {
                        "text": part["text"],
                        "emotion": part["emotion"],
                        "intensity": part["intensity"],
                        "voice_style": part["voice_style"],
                        "audio_seconds": round(len(pcm) / 48000, 2),
                    }
                )
                time.sleep(2)
            with wave.open(str(folder / f"reply_{number}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(audio)
            report.append({"owner": text, "seconds": seconds, "error": error, "parts": shown})
            print(f"\n{number}. Владелец: {text}  ({seconds} с)")
            for s in shown:
                print(
                    f"   [{s['emotion']} {s['intensity']} {s['voice_style']}] {s['text']}",
                    flush=True,
                )
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        shutil.rmtree(app, ignore_errors=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
