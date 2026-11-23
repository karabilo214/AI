"""9.11: ответ по частям отдельными запросами против всего ответа одним запросом (Gemini TTS).

Берёт ответы из voice_reply_result.json; для каждого — WAV «parts» (как сейчас: часть = запрос)
и «whole» (весь текст одним запросом, подача частей описана по порядку). Ошибка озвучки
не заменяется тишиной, а попадает в отчёт.

usage: voice_whole_ab.py <файл ключей> <ответы.json> <папка WAV>
"""

import json
import sys
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import read_keys  # noqa: E402
from ramvi_character.speech import VOICE_DELIVERY, voice_style  # noqa: E402
from ramvi_service.contracts.speech import SpeechRequest  # noqa: E402
from ramvi_service.speech.cloud import CloudSpeech  # noqa: E402

PAUSE = b"\x00\x00" * (24000 // 5)


def speak(engine: CloudSpeech, request: SpeechRequest) -> bytes:
    for attempt in range(6):
        try:
            return b"".join(p.pcm for p in engine.synthesize(request, threading.Event()))
        except RuntimeError as error:
            if "429" not in str(error):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("speech.api_http_429 after retries")


def save(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    replies = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    folder = Path(sys.argv[3])
    folder.mkdir(parents=True, exist_ok=True)
    providers = json.loads((ROOT / "service/ramvi_service/speech/providers.json").read_text())
    model = next(p["model"] for p in providers["providers"] if p["id"] == "gemini")
    engine = CloudSpeech("gemini", model, keys["google"])
    for number, reply in enumerate(replies, 1):
        parts = reply["parts"]
        pcm = b""
        for part in parts:
            style = voice_style(part["emotion"], part["intensity"], part["voice_style"])
            pcm += (
                speak(engine, SpeechRequest(part["text"], voice="Kore", language="ru", style=style))
                + PAUSE
            )
            time.sleep(3)
        save(folder / f"reply_{number}_parts.wav", pcm)
        plan = " Then ".join(
            f'"{p["text"]}" — {p["emotion"]} at intensity {p["intensity"]:.2f}: '
            f"{VOICE_DELIVERY.get(p['voice_style'], p['voice_style'])}."
            for p in parts
        )
        style = "One continuous natural utterance; the delivery changes mid-way. " + plan
        text = " ".join(p["text"] for p in parts)
        whole = speak(engine, SpeechRequest(text, voice="Kore", language="ru", style=style))
        save(folder / f"reply_{number}_whole.wav", whole)
        print(
            number,
            "parts",
            round(len(pcm) / 48000, 2),
            "с; whole",
            round(len(whole) / 48000, 2),
            "с",
            flush=True,
        )
        time.sleep(3)


if __name__ == "__main__":
    main()
