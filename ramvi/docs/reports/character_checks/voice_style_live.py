"""9.11: эмоция части ответа доходит до голоса (Gemini TTS, Mac).

Одни и те же слова синтезируются с нейтральной пометкой и с эмоциональной — так, как
персонаж передаёт её из части ответа (emotion, intensity, voice_style → style). Звук
сохраняется в WAV для прослушивания; печатаются длительность и громкость (разница в них —
не доказательство эмоции, слушать нужно ушами).

usage: voice_style_live.py <файл ключей> <папка для WAV> <отчёт.json>
"""

import json
import sys
import threading
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import read_keys  # noqa: E402
from ramvi_service.contracts.speech import SpeechRequest  # noqa: E402
from ramvi_service.speech.cloud import CloudSpeech  # noqa: E402

# (имя, текст, emotion, intensity, voice_style) — пример из 9.11 и нейтральные пары к нему.
CASES = [
    ("ok_neutral", "Ок!", "neutral", 0.2, "neutral"),
    ("ok_excited", "Ок!", "joy", 0.9, "excited"),
    ("try_neutral", "Давай попробуем.", "neutral", 0.2, "neutral"),
    ("try_intriguing", "Давай попробуем.", "interest", 0.7, "intriguing"),
]


def style(emotion: str, intensity: float, voice_style: str) -> str:
    """Как персонаж собирает пометку для голоса (ramvi_character/speech.py, _start)."""
    return f"Use a {emotion} tone at relative intensity {intensity:.2f}. Delivery: {voice_style}."


def stats(pcm: bytes, rate: int = 24000) -> dict[str, float]:
    samples = [
        int.from_bytes(pcm[i : i + 2], "little", signed=True) for i in range(0, len(pcm) - 1, 2)
    ]
    voiced = [abs(s) for s in samples if abs(s) > 500]
    rms = (sum(s * s for s in samples) / max(1, len(samples))) ** 0.5
    return {
        "seconds": round(len(samples) / rate, 2),
        "voiced_seconds": round(len(voiced) / rate, 2),
        "rms": round(rms, 1),
        "peak": max((abs(s) for s in samples), default=0),
    }


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    folder, out = Path(sys.argv[2]), Path(sys.argv[3])
    folder.mkdir(parents=True, exist_ok=True)
    providers = json.loads((ROOT / "service/ramvi_service/speech/providers.json").read_text())
    model = next(p["model"] for p in providers["providers"] if p["id"] == "gemini")
    report: dict[str, Any] = {"model": model, "cases": []}
    for name, text, emotion, intensity, voice_style in CASES:
        request = SpeechRequest(
            text, voice="Kore", language="ru", style=style(emotion, intensity, voice_style)
        )
        pcm = b"".join(
            p.pcm
            for p in CloudSpeech("gemini", model, keys["google"]).synthesize(
                request, threading.Event()
            )
        )
        with wave.open(str(folder / f"{name}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        row = {"name": name, "text": text, "style": request.style, **stats(pcm)}
        report["cases"].append(row)
        print(row, flush=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
