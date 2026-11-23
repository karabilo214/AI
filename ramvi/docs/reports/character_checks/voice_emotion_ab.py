"""9.11: слышна ли эмоция части ответа — сравнение способов передать её голосу.

Фразы владельца: «Бля!» — вскрик; «Ну какого хера ты тупишь?» — спокойнее, но с укором.
Каждая озвучивается: без пометки; Gemini с пометкой отдельным полем (как сейчас);
Gemini с указанием прямо в тексте; OpenAI TTS с instructions. WAV — для прослушивания,
замер: длительность, громкость, высота голоса (средняя и разброс) — разница в цифрах не
доказывает эмоцию, слушать ушами.

usage: voice_emotion_ab.py <файл ключей> <папка WAV> <отчёт.json>
"""

import json
import sys
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import read_keys  # noqa: E402
from ramvi_service.contracts.speech import SpeechRequest  # noqa: E402
from ramvi_service.speech.cloud import CloudSpeech  # noqa: E402

PHRASES = [
    (
        "shout",
        "Бля!",
        "a sudden loud startled exclamation, almost a shout",
        "Выкрикни резко и громко, как внезапный испуганный вскрик",
    ),
    (
        "reproach",
        "Ну какого хера ты тупишь?",
        "calmer, annoyed and reproachful, with a weary intonation",
        "Скажи спокойнее, но раздражённо и с укором, устало",
    ),
]


def pitch(pcm: bytes, rate: int = 24000) -> tuple[float, float]:
    """Средняя высота и разброс (Гц) по озвученным окнам, автокорреляция 70–400 Гц."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    size, step, found = 1024, 512, []
    for start in range(0, len(samples) - size, step):
        frame = samples[start : start + size]
        if np.sqrt(np.mean(frame**2)) < 500:
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, "full")[size - 1 :]
        low, high = rate // 400, rate // 70
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] > 0.3 * corr[0]:
            found.append(rate / lag)
    if not found:
        return 0.0, 0.0
    return round(float(np.mean(found)), 1), round(float(np.std(found)), 1)


def stats(pcm: bytes) -> dict[str, float]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    mean, spread = pitch(pcm)
    return {
        "seconds": round(len(samples) / 24000, 2),
        "rms": round(float(np.sqrt(np.mean(samples**2))), 1),
        "pitch_hz": mean,
        "pitch_spread_hz": spread,
    }


def save(folder: Path, name: str, pcm: bytes) -> None:
    with wave.open(str(folder / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    folder, out = Path(sys.argv[2]), Path(sys.argv[3])
    folder.mkdir(parents=True, exist_ok=True)
    providers = json.loads((ROOT / "service/ramvi_service/speech/providers.json").read_text())
    models = {p["id"]: p["model"] for p in providers["providers"]}
    gemini = CloudSpeech("gemini", models["gemini"], keys["google"])
    openai = CloudSpeech("openai", models["openai"], keys["openai"])
    rows: list[dict[str, Any]] = []
    for name, text, style_en, style_ru in PHRASES:
        variants = {
            "gemini_plain": (gemini, SpeechRequest(text, voice="Kore", language="ru")),
            "gemini_field": (
                gemini,
                SpeechRequest(text, voice="Kore", language="ru", style=style_en),
            ),
            "gemini_in_text": (
                gemini,
                SpeechRequest(f"{style_ru}: {text}", voice="Kore", language="ru"),
            ),
            "openai_plain": (openai, SpeechRequest(text, voice="onyx", language="ru")),
            "openai_instructions": (
                openai,
                SpeechRequest(text, voice="onyx", language="ru", style=style_en),
            ),
        }
        for variant, (engine, request) in variants.items():
            try:
                pcm = b"".join(p.pcm for p in engine.synthesize(request, threading.Event()))
            except Exception as error:  # noqa: BLE001 — сравнение продолжается
                rows.append({"phrase": name, "variant": variant, "error": str(error)[:200]})
                print(rows[-1], flush=True)
                continue
            save(folder, f"{name}_{variant}", pcm)
            rows.append({"phrase": name, "variant": variant, **stats(pcm)})
            print(rows[-1], flush=True)
    out.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
