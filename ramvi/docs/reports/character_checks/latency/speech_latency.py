"""9.23: первый звук и перебивание через настоящий SpeechService и звуковой вывод RAMVI (Mac).

Первый звук — от `start` задания до первой записи PCM на устройство (`SoundOutput.write`), для
системного голоса, Google (Gemini TTS) и OpenAI TTS (поток). Перебивание — от `cancel`
звучащего ответа до того, как вывод остановлен и новых записей нет (системный голос).
Облачных запросов — по `RUNS` коротких фраз на голос; ключи — из файла владельца, только в память.

usage: speech_latency.py <файл ключей> <отчёт.json> [голоса через запятую, по умолчанию все]
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine")]

from ramvi_service.contracts.modules import Authority  # noqa: E402
from ramvi_service.contracts.speech import SPEAKER  # noqa: E402
from ramvi_service.resources import ResourceBroker  # noqa: E402
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets  # noqa: E402
from ramvi_service.speech.native import SystemSpeech  # noqa: E402
from ramvi_service.speech.output import SoundOutput  # noqa: E402
from ramvi_service.speech.service import SpeechService  # noqa: E402

RUNS = 3
PAUSE = float(__import__("os").environ.get("RAMVI_TTS_PAUSE", "0"))
PHRASES = [
    "Привет! Рад тебя слышать.",
    "Сейчас расскажу, что получилось.",
    "Хороший вопрос, давай разберёмся.",
]
LONG = "Сейчас звучит длинный ответ, который мы перебьём на середине. " * 6
outputs: list["Recorded"] = []


class Recorded:
    def __init__(self, rate: int) -> None:
        self.backend = SoundOutput(rate)
        self.first: float | None = None
        self.last: float | None = None
        self.cancelled: float | None = None
        self.writes = 0
        outputs.append(self)

    def start(self) -> None:
        self.backend.start()

    def write(self, pcm: bytes) -> None:
        self.backend.write(pcm)
        now = time.monotonic()
        self.first = self.first or now
        self.last = now
        self.writes += 1

    def playback_view(self, alignment: Any = None) -> dict[str, Any]:
        return self.backend.playback_view(alignment)

    def drain(self) -> None:
        self.backend.drain()

    def interrupt(self) -> None:
        self.backend.interrupt()

    def cancel(self) -> None:
        self.backend.cancel()
        self.cancelled = time.monotonic()

    def close(self) -> None:
        self.backend.close()


def read_keys(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text("utf-8").splitlines()]
    keys = {}
    for name, label in (("gemini", "API Key"), ("openai", "Codex API key")):
        index = lines.index(label)
        keys[name] = next(line for line in lines[index + 1 :] if line)
    return keys


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    report: dict[str, Any] = {"first_audio_s": {}, "interrupt": []}
    native = SystemSpeech()
    voice = next((v["id"] for v in native.voices() if "ru" in v["languages"]), "")
    native.close()
    with tempfile.TemporaryDirectory() as folder:
        authority = Authority()
        owner = authority.issue("character", "user", "latency")
        broker = ResourceBroker(authority, {SPEAKER: 1})
        data = ScopedData(Path(folder), "character")
        secrets = ScopedSecrets(MemorySecrets(), "character")
        for provider, key in keys.items():
            secrets.set("tts_" + provider, key)
        service = SpeechService(broker, output_factory=Recorded)
        generation = 0

        def wait(identifier: str, predicate: Any, limit: float = 90) -> dict[str, Any]:
            until = time.monotonic() + limit
            while True:
                job = service.status(owner, identifier)["job"]
                if predicate(job) or time.monotonic() > until:
                    return job
                time.sleep(0.005)

        try:
            only = sys.argv[3].split(",") if len(sys.argv) > 3 else ["system", "gemini", "openai"]
            for provider, chosen in (("system", voice), ("gemini", "Kore"), ("openai", "coral")):
                if provider not in only:
                    continue
                service.configure(
                    owner,
                    data,
                    {
                        "enabled": True,
                        "provider": provider,
                        "voice": chosen,
                        "remote_text_enabled": provider != "system",
                    },
                )
                rows = []
                for text in PHRASES[:RUNS]:
                    generation += 1
                    service.advance(owner, generation, f"r{generation}")
                    started = time.monotonic()
                    job_id = service.start(
                        owner,
                        data,
                        secrets,
                        {
                            "generation": generation,
                            "response_id": f"r{generation}",
                            "segment_id": "s1",
                            "request": {"text": text},
                        },
                    )["job_id"]
                    job = wait(job_id, lambda j: j["settled"])
                    out = outputs[-1]
                    rows.append(
                        {
                            "state": job["state"],
                            "error": job.get("error"),
                            "first_audio_s": round(out.first - started, 3) if out.first else None,
                            "total_s": round((out.last or started) - started, 3),
                        }
                    )
                    print(provider, rows[-1], flush=True)
                    if provider != "system":
                        time.sleep(
                            PAUSE
                        )  # бесплатный лимит облачного голоса — несколько запросов в минуту
                firsts = [r["first_audio_s"] for r in rows if r["first_audio_s"] is not None]
                report["first_audio_s"][provider] = {
                    "runs": rows,
                    "median": round(statistics.median(firsts), 3) if firsts else None,
                }
            # Перебивание: системный голос, ответ звучит — cancel — вывод остановлен.
            if "system" in only:
                service.configure(
                    owner,
                    data,
                    {
                        "enabled": True,
                        "provider": "system",
                        "voice": voice,
                        "remote_text_enabled": False,
                    },
                )
                for _ in range(5):
                    generation += 1
                    service.advance(owner, generation, f"long{generation}")
                    job_id = service.start(
                        owner,
                        data,
                        secrets,
                        {
                            "generation": generation,
                            "response_id": f"long{generation}",
                            "segment_id": "s1",
                            "request": {"text": LONG},
                        },
                    )["job_id"]
                    wait(job_id, lambda j: j["state"] == "playing" and outputs[-1].writes > 3)
                    time.sleep(0.5)  # звучит
                    out = outputs[-1]
                    writes = out.writes
                    pressed = time.monotonic()
                    service.cancel(owner, job_id)
                    returned = time.monotonic()
                    job = wait(job_id, lambda j: j["settled"], 10)
                    settled = time.monotonic()
                    time.sleep(0.3)
                    report["interrupt"].append(
                        {
                            "cancel_call_ms": round((returned - pressed) * 1000, 1),
                            "settled_ms": round((settled - pressed) * 1000, 1),
                            "writes_after_cancel": out.writes - writes
                            if out.last and out.last > returned
                            else 0,
                            "state": job["state"],
                        }
                    )
                    print("interrupt", report["interrupt"][-1], flush=True)
            report["leases_after"] = broker.status()["leases"]
        finally:
            service.close()
    text = json.dumps(report, ensure_ascii=False, indent=1)
    for key in keys.values():
        text = text.replace(key, "***")
    Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
