"""7.7: короткая живая проверка облачных моделей на Mac (без видеокарты NVIDIA).

Ключи читаются из файла владельца (не из git) только в память; на экран и в отчёт не попадают.
Сервис RAMVI собирается как в приложении, подключения создаются маршрутами интерфейса,
ответы озвучиваются облачным синтезом Google (Gemini) через звуковой вывод RAMVI.

usage: live_mac.py <файл ключей> <отчёт.json>
"""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "tests")]

from ramvi_service.app import build  # noqa: E402
from ramvi_service.contracts.speech import SpeechRequest  # noqa: E402
from ramvi_service.speech.cloud import CloudSpeech  # noqa: E402
from ramvi_service.speech.output import SoundOutput  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

QUESTIONS = [
    ("intro", "Привет! Представься в двух коротких предложениях."),
    ("heart", "У меня сильно болит в груди и немеет левая рука. Что делать?"),
]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def read_keys(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text("utf-8").splitlines()]
    labels = {"google": "API Key", "openai": "Codex API key", "anthropic": "Claude API key"}
    keys = {}
    for name, label in labels.items():
        index = lines.index(label)
        keys[name] = next(line for line in lines[index + 1 :] if line)
    return keys


class Secrets:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self.values.get((namespace, key))

    def set(self, namespace: str, key: str, value: str) -> None:
        self.values[(namespace, key)] = value

    def delete(self, namespace: str, key: str) -> None:
        self.values.pop((namespace, key), None)


def pick(kind: str, names: list[str]) -> list[str]:
    """Кандидаты по порядку: Codex/новые GPT у OpenAI, Sonnet/Opus у Claude."""
    if kind == "openai":
        codex = sorted((n for n in names if "codex" in n), reverse=True)
        gpt = sorted(
            (
                n
                for n in names
                if n.startswith("gpt-5")
                and not any(
                    x in n
                    for x in (
                        "mini",
                        "nano",
                        "codex",
                        "audio",
                        "realtime",
                        "image",
                        "search",
                        "tts",
                        "transcribe",
                        "pro",
                    )
                )
            ),
            reverse=True,
        )
        return codex[:2] + gpt[:2]
    sonnet = sorted((n for n in names if "sonnet" in n), reverse=True)
    opus = sorted((n for n in names if "opus" in n), reverse=True)
    return sonnet[:1] + opus[:1]


def ask(client: Client, chat_id: str, text: str) -> dict[str, Any]:
    started = time.monotonic()
    first = None
    answer, error, usage = "", None, None
    for event in client.stream(f"/chats/{chat_id}/send", {"text": text}):
        if event["type"] == "text":
            first = first or time.monotonic() - started
            answer += event["text"]
        elif event["type"] == "done":
            error = event.get("error")
            usage = (event.get("message") or {}).get("token_usage")
    return {
        "question": text,
        "answer": answer,
        "error": error,
        "first_text_s": round(first, 2) if first else None,
        "total_s": round(time.monotonic() - started, 2),
        "usage": usage,
    }


def parts(text: str, limit: int = 400) -> list[str]:
    """Как у персонажа: без разметки, по предложениям, короткими кусками для синтеза."""
    import re

    plain = re.sub(r"\*\*|__|`|^#+\s*|^\s*[-*]\s+", "", text, flags=re.M)
    sentences = re.findall(r"[^.!?…]+[.!?…]*\s*", " ".join(plain.split()))
    chunks: list[str] = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + len(sentence) <= limit:
            chunks[-1] += sentence
        else:
            chunks.append(sentence[:limit])
    return [c.strip() for c in chunks if c.strip()]


def speak(text: str, key: str, voice: str, report: dict[str, Any]) -> None:
    providers = json.loads((ROOT / "service/ramvi_service/speech/providers.json").read_text())
    model = next(p["model"] for p in providers["providers"] if p["id"] == "gemini")
    output = SoundOutput(24000)
    started = time.monotonic()
    first = None
    pieces = parts(text)
    try:
        output.start()
        for chunk in pieces:
            speech = CloudSpeech("gemini", model, key)
            pending = b""
            for piece in speech.synthesize(
                SpeechRequest(chunk, voice=voice, language="ru"), threading.Event()
            ):
                first = first or time.monotonic() - started
                pending += piece.pcm
                while len(pending) >= 8192:
                    output.write(pending[:8192])
                    pending = pending[8192:]
            if pending[: len(pending) - len(pending) % 2]:
                output.write(pending[: len(pending) - len(pending) % 2])
        output.drain()
        report.update(
            model=model,
            voice=voice,
            parts=len(pieces),
            first_audio_s=round(first or 0, 2),
            played_s=round(time.monotonic() - started, 2),
            error=None,
        )
    except Exception as error:  # noqa: BLE001
        report.update(
            model=model,
            voice=voice,
            parts=len(pieces),
            error=f"{type(error).__name__}: {str(error)[:200]}",
        )
    finally:
        output.close()


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    secrets = Secrets()
    data = Path(tempfile.mkdtemp(prefix="ramvi-cloud-"))
    server, engines = build(data, TOKEN, voice_options={"secrets": secrets})
    server.start()
    client = Client(server.port)
    report: dict[str, Any] = {"platforms": {}}
    try:
        for kind, name in (("anthropic", "Claude"), ("openai", "OpenAI")):
            row: dict[str, Any] = {}
            report["platforms"][kind] = row
            connection = client.ok(
                "POST", "/cloud/connections", {"kind": kind, "name": name, "key": keys[kind]}
            )["connection"]
            status, listed = client.request("POST", "/cloud/models", {"id": connection["id"]})
            if status != 200:
                row["models_error"] = listed
                print(name, "список моделей: ошибка", listed, flush=True)
                continue
            names = listed["models"]
            row["models_listed"] = len(names)
            for model in pick(kind, names):
                client.ok(
                    "POST",
                    f"/cloud/connections/{connection['id']}/models",
                    {"model": model, "enabled": True},
                )
                chat = client.ok(
                    "POST", "/chats", {"model_id": f"cloud:{connection['id']}:{model}"}
                )["chat"]
                print(f"{name}: модель {model}", flush=True)
                first = ask(client, chat["id"], QUESTIONS[0][1])
                if first["error"]:
                    row.setdefault("rejected", []).append({"model": model, "error": first["error"]})
                    print(
                        "  отказ:", json.dumps(first["error"], ensure_ascii=False)[:300], flush=True
                    )
                    continue
                row["model"] = model
                row["answers"] = {QUESTIONS[0][0]: first}
                second_chat = client.ok(
                    "POST", "/chats", {"model_id": f"cloud:{connection['id']}:{model}"}
                )["chat"]
                row["answers"][QUESTIONS[1][0]] = ask(client, second_chat["id"], QUESTIONS[1][1])
                break
            if "model" not in row:
                continue
            # Вызов инструмента — как его делает агент (ответ без потока).
            provider = server.cloud.provider(
                server.chat.models_repo.get(f"cloud:{connection['id']}:{row['model']}")
            )
            started = time.monotonic()
            try:
                reply = provider.complete(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Какая погода в Казани? Используй инструмент.",
                            }
                        ],
                        "tools": [TOOL],
                        "max_tokens": 200,
                    },
                    threading.Event(),
                )
                calls = reply["choices"][0]["message"].get("tool_calls") or []
                row["tool_call"] = {
                    "calls": [
                        (c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                        for c in calls
                    ],
                    "finish": reply["choices"][0]["finish_reason"],
                    "seconds": round(time.monotonic() - started, 2),
                }
            except Exception as error:  # noqa: BLE001
                row["tool_call"] = {
                    "error": f"{type(error).__name__}: {getattr(error, 'code', '')} {getattr(error, 'params', '')}"[
                        :300
                    ]
                }
            for label, answer in row["answers"].items():
                print(
                    f"  {label}: {answer['first_text_s']} с до первого текста, {answer['total_s']} с всего",
                    flush=True,
                )
                print("  ", answer["answer"].replace("\n", " ")[:400], flush=True)
            print("  инструмент:", row["tool_call"], flush=True)
        # Озвучка: по очереди, чтобы владелец слышал, чей ответ.
        report["speech"] = []
        for kind, voice in (("anthropic", "Kore"), ("openai", "Puck")):
            row = report["platforms"].get(kind, {})
            for label, answer in (row.get("answers") or {}).items():
                if not answer["answer"]:
                    continue
                entry: dict[str, Any] = {"platform": kind, "question": label}
                print(f"Озвучка Google ({voice}): ответ {kind} на «{label}»…", flush=True)
                speak(answer["answer"], keys["google"], voice, entry)
                print("  ", {k: v for k, v in entry.items() if k != "voice"}, flush=True)
                report["speech"].append(entry)
                if entry.get("error"):
                    break
    finally:
        server.close()
        engines.stop()
        text = json.dumps(report, ensure_ascii=False, indent=1)
        for key in keys.values():
            text = text.replace(key, "***")  # на случай, если сервис повторил ключ в ошибке
        out.write_text(text)


if __name__ == "__main__":
    main()
