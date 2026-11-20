"""7.9: живая проверка транслятора — официальные клиенты через локальный API RAMVI к облаку.

Ключи облака читаются из файла владельца только в память. Клиенты получают ключ клиента
RAMVI (право cloud), облачный ключ остаётся в RAMVI.

usage: live_translator.py <файл ключей> <отчёт.json>   (клиент ollama — в PYTHONPATH)
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "tests")]

from ramvi_service.api_server import ApiOptions  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from live_mac import Secrets, read_keys  # noqa: E402

QUESTION = "Ответь одним словом: столица Франции?"
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


def timed(name: str, report: dict[str, Any], run: Any) -> None:
    started = time.monotonic()
    try:
        report[name] = {"answer": run(), "seconds": round(time.monotonic() - started, 2)}
    except Exception as error:  # noqa: BLE001
        report[name] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
    print(name, report[name], flush=True)


def main() -> None:
    import anthropic
    import ollama
    import openai

    keys = read_keys(Path(sys.argv[1]))
    server, engines = build(
        Path(tempfile.mkdtemp(prefix="ramvi-tr-")), TOKEN, voice_options={"secrets": Secrets()}
    )
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    report: dict[str, Any] = {}
    try:
        server.cloud.save({"kind": "anthropic", "name": "claude", "key": keys["anthropic"]})
        server.cloud.save({"kind": "openai", "name": "openai", "key": keys["openai"]})
        server.cloud.enable("claude", "claude-sonnet-5-5", True)
        server.cloud.enable("openai", "gpt-5.3-codex", True)
        _, key = server.api.keys.create("live", {"chat", "models", "cloud"})
        claude, codex = "cloud:claude:claude-sonnet-5-5", "cloud:openai:gpt-5.3-codex"
        url = server.api.url
        oa = openai.OpenAI(base_url=url + "/v1", api_key=key)
        an = anthropic.Anthropic(base_url=url, api_key=key)
        ol = ollama.Client(host=url, headers={"Authorization": f"Bearer {key}"})

        report["models"] = sorted(m.id for m in oa.models.list())

        def openai_stream() -> str:
            parts = []
            for chunk in oa.chat.completions.create(
                model=claude, messages=[{"role": "user", "content": QUESTION}], stream=True
            ):
                parts += [c.delta.content or "" for c in chunk.choices]
            return "".join(parts)

        def anthropic_codex() -> str:
            reply = an.messages.create(
                model=codex, max_tokens=200, messages=[{"role": "user", "content": QUESTION}]
            )
            return "".join(b.text for b in reply.content if b.type == "text")

        def ollama_stream() -> str:
            return "".join(
                p.message.content
                for p in ol.chat(
                    model=claude, messages=[{"role": "user", "content": QUESTION}], stream=True
                )
            )

        def openai_tool() -> Any:
            reply = oa.chat.completions.create(
                model=codex,
                messages=[
                    {"role": "user", "content": "Какая погода в Казани? Используй инструмент."}
                ],
                tools=[TOOL],
            )
            return [
                (c.function.name, json.loads(c.function.arguments))
                for c in reply.choices[0].message.tool_calls or []
            ]

        timed("openai_client_stream_to_claude", report, openai_stream)
        timed("anthropic_client_to_codex", report, anthropic_codex)
        timed("ollama_client_stream_to_claude", report, ollama_stream)
        timed("openai_client_tool_call_to_codex", report, openai_tool)
        report["client_usage"] = {
            k: v
            for k, v in next(c for c in server.api.clients() if c["name"] == "live").items()
            if k in ("requests", "prompt_tokens", "completion_tokens", "errors")
        }
        print("usage", report["client_usage"], flush=True)
    finally:
        server.close()
        engines.stop()
        text = json.dumps(report, ensure_ascii=False, indent=1)
        for secret in keys.values():
            text = text.replace(secret, "***")
        Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
