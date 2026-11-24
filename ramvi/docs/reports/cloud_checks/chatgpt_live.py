"""Вход через ChatGPT: живая проверка через сервис RAMVI и официальный Codex CLI (подписка владельца).

Сервис: состояние входа, подключение chatgpt без ключа, модели подписки, ответ персонажа
(профиль «Злой») на двух репликах, отказ агенту с инструментами (понятная ошибка).
Нужно: codex установлен и вошёл через ChatGPT.

usage: chatgpt_live.py <токенизатор Qwen3> <модель подписки> <отчёт.json> [мат 0-3] [реплика ...]
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import Secrets  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402


def main() -> None:
    os.environ["RAMVI_TOKENIZER"] = sys.argv[1]
    choice, out = sys.argv[2], Path(sys.argv[3])
    level = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    lines = sys.argv[5:] or ["Привет, как дела?", "Ты чё такой тормозной, а?"]
    app = Path(tempfile.mkdtemp(prefix="ramvi-chatgpt-"))
    shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    report: dict[str, Any] = {}
    try:
        report["state"] = client.ok("GET", "/cloud/chatgpt")
        connection = client.ok(
            "POST", "/cloud/connections", {"kind": "chatgpt", "name": "ChatGPT"}
        )["connection"]
        names = client.ok("POST", "/cloud/models", {"id": connection["id"]})["models"]
        report["models"] = names
        client.ok(
            "POST",
            f"/cloud/connections/{connection['id']}/models",
            {"model": choice, "enabled": True},
        )
        model = f"cloud:{connection['id']}:{choice}"
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        client.ok("POST", path, {"operation": "profile", "arguments": {"id": "angry"}})
        if level:
            client.ok(
                "POST",
                path,
                {"operation": "boundary", "arguments": {"name": "profanity", "allowed": True}},
            )
            client.ok("POST", path, {"operation": "profanity_level", "arguments": {"level": level}})
        turns = []
        for text in lines:
            started = time.monotonic()
            client.ok(
                "POST", path, {"operation": "start", "arguments": {"text": text, "model": model}}
            )
            while True:
                job = client.ok("POST", path, {"operation": "poll"})["job"]
                if job and job["state"] != "running" and job.get("settled", True):
                    break
                time.sleep(0.3)
            reply = " ".join(e["text"] for e in job["events"] if e.get("type") == "segment")
            decision = (job.get("brain") or {}).get("decision") or {}
            turns.append(
                {
                    "owner": text,
                    "reply": reply,
                    "strategy": decision.get("strategy"),
                    "error": job.get("error_detail"),
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
            print(turns[-1], flush=True)
        report["character"] = turns
        report["spending"] = client.ok("GET", "/cloud")["spending"]["recent"][:4]
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        shutil.rmtree(app, ignore_errors=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({k: report[k] for k in ("state", "models")})


if __name__ == "__main__":
    main()
