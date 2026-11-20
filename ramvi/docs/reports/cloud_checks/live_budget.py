"""7.10: живая проверка учёта — токены облачного ответа, стоимость по цене, лимит.

Цена — условная, вписана для проверки (источник «проверка RAMVI»): проверяется арифметика
и остановка по лимиту, а не тариф сервиса.

usage: live_budget.py <файл ключей> <отчёт.json>
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "tests")]

from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from live_mac import Secrets, read_keys  # noqa: E402

PRICE = {"input": 3.0, "output": 15.0, "source": "проверка RAMVI (условная цена)"}


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    server, engines = build(
        Path(tempfile.mkdtemp(prefix="ramvi-budget-")), TOKEN, voice_options={"secrets": Secrets()}
    )
    server.start()
    report: dict[str, Any] = {}
    try:
        client = Client(server.port)
        client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "anthropic", "name": "claude", "key": keys["anthropic"]},
        )
        client.ok(
            "POST",
            "/cloud/connections/claude/models",
            {"model": "claude-sonnet-5-5", "enabled": True},
        )
        client.ok(
            "POST", "/cloud/connections/claude/prices", {"model": "claude-sonnet-5-5", **PRICE}
        )
        answers = []
        for text in ("Ответь одним словом: столица Франции?", "Назови одно число от 1 до 10."):
            chat = client.ok("POST", "/chats", {"model_id": "cloud:claude:claude-sonnet-5-5"})[
                "chat"
            ]
            events = client.events(f"/chats/{chat['id']}/send", {"text": text})
            done = events[-1]
            answers.append(
                {
                    "text": done["message"]["content"],
                    "token_usage": done["message"].get("token_usage"),
                }
            )
        spending = client.ok("GET", "/cloud")["spending"]
        recorded = list(reversed(spending["recent"]))
        report["answers"] = answers
        report["recorded"] = recorded
        report["today"] = spending["today"]
        report["expected_cost"] = [
            (r["prompt_tokens"] * PRICE["input"] + r["completion_tokens"] * PRICE["output"])
            / 1_000_000
            for r in recorded
        ]
        # Лимит ниже уже потраченного: следующий запрос не уходит.
        client.ok("POST", "/cloud/limits", {"daily": spending["today"] / 2})
        chat = client.ok("POST", "/chats", {"model_id": "cloud:claude:claude-sonnet-5-5"})["chat"]
        events = client.events(f"/chats/{chat['id']}/send", {"text": "Ещё вопрос"})
        report["after_limit"] = {
            "error": events[-1].get("error"),
            "recorded": len(client.ok("GET", "/cloud")["spending"]["recent"]),
        }
        print(json.dumps(report, ensure_ascii=False, indent=1))
    finally:
        server.close()
        engines.stop()
        text = json.dumps(report, ensure_ascii=False, indent=1)
        for secret in keys.values():
            text = text.replace(secret, "***")
        Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
