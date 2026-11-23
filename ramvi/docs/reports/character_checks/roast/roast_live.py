"""Профиль «Подкольщик»: живой диалог на облачной модели (Mac, без NVIDIA).

Персонаж RAMVI (профиль roaster, мат разрешён) разговаривает с «собеседником», которого
играет дешёвая облачная модель: парень учит английский, хвастается, ошибается, противоречит
себе. Проверяется глазами по стенограмме: сам ли персонаж подкалывает и начинает игры,
ловит ли промахи, придумывает ли реакцию именно на случившееся, как матерится, возвращается
ли к прежним промахам. Ключи — из файла владельца только в память.

usage: roast_live.py <файл ключей> <токенизатор Qwen3> <модель персонажа> <отчёт.md> [ходов] [мат 1-3]
модель персонажа: подстрока имени модели OpenAI (gpt-5.5, gpt-5.4-mini) или claude:<подстрока>.
Собеседника играет дешёвая модель OpenAI (gpt-5.4-mini).
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import Secrets, read_keys  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

FRIEND = """Ты — Дима, 30 лет, переписываешься с приятелем в чате. Ты учишь английский и любишь
похвастаться, но часто ошибаешься: путаешь ложных друзей переводчика (magazine — магазин,
sympathetic — симпатичный, accurate — аккуратный, fabric — фабрика), путаешь факты, споришь,
противоречишь сам себе, иногда несёшь чушь с уверенным видом. На подколы огрызаешься или
оправдываешься, но не обижаешься всерьёз. Пиши коротко, 1–2 фразы, разговорно, по-русски,
без смайликов. Ты не ассистент и не помогаешь — ты просто Дима."""
OPENING = "Здорово. Скучно мне, давай поболтаем."
CLOSING = "Ладно, мне работать пора, давай."


def pick(client: Client, keys: dict[str, str], choice: str) -> str:
    """Модель проверки: по умолчанию OpenAI (подстрока имени, например gpt-5.5); Claude —
    только явно, claude:<подстрока> (владелец 2026-11-23: кредиты Claude на исходе)."""
    if choice.startswith("claude:"):
        return connect(client, keys, "anthropic", choice.removeprefix("claude:"))
    return connect(client, keys, "openai", choice.removeprefix("openai:"))


def connect(client: Client, keys: dict[str, str], kind: str, hint: str) -> str:
    connection = client.ok(
        "POST", "/cloud/connections", {"kind": kind, "name": kind, "key": keys[kind]}
    )["connection"]
    names = client.ok("POST", "/cloud/models", {"id": connection["id"]})["models"]
    # Точное имя, иначе самое короткое совпадение: «gpt-5.5» — не дорогая «gpt-5.5-pro-…».
    model = hint if hint in names else min((n for n in names if hint in n), key=len)
    client.ok(
        "POST", f"/cloud/connections/{connection['id']}/models", {"model": model, "enabled": True}
    )
    return f"cloud:{connection['id']}:{model}"


def say(client: Client, path: str, text: str, model: str) -> tuple[str, float, Any]:
    started = time.monotonic()
    client.ok("POST", path, {"operation": "start", "arguments": {"text": text, "model": model}})
    while True:
        job = client.ok("POST", path, {"operation": "poll"})["job"]
        if job and job["state"] != "running" and job.get("settled", True):
            break
        time.sleep(0.3)
    reply = " ".join(e["text"] for e in job["events"] if e.get("type") == "segment")
    return reply, round(time.monotonic() - started, 1), job.get("error_detail")


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    os.environ["RAMVI_TOKENIZER"] = sys.argv[2]
    choice, out = sys.argv[3], Path(sys.argv[4])
    turns = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    level = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    app = Path(tempfile.mkdtemp(prefix="ramvi-roast-"))
    shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    lines: list[str] = []
    try:
        friend_id = connect(client, keys, "openai", "gpt-5.4-mini")
        model = (
            friend_id
            if choice in ("gpt-5.4-mini", "openai:gpt-5.4-mini")
            else pick(client, keys, choice)
        )
        friend = server.cloud.provider(server.chat.models_repo.get(friend_id))
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        client.ok("POST", path, {"operation": "profile", "arguments": {"id": "roaster"}})
        client.ok(
            "POST",
            path,
            {"operation": "boundary", "arguments": {"name": "profanity", "allowed": True}},
        )
        client.ok("POST", path, {"operation": "profanity_level", "arguments": {"level": level}})
        lines += [
            f"# Подкольщик — {model.rsplit(':', 1)[-1]}",
            "",
            f"Собеседник: {friend_id.rsplit(':', 1)[-1]}, мат: {level}",
            "",
        ]
        # Для модели-собеседника её реплики — assistant, реплики персонажа — user.
        talk: list[dict[str, str]] = [{"role": "user", "content": "(чат открыт)"}]
        text = OPENING
        for turn in range(turns + 1):
            if turn == turns:
                text = CLOSING
            lines.append(f"**Дима:** {text}")
            reply, seconds, error = say(client, path, text, model)
            lines.append(f"**RAMVI** ({seconds} с): {reply or '[нет ответа: ' + str(error) + ']'}")
            lines.append("")
            print(lines[-3], "\n", lines[-2], flush=True)
            if turn == turns:
                break
            talk += [{"role": "assistant", "content": text}, {"role": "user", "content": reply}]
            answer = friend.complete(
                {"messages": [{"role": "system", "content": FRIEND}, *talk], "max_tokens": 1500},
                threading.Event(),
            )
            text = answer["choices"][0]["message"]["content"].strip()
        usage = client.ok("GET", "/cloud").get("spending")
        lines += ["", "```", json.dumps(usage, ensure_ascii=False), "```"]
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        shutil.rmtree(app, ignore_errors=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
