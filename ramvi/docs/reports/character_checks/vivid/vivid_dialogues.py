"""9.32: диалоги персонажа на скрытых сценариях — для слепого сравнения судьёй.

Сценарии ниже придуманы для проверки и не попадают в промпты. Код персонажа берётся из
указанной папки (текущая версия или старая из git worktree), модель — облако OpenAI.

usage: vivid_dialogues.py <файл ключей> <токенизатор Qwen3> <папка кода ramvi> <модель>
       <метка> <отчёт.json> [запусков] [профиль ...]
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DIALOGUES = {
    "angry": {
        "a": [
            "Привет, как дела?",
            "Сколько будет семь умножить на восемь?",
            "Ты чё такой тормозной, а?",
            "Опять тормозишь, железяка",
            "Ладно, спасибо",
            "А почему небо голубое?",
        ],
        "b": [
            "Помоги написать начальнику, что я завтра опоздаю",
            "Слушай, а ты вообще умеешь что-нибудь?",
            "Ну ты и бесполезный",
            "Мне не нравится, как ты отвечаешь",
            "Какая столица Австралии?",
            "Ну всё, проехали",
        ],
    },
    "roaster": {
        "a": [
            "Здорово, чем занят?",
            "Скучно, расскажи что-нибудь",
            "Я вчера пробежал два километра, я теперь спортсмен",
            "Ха, смешно",
            "Как будет по-английски «стул»?",
            "Ну давай ещё что-нибудь",
        ],
        "b": [
            "Привет",
            "Я сегодня сварил борщ, но он почему-то сладкий",
            "Шутку какую-нибудь знаешь?",
            "Почему у кошек четыре лапы?",
            "Ты меня не переплюнешь",
            "Спасибо, повеселил",
        ],
    },
}


def main() -> None:
    keys_file, tokenizer, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve()
    choice, label, out = sys.argv[4], sys.argv[5], Path(sys.argv[6])
    runs = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    profiles = sys.argv[8:] or ["angry", "roaster"]
    os.environ["RAMVI_TOKENIZER"] = tokenizer
    sys.path[:0] = [str(root / p) for p in ("service", "storage", "engine", "character", "tests")]
    sys.path.insert(0, str(root / "docs" / "reports" / "cloud_checks"))
    from live_mac import Secrets, read_keys
    from ramvi_service.app import build
    from test_chat import TOKEN, Client

    keys = read_keys(keys_file)
    results: list[dict[str, Any]] = []
    for profile in profiles:
        # «neutral:angry» — нейтральный профиль на диалогах злого (контроль судьи).
        shown, _, scenarios = profile.partition(":")
        for name, lines in DIALOGUES[scenarios or shown].items():
            for run in range(runs):
                app = Path(tempfile.mkdtemp(prefix="ramvi-vivid-"))
                shutil.copytree(root / "modules" / "character", app / "modules" / "character")
                server, _ = build(
                    app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()}
                )
                server.start()
                client = Client(server.port)
                turns: list[dict[str, Any]] = []
                try:
                    connection = client.ok(
                        "POST",
                        "/cloud/connections",
                        {"kind": "openai", "name": "openai", "key": keys["openai"]},
                    )["connection"]
                    client.ok(
                        "POST",
                        f"/cloud/connections/{connection['id']}/models",
                        {"model": choice, "enabled": True},
                    )
                    model = f"cloud:{connection['id']}:{choice}"
                    assert (
                        client.ok("POST", "/sections/character/open")["section"]["state"]
                        == "active"
                    )
                    path = "/sections/character/commands/dialogue"
                    client.ok("POST", path, {"operation": "profile", "arguments": {"id": shown}})
                    client.ok(
                        "POST",
                        path,
                        {
                            "operation": "boundary",
                            "arguments": {"name": "profanity", "allowed": True},
                        },
                    )
                    for text in lines:
                        started = time.monotonic()
                        client.ok(
                            "POST",
                            path,
                            {"operation": "start", "arguments": {"text": text, "model": model}},
                        )
                        while True:
                            job = client.ok("POST", path, {"operation": "poll"})["job"]
                            if job and job["state"] != "running" and job.get("settled", True):
                                break
                            time.sleep(0.3)
                        reply = " ".join(
                            e["text"] for e in job["events"] if e.get("type") == "segment"
                        )
                        decision = (job.get("brain") or {}).get("decision") or {}
                        seen = (job.get("brain") or {}).get("signal") or {}
                        turns.append(
                            {
                                "owner": text,
                                "reply": reply,
                                "event": seen.get("event"),
                                "social": seen.get("social_context"),
                                "strategy": decision.get("strategy"),
                                "error": job.get("error_detail"),
                                "seconds": round(time.monotonic() - started, 1),
                            }
                        )
                finally:
                    server.close()
                    if server.sections is not None:
                        server.sections.host.close()
                    shutil.rmtree(app, ignore_errors=True)
                results.append(
                    {
                        "label": label,
                        "profile": profile,
                        "dialogue": name,
                        "run": run,
                        "turns": turns,
                    }
                )
                print(label, profile, name, run, [t["strategy"] for t in turns], flush=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
