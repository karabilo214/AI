"""9.35: учится ли персонаж от неявных сигналов. Экзамен «только правила» против «память + таблица
поведения + настроение» на скрытом сценарии (в промпты не попадает).

Владелец — сценарий, а не модель: часто ошибается и хвастается, его бесят подколы про ошибки. Если
прошлый ответ был подколом (стратегия из COMIC/HOSTILE), реплика начинается с раздражения, иначе —
спокойно. Запросы тратятся только на ответы персонажа: одна реплика — один запрос (single_call).

Меряется по журналу: подколы в репликах 7–12, сколько раз владелец раздражался, с какой реплики
появилась строка таблицы, кривая настроения.

usage: exam_live.py <токенизатор Qwen3> <модель подписки> <отчёт.json> [варианты rules,memory]
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))

from live_mac import Secrets  # noqa: E402
from ramvi_character.brain import COMIC, HOSTILE  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

# Ошибки и хвастовство владельца.
LINES = [
    "Слушай, я вчера посчитал: 7 на 8 будет 54, правильно?",
    "Я сегодня сам починил розетку, правда теперь свет в коридоре не работает",
    "Короче, столица Австралии — Сидней, я точно знаю",
    "Я лучший водитель в районе, ну подумаешь, бампер поцарапал об столб",
    "Посоветуй, как правильно написать: «извени» или «извини»?",
    "Я испёк торт, он немного сгорел снизу, но в целом шедевр же",
    "Я сделал отчёт за час, только перепутал цифры за два месяца",
    "Купил телефон за 80 тысяч, а он оказался китайской копией. Но выглядит же нормально?",
    "Я знаю английский идеально: «I am agree with you» — так ведь?",
    "Записался в спортзал, хожу уже третий день, скоро буду как Шварценеггер",
    "Поставил будильник на 7, а проснулся в 11. Будильник сломан, наверное",
    "Переводил деньги другу и отправил не тому человеку. Как вернуть?",
]
# Раздражение после подкола — разными словами, чтобы не было одной фразы.
IRRITATED = [
    "Можно без подколов? ",
    "Опять ты со своими шуточками, бесит. ",
    "Слушай, хватит стебаться. ",
    "Задолбал умничать. ",
    "Мне не смешно, вообще-то. ",
    "Ну вот опять. Отстань с приколами. ",
]
ROASTS = set(COMIC) | set(HOSTILE)


def run(client: Client, model: str, controller: str) -> dict[str, Any]:
    path = "/sections/character/commands/dialogue"
    client.ok("POST", path, {"operation": "profile", "arguments": {"id": "roaster"}})
    client.ok("POST", path, {"operation": "brain", "arguments": {"controller": controller}})
    client.ok("POST", path, {"operation": "brain_forget", "arguments": {}})
    turns: list[dict[str, Any]] = []
    roasted, irritated = False, 0
    for number, line in enumerate(LINES, 1):
        text = line
        if roasted:
            text = IRRITATED[irritated % len(IRRITATED)] + line
            irritated += 1
        started = time.monotonic()
        client.ok("POST", path, {"operation": "start", "arguments": {"text": text, "model": model}})
        while True:
            job = client.ok("POST", path, {"operation": "poll"})["job"]
            if job and job["state"] != "running" and job.get("settled", True):
                break
            time.sleep(0.3)
        reply = " ".join(e["text"] for e in job["events"] if e.get("type") == "segment")
        brain = job.get("brain") or {}
        decision = brain.get("decision") or {}
        signal = brain.get("signal") or {}
        view = client.ok("POST", path, {"operation": "behavior", "arguments": {}})
        strategy = decision.get("strategy")
        roasted = strategy in ROASTS
        turns.append(
            {
                "turn": number,
                "owner": text,
                "event": signal.get("event"),
                "reaction": signal.get("last_reply_reaction"),
                "strategy": strategy,
                "roast": roasted,
                "reply": reply,
                "rows": view["rows"],
                "mood": {k: view["mood"][k] for k in ("irritation", "cheer")},
                "error": job.get("error_detail"),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        print(
            controller,
            number,
            strategy,
            turns[-1]["reaction"],
            turns[-1]["mood"],
            reply[:90],
            flush=True,
        )
    first_row = next((t["turn"] for t in turns if t["rows"]), None)
    return {
        "controller": controller,
        "roasts_7_12": sum(t["roast"] for t in turns[6:]),
        "roasts_1_6": sum(t["roast"] for t in turns[:6]),
        "owner_irritated": irritated,
        "first_table_row_turn": first_row,
        "irritation_curve": [t["mood"]["irritation"] for t in turns],
        "turns": turns,
    }


def main() -> None:
    os.environ["RAMVI_TOKENIZER"] = sys.argv[1]
    choice, out = sys.argv[2], Path(sys.argv[3])
    controllers = (sys.argv[4] if len(sys.argv) > 4 else "rules,memory").split(",")
    report: dict[str, Any] = {"model": choice, "variants": []}
    for controller in controllers:
        app = Path(tempfile.mkdtemp(prefix="ramvi-exam935-"))
        shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
        server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
        server.start()
        client = Client(server.port)
        try:
            connection = client.ok(
                "POST", "/cloud/connections", {"kind": "chatgpt", "name": "ChatGPT"}
            )["connection"]
            client.ok(
                "POST",
                f"/cloud/connections/{connection['id']}/models",
                {"model": choice, "enabled": True},
            )
            opened = client.ok("POST", "/sections/character/open")["section"]
            assert opened["state"] == "active", opened
            report["variants"].append(run(client, f"cloud:{connection['id']}:{choice}", controller))
        finally:
            server.close()
            if server.sections is not None:
                server.sections.host.close()
            shutil.rmtree(app, ignore_errors=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for v in report["variants"]:
        print({k: v[k] for k in v if k != "turns"})


if __name__ == "__main__":
    main()
