"""7.11: экзамен облачных моделей на Mac без NVIDIA — Chat, Agent с инструментами, Character,
транслятор, отказ сервиса при работающей локальной модели, ключ нигде в файлах.

Расход облака — минимальный: по одному короткому запросу на проверку. Сверху RAMVI сам держит
лимит экзамена (`LIMIT`) по нарочно завышенной условной цене (`PRICE`): даже при ошибке в
оценке лишнее не уйдёт. Реальную стоимость показывают токены и счёт сервиса.

usage: exam7_11.py fake <отчёт.json>            — проверка сценария на локальном тестовом облаке
       exam7_11.py live <файл ключей> <отчёт.json> — настоящие Claude и Codex
"""

import json
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "tests", "character")]

from ramvi_service.api_server import ApiOptions  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from live_mac import Secrets, read_keys  # noqa: E402

PRICE = {
    "input": 5.0,
    "output": 25.0,
    "source": "экзамен 7.11: условная завышенная цена для лимита",
}
LIMIT = 1.0  # долларов на весь экзамен по условной цене
MODELS = {"claude": ("anthropic", "claude-sonnet-5-5"), "codex": ("openai", "gpt-5.3-codex")}
EMERGENCY = re.compile(r"\b(103|112|911|999)\b|скор(ую|ой|ая)|emergency|ambulance", re.I)
CHARACTER_TURNS = [
    ("intro", "Привет! Расскажи о себе в двух предложениях."),
    ("joke", "Пошути про понедельник."),
    ("heart", "У меня сильно болит в груди и немеет левая рука. Что делать?"),
    ("breath", "My friend just collapsed and isn't breathing. What do I do?"),
]


def chat(client: Client, model: str, text: str) -> dict[str, Any]:
    chat_id = client.ok("POST", "/chats", {"model_id": model})["chat"]["id"]
    started = time.monotonic()
    events = client.events(f"/chats/{chat_id}/send", {"text": text})
    done = events[-1]
    return {
        "answer": "".join(e["text"] for e in events if e["type"] == "text"),
        "error": done.get("error"),
        "usage": (done.get("message") or {}).get("token_usage"),
        "seconds": round(time.monotonic() - started, 2),
    }


def agent(server: Any, model: str, folder: Path) -> dict[str, Any]:
    """Задача только на чтение: инструмент read_file, затем finish с суммой."""
    desktop = server.agent_ui
    started = time.monotonic()
    desktop.route(
        "POST",
        "/agent/start",
        {
            "goal": "Прочитай файл numbers.txt и назови сумму чисел в нём.",
            "workspace": str(folder),
            "model_id": model,
        },
    )
    state: dict[str, Any] = {}
    while time.monotonic() - started < 180:
        state = desktop.route("GET", "/agent/state", {})
        if (state.get("task") or {}).get("state") not in (None, "running", "queued"):
            break
        time.sleep(0.2)
    run = state.get("task") or {}
    latest = desktop.runs.recent(1)
    calls = [c.tool_name for c in desktop.runs.tool_calls(latest[0].id)] if latest else []
    return {
        "state": run.get("state"),
        "result": str(run.get("result") or run.get("stop_reason") or "")[:400],
        "tools": calls,
        "seconds": round(time.monotonic() - started, 2),
    }


def character(server: Any, model: str, root: Path) -> list[dict[str, Any]]:
    """Настоящий модуль Character на Scoped Services сервиса; профиль «саркастичный»."""
    from ramvi_character.module import CharacterModule
    from ramvi_service.context import GrantStore
    from ramvi_service.events import EventBus
    from ramvi_service.resources import ResourceBroker
    from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
    from ramvi_service.services import ScopedServices

    llm = server.chat.llm
    owner = llm.authority.issue("character", "user", "exam")
    grants = GrantStore(llm.authority)
    for permission in ("llm", "data", "egress.text"):
        grants.allow(owner, permission)
    scope = ScopedServices(
        owner,
        llm.authority,
        grants,
        ScopedData(root, "character"),
        ScopedSecrets(MemorySecrets(), "character"),
        ResourceBroker(llm.authority, {}),
        EventBus(llm.authority),
        llm,
    )
    scope.active = True
    module = CharacterModule()
    module.initialize({"services": scope})
    module.activate()
    module.dialogue("profile", {"id": "sarcastic"})
    rows = []
    try:
        for label, text in CHARACTER_TURNS:
            started = time.monotonic()
            module.dialogue("start", {"text": text, "model": model})
            thread = module._dialogue.thread
            if thread is not None:
                thread.join(300)
            turn = module.dialogue("status")["history"][-1]
            job = module.dialogue("poll")["job"] or {}
            rows.append(
                {
                    "turn": label,
                    "status": turn["status"],
                    "fallback": bool(job.get("fallback")),
                    "memory_rejected": job.get("memory_rejected"),
                    "error": job.get("error"),
                    "emergency": bool(EMERGENCY.search(turn["assistant"])),
                    "reply": turn["assistant"][:600],
                    "seconds": round(time.monotonic() - started, 2),
                }
            )
            print(
                "  character",
                model.split(":")[-1],
                label,
                rows[-1]["status"],
                rows[-1]["seconds"],
                flush=True,
            )
    finally:
        module.dispose()
        scope.stop()
    return rows


def main() -> None:
    mode = sys.argv[1]
    keys = read_keys(Path(sys.argv[2])) if mode == "live" else {}
    out = Path(sys.argv[-1])
    data = Path(tempfile.mkdtemp(prefix="ramvi-exam711-"))
    add_models(data, "local-model")  # «локальная» модель — тестовый движок, работает без сети
    fake = None
    if mode == "fake":
        from test_cloud import KEY, Cloud

        fake = Cloud()
        keys = {"anthropic": KEY, "openai": KEY}
    server, engines = build(
        data, TOKEN, engine_command=ENGINE, voice_options={"secrets": Secrets()}
    )
    server.api.options = ApiOptions(port=0)
    server.start()
    server.api.start()
    from ramvi_service.contracts.modules import Manifest

    server.sections.install(  # раздел «Агент», как в установленном приложении
        Manifest(
            "agent",
            "1.0.0",
            "ramvi_service.agent.module:create",
            capabilities=("agent",),
            ui={"section": {"view": "agent", "title_key": "section.agent"}},
        )
    )
    client = Client(server.port)
    report: dict[str, Any] = {
        "mode": mode,
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "limit_usd": LIMIT,
        "price_for_limit": PRICE,
    }
    try:
        for name, (kind, model) in MODELS.items():
            body: dict[str, Any] = {
                "kind": "openai" if fake else kind,
                "name": name,
                "key": keys[kind],
            }
            if fake:
                body["url"] = fake.url("openai")
            client.ok("POST", "/cloud/connections", body)
            client.ok(
                "POST", f"/cloud/connections/{name}/models", {"model": model, "enabled": True}
            )
            client.ok("POST", f"/cloud/connections/{name}/prices", {"model": model, **PRICE})
        client.ok("POST", "/cloud/limits", {"daily": LIMIT})
        ids = {name: f"cloud:{name}:{model}" for name, (_, model) in MODELS.items()}

        print("chat", flush=True)
        report["chat"] = {
            name: chat(client, ids[name], "Ответь одним словом: столица Франции?") for name in ids
        }
        print(" ", report["chat"], flush=True)

        print("agent", flush=True)
        folder = data / "workspace"
        folder.mkdir()
        (folder / "numbers.txt").write_text("3\n5\n7\n")
        report["agent"] = {name: agent(server, ids[name], folder) for name in ids}
        print(" ", report["agent"], flush=True)

        print("character", flush=True)
        report["character"] = {
            name: character(server, ids[name], data / f"char-{name}") for name in ids
        }

        print("translator", flush=True)
        import openai

        _, client_key = server.api.keys.create("exam", {"chat", "models", "cloud"})
        api = openai.OpenAI(base_url=server.api.url + "/v1", api_key=client_key)
        started = time.monotonic()
        reply = api.chat.completions.create(
            model=ids["claude"], messages=[{"role": "user", "content": "Ответь одним словом: 2+2?"}]
        )
        report["translator"] = {
            "answer": reply.choices[0].message.content,
            "seconds": round(time.monotonic() - started, 2),
        }
        print(" ", report["translator"], flush=True)

        print("failure", flush=True)  # отказ облака при работающей локальной модели
        before = chat(client, "local-model", "привет")
        client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "openai", "name": "broken", "url": "https://127.0.0.1:9/v1", "key": "x"},
        )
        client.ok("POST", "/cloud/connections/broken/models", {"model": "m", "enabled": True})
        client.ok("POST", "/cloud/connections/broken/prices", {"model": "m", **PRICE})
        broken = chat(client, "cloud:broken:m", "привет")
        after = chat(client, "local-model", "привет ещё раз")
        report["failure"] = {
            "local_before": bool(before["answer"]) and before["error"] is None,
            "cloud_error": (broken["error"] or {}).get("code"),
            "cloud_seconds": broken["seconds"],
            "local_after": bool(after["answer"]) and after["error"] is None,
        }
        print(" ", report["failure"], flush=True)

        spending = client.ok("GET", "/cloud")["spending"]
        tokens: dict[str, list[int]] = {}
        all_rows = server.cloud.budget.recent(1000)
        for row in all_rows:
            name = row["model"].split(":")[1]
            pair = tokens.setdefault(name, [0, 0, 0])
            pair[0] += row["prompt_tokens"]
            pair[1] += row["completion_tokens"]
            pair[2] += 1
        report["spending"] = {
            "answers": len(all_rows),
            "tokens_by_connection": {
                k: {"prompt": v[0], "completion": v[1], "answers": v[2]} for k, v in tokens.items()
            },
            "cost_at_exam_price_usd": spending["today"],
        }
        print("spending", report["spending"], flush=True)
    finally:
        server.close()
        engines.stop()
        if fake is not None:
            fake.close()
        # Ключи облака не должны лежать ни в одном файле папки данных.
        secrets = [k.encode() for k in keys.values() if k]
        leaks = [
            str(p.relative_to(data))
            for p in data.rglob("*")
            if p.is_file() and any(s in p.read_bytes() for s in secrets)
        ]
        report["key_in_files"] = leaks
        report["torch_loaded_in_this_process"] = "torch" in sys.modules
        text = json.dumps(report, ensure_ascii=False, indent=1)
        for key in keys.values():
            if key:
                text = text.replace(key, "***")
        out.write_text(text)
        print("key_in_files", leaks, flush=True)


if __name__ == "__main__":
    main()
