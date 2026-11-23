"""6.14 и 6.15 на ноутбуке с RTX 5060: настоящая локальная модель через сервис комплекта.

1. «Авто мод» (6.15): подтверждения в настройках включены, auto_mode включён — агент чинит
   ошибку и запускает тест без единого вопроса пользователю.
2. Просьба включить инструмент (6.14): run_command выключен — агент просит его включить;
   скрипт как пользователь разрешает «на эту задачу».
Настройки агента возвращаются к прежним.

usage: agent_win_live.py <папка комплекта> <подстрока модели> <отчёт.json>
"""

import json
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

APP = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(APP / "ramvi" / p) for p in ("service", "storage", "engine", "character")]

from ramvi_service.app import build  # noqa: E402

TOKEN = secrets.token_urlsafe(24)
# Системного Python на ноутбуке нет — тесты запускаются встроенным Python комплекта.
GOAL = (
    "В calc.py функция add работает неправильно. Исправь её и запусти тесты командой "
    f'`"{sys.executable}" -m unittest`, потом скажи, проходят ли они.'
)


def project(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (folder / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n\nclass T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
        encoding="utf-8",
    )


def run(desktop: Any, model: str, folder: Path) -> dict[str, Any]:
    project(folder)
    started = time.monotonic()
    desktop.route(
        "POST", "/agent/start", {"goal": GOAL, "workspace": str(folder), "model_id": model}
    )
    asked: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    while time.monotonic() - started < 900:
        state = desktop.route("GET", "/agent/state", {})
        pending = state.get("pending")
        if pending:
            asked.append({k: pending.get(k) for k in ("kind", "title", "tool", "why", "command")})
            desktop.route("POST", "/agent/decision", {"id": pending["id"], "action": "approve"})
        elif (state.get("task") or {}).get("state") not in (None, "running", "queued"):
            break
        time.sleep(0.5)
    task = state.get("task") or {}
    latest = desktop.runs.recent(1)
    calls = [c.tool_name for c in desktop.runs.tool_calls(latest[0].id)] if latest else []
    return {
        "state": task.get("state"),
        "result": str((task.get("result") or {}).get("result") or task.get("stop_reason") or "")[
            :600
        ],
        "asked": asked,
        "tools": calls,
        "fixed": (folder / "calc.py").read_text(encoding="utf-8"),
        "seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    out = Path(sys.argv[3])
    server, engines = build(APP / "data", TOKEN, app_dir=APP)
    server.start()
    desktop = server.agent_ui
    assert desktop is not None
    original = desktop.route("GET", "/agent/settings", {})["settings"]
    report: dict[str, Any] = {}
    work = Path(tempfile.mkdtemp(prefix="ramvi-agent-"))
    try:
        models = [m.id for m in server.chat.models_repo.all()]
        model = next(m for m in models if sys.argv[2] in m)
        report["model"] = model
        auto = json.loads(json.dumps(original))
        auto["permissions"].update({"write": True, "terminal": True})
        auto["access"].update(
            {"disabled": [], "confirm_writes": True, "confirm_commands": True, "auto_mode": True}
        )
        desktop.route("POST", "/agent/settings", auto)
        report["auto_mode"] = run(desktop, model, work / "auto")
        ask = json.loads(json.dumps(auto))
        ask["access"].update({"disabled": ["run_command"], "auto_mode": False})
        desktop.route("POST", "/agent/settings", ask)
        report["tool_request"] = run(desktop, model, work / "ask")
    finally:
        desktop.route("POST", "/agent/settings", original)
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        engines.stop()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
