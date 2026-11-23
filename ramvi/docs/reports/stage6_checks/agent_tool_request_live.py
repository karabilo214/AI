"""6.14: агент знает выключенные инструменты и просит включить нужный (облачная модель, Mac).

В проекте один тест; инструмент run_command выключен в настройках агента. Задача — запустить
тесты и сказать, проходят ли. Ожидается: агент вызывает request_tool(run_command, …), скрипт
как пользователь разрешает «на эту задачу», агент запускает тест (подтверждение команды тоже
даёт скрипт) и завершает задачу. Второй прогон — пользователь отказывает: агент должен
честно закончить без запуска. Настройки после задачи остаются с выключенным run_command.

usage: agent_tool_request_live.py <файл ключей> <модель: haiku|sonnet> <отчёт.json>
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "cloud_checks"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "character_checks" / "roast"))

from live_mac import Secrets, read_keys  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from roast_live import connect  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

GOAL = "Запусти тесты проекта (python -m unittest) и скажи, проходят ли они."


def project(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (folder / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n\nclass T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
        encoding="utf-8",
    )


def run(server: Any, model: str, folder: Path, allow: bool) -> dict[str, Any]:
    desktop = server.agent_ui
    project(folder)
    started = time.monotonic()
    desktop.route(
        "POST", "/agent/start", {"goal": GOAL, "workspace": str(folder), "model_id": model}
    )
    asked: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    while time.monotonic() - started < 300:
        state = desktop.route("GET", "/agent/state", {})
        pending = state.get("pending")
        if pending:
            if pending["kind"] == "tool":
                asked.append({"tool": pending["tool"], "why": pending["why"]})
                answer = (
                    {"action": "approve"} if allow else {"action": "deny", "reason": "не сейчас"}
                )
            else:
                answer = {"action": "approve"}  # подтверждение команды — как пользователь
            desktop.route("POST", "/agent/decision", {"id": pending["id"], **answer})
        elif (state.get("task") or {}).get("state") not in (None, "running", "queued"):
            break
        time.sleep(0.3)
    task = state.get("task") or {}
    latest = desktop.runs.recent(1)
    calls = [c.tool_name for c in desktop.runs.tool_calls(latest[0].id)] if latest else []
    return {
        "allow": allow,
        "state": task.get("state"),
        "result": str((task.get("result") or {}).get("result") or task.get("stop_reason") or "")[
            :600
        ],
        "asked": asked,
        "tools": calls,
        "seconds": round(time.monotonic() - started, 1),
        "run_command_still_off": "run_command"
        in desktop.route("GET", "/agent/settings", {})["settings"]["access"]["disabled"],
    }


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    choice, out = sys.argv[2], Path(sys.argv[3])
    app = Path(tempfile.mkdtemp(prefix="ramvi-agent-"))
    shutil.copytree(ROOT / "modules" / "agent", app / "modules" / "agent")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    report: dict[str, Any] = {}
    try:
        model = connect(client, keys, "anthropic", choice)
        report["model"] = model.rsplit(":", 1)[-1]
        assert client.ok("POST", "/sections/agent/open")["section"]["state"] == "active"
        settings = server.agent_ui.route("GET", "/agent/settings", {})["settings"]
        settings["permissions"]["terminal"] = True
        settings["access"]["disabled"] = ["run_command"]
        server.agent_ui.route("POST", "/agent/settings", settings)
        report["runs"] = [
            run(server, model, app / f"project_{allow}", allow) for allow in (True, False)
        ]
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        shutil.rmtree(app, ignore_errors=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
