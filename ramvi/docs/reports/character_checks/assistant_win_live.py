"""9.28: помощник в приложениях на ноутбуке с RTX 5060 — настоящий Блокнот и Windows UI Automation.

Запускать в сессии пользователя (окна видны). В Блокноте текст с опечаткой, выделено слово.
1. Окно выбрано, чтение разрешено — персонаж видит выделенное и отвечает про него.
2. Чтение не разрешено — персонаж говорит, что доступа нет (не гадает).
3. Блокнот закрыт — окно недоступно, персонаж об этом говорит.
4. Поручение агенту действовать в окне — агент исправляет слово (подтверждения даёт скрипт).

usage: assistant_win_live.py <папка комплекта> <подстрока модели> <отчёт.json> [файл ключей]
С файлом ключей модель берётся из облака OpenAI (подстрока, например gpt-5.5); ключ — только в
памяти, подключение удаляется после проверки.
"""

import json
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

APP = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(APP / "ramvi" / p) for p in ("service", "storage", "engine", "character")]

from ramvi_service.app import build  # noqa: E402

TOKEN = secrets.token_urlsafe(24)
TEXT = "Превет мир, как дила?"
QUESTION = "Вот здесь ошибка? Что не так?"


def powershell(script: str) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False, timeout=30)


def notepad_running() -> bool:
    # Вывод tasklist — в кодировке консоли Windows: сравнивать байты.
    found = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq notepad.exe", "/NH"], capture_output=True
    )
    return b"notepad.exe" in found.stdout.lower()


BEFORE = notepad_running()  # Блокнот пользователя, открытый до теста, не трогаем
# Блокнот Windows 11 запоминает вкладки (и несохранённые правки) — убрать только свои.
FORGET_TABS = (
    "$d=Join-Path $env:LOCALAPPDATA 'Packages\\Microsoft.WindowsNotepad_8wekyb3d8bbwe\\LocalState\\TabState'; "
    "if (Test-Path $d) { Get-ChildItem $d -File -Filter '*.bin' | Where-Object { $_.Name -notmatch '\\.\\d\\.bin$' } | "
    "ForEach-Object { $b=[IO.File]::ReadAllBytes($_.FullName); "
    "$t=[Text.Encoding]::Unicode.GetString($b)+[Text.Encoding]::UTF8.GetString($b); "
    "if ($b.Length -gt 1) { $t+=[Text.Encoding]::Unicode.GetString($b,1,$b.Length-1) }; "
    "if ($t -match 'ramvi-assistant-') { Get-ChildItem $d -File -Filter ($_.BaseName+'*') | Remove-Item -Force } } }"
)


def open_notepad(folder: Path) -> subprocess.Popen[bytes]:
    note = folder / "zametki.txt"
    note.write_text(TEXT, encoding="utf-8")
    process = subprocess.Popen(["notepad.exe", str(note)])
    time.sleep(3)
    # Вывести Блокнот вперёд и выделить первое слово (Ctrl+Home, Ctrl+Shift+Right).
    powershell(
        "$w=New-Object -ComObject wscript.shell; [void]$w.AppActivate('zametki'); "
        "Start-Sleep -Milliseconds 500; $w.SendKeys('^{HOME}'); $w.SendKeys('^+{RIGHT}')"
    )
    time.sleep(1)
    return process


def ask(server: Any, path: str, model: str, text: str) -> dict[str, Any]:
    route = server.sections.command
    route(
        "character", "dialogue", {"operation": "start", "arguments": {"text": text, "model": model}}
    )
    started = time.monotonic()
    while time.monotonic() - started < 600:
        job = route("character", "dialogue", {"operation": "poll"})["job"]
        if job and job["state"] != "running" and job.get("settled", True):
            break
        time.sleep(0.5)
    return {
        "state": job["state"],
        "answer": " ".join(e["text"] for e in job["events"] if e.get("type") == "segment"),
        "seconds": round(time.monotonic() - started, 1),
        "error": job.get("error_detail"),
    }


CLOUD = len(sys.argv) > 4  # облачной модели прочитанное отправлять можно (разрешение владельца)
CONNECTIONS: list[str] = []  # облачные подключения проверки — удаляются вместе с ключом


def cloud_model(server: Any, keys: Path, hint: str) -> str:
    lines = [line.strip() for line in keys.read_text("utf-8").splitlines()]
    key = next(line for line in lines[lines.index("Codex API key") + 1 :] if line)
    cloud = server.cloud
    connection = cloud.save({"kind": "openai", "name": "OpenAI", "key": key})
    CONNECTIONS.append(connection["id"])
    names = cloud.remote_models({"id": connection["id"]})
    model = hint if hint in names else min((n for n in names if hint in n), key=len)
    cloud.enable(connection["id"], model, True)
    return f"cloud:{connection['id']}:{model}"


def main() -> None:
    out = Path(sys.argv[3])
    server, engines = build(APP / "data", TOKEN, app_dir=APP)
    server.start()
    desktop, sections = server.agent_ui, server.sections
    assert desktop is not None and sections is not None
    original = desktop.route("GET", "/agent/settings", {})["settings"]
    work = Path(tempfile.mkdtemp(prefix="ramvi-assistant-"))
    report: dict[str, Any] = {}
    notepad = None
    try:
        if len(sys.argv) > 4:
            model = cloud_model(server, Path(sys.argv[4]), sys.argv[2])
        else:
            model = next(m.id for m in server.chat.models_repo.all() if sys.argv[2] in m.id)
        report["model"] = model
        sections.open("character")
        notes, project = work / "notes", work / "project"  # файл Блокнота — не в папке агента
        notes.mkdir()
        project.mkdir()
        notepad = open_notepad(notes)
        target = desktop.route("POST", "/agent/target", {})["target"]
        report["target"] = {k: target[k] for k in ("application", "title")}
        desktop.route(
            "POST",
            "/agent/target/module",
            {"module": "character", "ui": True, "screen": False, "egress": CLOUD, "control": True},
        )
        command = sections.command
        # История персонажа на тестовом комплекте копится от прогона к прогону — начать с чистой.
        command("character", "dialogue", {"operation": "history_clear", "arguments": {}})
        command(
            "character", "dialogue", {"operation": "assistant", "arguments": {"target": target}}
        )
        seen = command("character", "dialogue", {"operation": "assistant_view", "arguments": {}})
        context = seen.get("context") or {}
        report["0_what_character_sees"] = {
            "status": context.get("status"),
            "selected_text": context.get("selected_text"),
            "focused_text": context.get("focused_text"),
            "elements": len(context.get("elements") or []),
        }
        report["1_reads_selection"] = ask(server, "", model, QUESTION)

        # 9.29: другое окно на переднем плане — персонаж читает выбранное, а не активное.
        # Блокнот Windows 11 открыл бы второй файл вкладкой того же окна — берём окно другой
        # программы: отдельную консоль с заголовком drugoe.
        console = subprocess.Popen(
            ["cmd.exe", "/k", "title drugoe"],
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        time.sleep(2)
        powershell("$w=New-Object -ComObject wscript.shell; [void]$w.AppActivate('drugoe')")
        time.sleep(1)
        started = time.monotonic()
        seen = command("character", "dialogue", {"operation": "assistant_view", "arguments": {}})
        context = seen.get("context") or {}
        report["5_foreground_is_another_window"] = {
            "status": context.get("status"),
            "selected_text": context.get("selected_text"),
            "focused_text": context.get("focused_text"),
            "capture_seconds": round(time.monotonic() - started, 2),
        }
        console.kill()
        # Действия агента — только в активном окне: вернуть Блокнот вперёд, как сделал бы
        # пользователь за 5 секунд после подтверждения.
        powershell("$w=New-Object -ComObject wscript.shell; [void]$w.AppActivate('zametki')")
        time.sleep(1)

        desktop.route(
            "POST",
            "/agent/target/module",
            {
                "module": "character",
                "ui": False,
                "screen": False,
                "egress": False,
                "control": False,
            },
        )
        report["2_no_access"] = ask(server, "", model, QUESTION)

        desktop.route(
            "POST",
            "/agent/target/module",
            {"module": "character", "ui": True, "screen": False, "egress": CLOUD, "control": True},
        )
        # 4. Поручение агенту действовать в окне (до закрытия Блокнота).
        policy = json.loads(json.dumps(original))
        policy["permissions"].update({"write": True})
        desktop.route("POST", "/agent/settings", policy)
        # Право действовать в окне — у персонажа (задача агента идёт от его имени), выдано выше.
        binding = desktop.route(
            "POST",
            "/agent/delegate",
            {
                "module": "character",
                "enabled": True,
                "workspace": str(project),
                "model_id": model,
                "use_ui": True,
            },
        )["binding_id"]
        command(
            "character",
            "dialogue",
            {"operation": "agent_bind", "arguments": {"binding_id": binding}},
        )
        command(
            "character",
            "dialogue",
            {
                "operation": "delegate",
                "arguments": {"goal": "Исправь опечатку в выделенном слове в Блокноте."},
            },
        )
        asked: list[dict[str, Any]] = []
        started = time.monotonic()
        view: dict[str, Any] = {}
        while time.monotonic() - started < 600:
            state = desktop.route("GET", "/agent/state", {})
            pending = state.get("pending")
            if pending:
                asked.append({k: pending.get(k) for k in ("kind", "title", "value", "plan")})
                desktop.route("POST", "/agent/decision", {"id": pending["id"], "action": "approve"})
            view = command("character", "dialogue", {"operation": "delegation"})["delegation"]
            if view.get("task") and view["task"]["state"] != "running":
                break
            time.sleep(0.5)
        task = view.get("task") or {}
        after = command("character", "dialogue", {"operation": "assistant_view", "arguments": {}})
        report["4_window_text_after_agent"] = (after.get("context") or {}).get("focused_text")
        report["4_agent_in_window"] = {
            "state": task.get("state"),
            "result": task.get("result"),
            "asked": asked,
            "seconds": round(time.monotonic() - started, 1),
            "file_untouched": (notes / "zametki.txt").read_text(encoding="utf-8") == TEXT,
        }

        # Окно закрыто — цель устарела. Блокнот Windows 11 — один процесс с вкладками: закрыть
        # именно это окно (по заголовку), а не процесс-запускатель.
        # Правка агента не сохраняется: если до теста Блокнота не было, закрыть его без вопроса
        # «Сохранить?»; иначе — только своё окно.
        if not BEFORE:
            powershell("Stop-Process -Name notepad -Force -ErrorAction SilentlyContinue")
            time.sleep(1)
            powershell(FORGET_TABS)
        else:
            powershell(
                "Get-Process notepad -ErrorAction SilentlyContinue | "
                "Where-Object { $_.MainWindowTitle -like 'zametki*' } | "
                "ForEach-Object { $_.CloseMainWindow() | Out-Null }"
            )
        time.sleep(2)
        notepad.kill()
        notepad = None
        time.sleep(1)
        seen = command("character", "dialogue", {"operation": "assistant_view", "arguments": {}})
        report["3_sees_after_close"] = (seen.get("context") or {}).get("status")
        report["3_window_closed"] = ask(server, "", model, QUESTION)

        # 9.29: окно пропало — заново открыть, выбрать и разрешить: помощник снова видит его.
        notepad = open_notepad(notes)
        target = desktop.route("POST", "/agent/target", {})["target"]
        desktop.route(
            "POST",
            "/agent/target/module",
            {"module": "character", "ui": True, "screen": False, "egress": CLOUD, "control": False},
        )
        command(
            "character", "dialogue", {"operation": "assistant", "arguments": {"target": target}}
        )
        seen = command("character", "dialogue", {"operation": "assistant_view", "arguments": {}})
        report["6_reconnected"] = {
            "title": target.get("title"),
            "status": (seen.get("context") or {}).get("status"),
            "selected_text": (seen.get("context") or {}).get("selected_text"),
        }
    finally:
        if notepad is not None:
            notepad.kill()
            if not BEFORE:
                powershell("Stop-Process -Name notepad -Force -ErrorAction SilentlyContinue")
                time.sleep(1)
                powershell(FORGET_TABS)
        desktop.route("POST", "/agent/settings", original)
        for connection_id in CONNECTIONS:
            server.cloud.delete(connection_id)  # ключ уходит из системного хранилища
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        engines.stop()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
