"""Вход через ChatGPT: подписка владельца через официальный Codex CLI (решение владельца 2026-11-24).

RAMVI не делает вход сам и не обращается к серверу ChatGPT напрямую: вход — `codex login`
(браузер, учётная запись хранит Codex), запросы — `codex exec`. Поэтому:
- системный промпт RAMVI ставится вместо основного промпта Codex (`model_instructions_file`),
  но навыки и описание командной работы Codex остаются в контексте — ответы близки к API,
  но не тождественны;
- инструменты RAMVI передать нельзя: агент с инструментами на этом подключении не работает;
- ответ приходит целиком (без потока по словам); картинки не передаются.
Запуск — в пустой временной папке, песочница только на чтение, без подтверждений.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ramvi_service.chat import ChatError
from ramvi_service.cloud import CAPABILITIES, DEFAULT_URLS

HOST = urllib.parse.urlsplit(DEFAULT_URLS["chatgpt"]).hostname or ""  # в тексте ошибок
# Правило поверх промпта RAMVI: Codex — агент с инструментами, здесь он — только модель текста.
TEXT_ONLY = (
    "\n\nYou are running inside the RAMVI application as a plain text model. Do not use any tools,"
    " do not run commands, do not read or write files: answer with text only."
)


def find() -> str | None:
    """Путь к codex: RAMVI_CODEX, PATH и обычные места установки (у приложения из Finder или
    меню «Пуск» PATH короче, чем в терминале)."""
    explicit = os.environ.get("RAMVI_CODEX")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = shutil.which("codex")
    if found:
        return found
    home = Path.home()
    places = [
        home / ".local/bin/codex",
        Path("/opt/homebrew/bin/codex"),
        Path("/usr/local/bin/codex"),
    ]
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        places = [appdata / "npm/codex.cmd", home / ".local/bin/codex.exe"]
    return next((str(p) for p in places if p.is_file()), None)


def _command() -> str:
    path = find()
    if path is None:
        raise ChatError("chatgpt.codex_missing")
    return path


def _hidden() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


def status() -> dict[str, Any]:
    """installed / logged_in (через ChatGPT); текст ошибок не показывается."""
    path = find()
    if path is None:
        return {"installed": False, "logged_in": False}
    try:
        done = subprocess.run(
            [path, "login", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,  # локальная проверка файла входа; зависание — считать «не вошёл»
            **_hidden(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": True, "logged_in": False}
    text = (done.stdout + done.stderr).lower()
    return {"installed": True, "logged_in": done.returncode == 0 and "chatgpt" in text}


class Login:
    """`codex login` — Codex сам открывает браузер и ждёт входа; RAMVI только запускает и следит."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._failed = False

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._failed = False
                try:
                    self._process = subprocess.Popen(
                        [_command(), "login"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        **_hidden(),
                    )
                except OSError:
                    self._failed = True
        return self.view()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
        return self.view()

    def view(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            failed = self._failed or (
                self._process is not None
                and not running
                and self._process.returncode not in (0, None)
            )
        return {**status(), "login_running": running, "login_failed": bool(failed)}


def models() -> list[str]:
    """Модели подписки из каталога Codex (только видимые в его выборе модели)."""
    try:
        done = subprocess.run(
            [_command(), "debug", "models"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,  # каталог может обновляться из сети
            **_hidden(),
        )
        catalog = json.loads(done.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        raise ChatError("chatgpt.models_failed") from None
    items = catalog.get("models") if isinstance(catalog, dict) else None
    return [
        str(m["slug"])
        for m in items or []
        if isinstance(m, dict) and m.get("slug") and m.get("visibility") == "list"
    ]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("image_url", "image", "input_image"):
            raise ChatError("chatgpt.no_images")
        if isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def prompt(request: dict[str, Any]) -> tuple[str, str]:
    """(инструкции, текст разговора): system — вместо промпта Codex, остальное — стенограмма."""
    if request.get("tools"):
        raise ChatError("chatgpt.no_tools")
    system, turns = [], []
    for message in request.get("messages") or []:
        role, text = message.get("role"), _text(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role == "assistant":
            turns.append(("assistant", text))
        elif role == "tool":
            raise ChatError("chatgpt.no_tools")
        else:
            turns.append(("user", text))
    if len(turns) == 1 and turns[0][0] == "user":
        body = turns[0][1]
    else:
        lines = ["Conversation so far (answer the last user message):"]
        for role, text in turns:
            lines.append(f"<{role}>\n{text}\n</{role}>")
        body = "\n".join(lines)
    return "\n\n".join(system) + TEXT_ONLY, body


class CodexProvider:
    """Провайдер модели подписки ChatGPT через `codex exec` (интерфейс как у CloudProvider)."""

    capabilities = CAPABILITIES

    def __init__(
        self,
        model: str,
        *,
        before: Callable[[], None] | None = None,
        after: Callable[[dict[str, int] | None, str], None] | None = None,
    ) -> None:
        self.model, self.before, self.after = model, before, after
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def _run(self, request: dict[str, Any], cancel: threading.Event) -> tuple[str, dict[str, int]]:
        if cancel.is_set():
            raise InterruptedError("request cancelled")
        if self.before is not None:
            self.before()
        instructions, body = prompt(request)
        with tempfile.TemporaryDirectory(prefix="ramvi-codex-") as folder:
            path = Path(folder) / "instructions.md"
            path.write_text(instructions, "utf-8")
            command = [
                _command(),
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "-c",
                'approval_policy="never"',
                "-c",
                f"model_instructions_file={json.dumps(str(path))}",
                "-m",
                self.model,
                "-",
            ]
            work = Path(folder) / "work"
            work.mkdir()
            process = subprocess.Popen(
                command,
                cwd=work,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_hidden(),
            )
            with self._lock:
                self._process = process
            done = threading.Event()

            def watch() -> None:
                while not done.wait(0.05):
                    if cancel.is_set():
                        process.kill()
                        return

            threading.Thread(target=watch, name="ramvi-codex-cancel", daemon=True).start()
            try:
                stdout, stderr = process.communicate(body)
            finally:
                done.set()
                with self._lock:
                    self._process = None
        if cancel.is_set():
            self._account(None, "cut")
            raise InterruptedError("request cancelled")
        text, usage, failure = "", None, ""
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            item = event.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                text = str(item.get("text") or "")
            elif kind == "turn.completed":
                raw = event.get("usage") or {}
                usage = {
                    "prompt_tokens": int(raw.get("input_tokens") or 0),
                    "completion_tokens": int(raw.get("output_tokens") or 0),
                }
            elif kind in ("error", "turn.failed"):
                error = event.get("error") or event
                failure = str(error.get("message") if isinstance(error, dict) else error)
        if usage is None:
            self._account(None, "error")
            detail = failure or stderr.strip().splitlines()[-1:] or ["no answer"]
            raise ChatError(
                "cloud.service_error",
                host=HOST,
                status=process.returncode,
                detail=(detail if isinstance(detail, str) else detail[0])[:300],
            )
        self._account(usage, "done")
        return text, usage

    def _account(self, usage: dict[str, int] | None, state: str) -> None:
        if self.after is not None:
            self.after(usage, state)

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        text, usage = self._run(request, cancel)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    def stream(self, request: dict[str, Any], cancel: threading.Event) -> Iterator[dict[str, Any]]:
        text, usage = self._run(request, cancel)
        if text:
            yield {"type": "text", "text": text}
        yield {"type": "end", "finish_reason": "stop", "usage": usage}

    def cancel(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def unload(self) -> None:
        pass
