"""B.16 на ноутбуке с RTX 5060: размер запроса персонажа на локальной 8B (контекст не меньше 8192).

1. Файл ~2400 токенов — помещается, персонаж называет код, спрятанный в конце файла.
2. Файл ~4400 токенов (до B.16 потолок реплики был 3072) и файл ~52 000 токенов — больше, чем
   остаётся в контексте 8192 после системного промпта и места под ответ: понятная ошибка
   «нужно N, есть M», без обращения к модели.
3. Обычная реплика после этого — ответ есть.

usage: context_local_b16.py <папка комплекта> <подстрока модели> <отчёт.json>
"""

import http.client
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

APP = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(APP / "ramvi" / p) for p in ("service", "storage", "engine", "character")]
os.environ.setdefault("RAMVI_TOKENIZER", str(APP / "runtime" / "tokenizers" / "Qwen3-8B"))

from ramvi_service.app import build  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKEN = secrets.token_urlsafe(24)
SECRET = "48213"
LINES = [f"Строка {i}: на складе всё по-прежнему, полки в порядке." for i in range(1, 180)]
FITS = {"name": "sklad.txt", "text": "\n".join([*LINES[:99], f"Код двери склада: {SECRET}."])}
OVER = {"name": "sklad2.txt", "text": "\n".join([*LINES, f"Код двери склада: {SECRET}."])}
HUGE = {"name": "huge.txt", "text": "\n".join(LINES * 12)}
PATH = "/sections/character/commands/dialogue"


class Client:
    """UI API сервиса комплекта (как в tests/test_chat.py; тестов в комплекте нет)."""

    def __init__(self, port: int) -> None:
        self.port = port

    def ok(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        raw = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
        conn.request(method, "/ui/v1" + path, raw, headers)
        response = conn.getresponse()
        value = json.loads(response.read())
        conn.close()
        assert response.status == 200, value
        return value


def turn(desktop: Client, text: str, model: str, files: list[dict[str, str]]) -> dict[str, Any]:
    started = time.monotonic()
    desktop.ok(
        "POST",
        PATH,
        {"operation": "start", "arguments": {"text": text, "model": model, "files": files}},
    )
    job: dict[str, Any] = {}
    while time.monotonic() - started < 600:
        job = desktop.ok("POST", PATH, {"operation": "poll"})["job"] or {}
        if job and job.get("state") != "running" and job.get("settled", True):
            break
        time.sleep(0.5)
    answer = " ".join(e["text"] for e in job.get("events", []) if e.get("type") == "segment")
    return {
        "state": job.get("state"),
        "error": job.get("error"),
        "error_detail": job.get("error_detail"),
        "overflow": job.get("overflow"),
        "answer": answer[:400],
        "seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    out = Path(sys.argv[3])
    server, engines = build(APP / "data", TOKEN, app_dir=APP)
    server.start()
    desktop = Client(server.port)
    tokenizer = AutoTokenizer.from_pretrained(os.environ["RAMVI_TOKENIZER"], local_files_only=True)
    report: dict[str, Any] = {
        "file_tokens": len(tokenizer.encode(FITS["text"])),
        "over_tokens": len(tokenizer.encode(OVER["text"])),
        "huge_tokens": len(tokenizer.encode(HUGE["text"])),
    }
    try:
        model = next(m.id for m in server.chat.models_repo.all() if sys.argv[2] in m.id)
        report["model"] = model
        opened = desktop.ok("POST", "/sections/character/open")["section"]
        assert opened["state"] == "active", opened
        question = "Посмотри файл и скажи, какой там код двери склада."
        report["fits"] = turn(desktop, question, model, [FITS])
        report["fits"]["names_the_code"] = SECRET in report["fits"]["answer"]
        report["over"] = turn(desktop, question, model, [OVER])
        report["huge"] = turn(desktop, "Что в этом файле?", model, [HUGE])
        report["after"] = turn(desktop, "Привет! Как дела?", model, [])
    finally:
        server.close()
        if server.sections is not None:
            server.sections.host.close()
        engines.stop()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
