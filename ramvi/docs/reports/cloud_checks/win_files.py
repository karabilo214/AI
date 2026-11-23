"""B.15: прикреплённые файлы на ноутбуке с RTX 5060 — настоящая локальная модель (qwen3 8B).

Запуск встроенным Python комплекта; сервис собирается как в приложении (папка комплекта).
1. Лимит «1 файл» из настроек: два файла в чате отвергаются до модели.
2. «Без ограничения»: файл больше контекста модели — «нужно N токенов, есть M» (не «ошибка ответа»).
3. Персонаж читает файл и называет число из него.
Настройки лимитов возвращаются к прежним значениям.

usage: win_files.py <папка комплекта> <отчёт.json>
"""

import http.client
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any

APP = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(APP / "ramvi" / p) for p in ("service", "storage", "engine", "character")]

from ramvi_service.app import build  # noqa: E402

TOKEN = secrets.token_urlsafe(24)
SECRET = "700021"
FILE = {"name": "sklad.txt", "text": f"Склад №3. Код двери: {SECRET}. Открыт с 8 до 17."}


class Client:
    def __init__(self, port: int) -> None:
        self.port = port

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        raw = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
        conn.request(method, "/ui/v1" + path, raw, headers)
        response = conn.getresponse()
        value = json.loads(response.read() or b"{}")
        conn.close()
        return response.status, value

    def ok(self, method: str, path: str, body: Any = None) -> Any:
        status, value = self.request(method, path, body)
        if status != 200:
            raise AssertionError((path, status, value))
        return value

    def stream(self, path: str, body: Any) -> list[dict[str, Any]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
        conn.request("POST", "/ui/v1" + path, json.dumps(body).encode(), headers)
        response = conn.getresponse()
        if response.status != 200:
            return [
                {"type": "http", "status": response.status, "body": json.loads(response.read())}
            ]
        events = [json.loads(line[6:]) for line in response if line.startswith(b"data: ")]
        conn.close()
        return events


def main() -> None:
    out = Path(sys.argv[2])
    server, engines = build(APP / "data", TOKEN, app_dir=APP)
    server.start()
    client = Client(server.port)
    report: dict[str, Any] = {}
    saved = client.ok("GET", "/settings")["settings"]
    names = ("attach_max_files", "attach_max_file_kb", "attach_max_total_kb")
    try:
        ids = [m["id"] for m in client.ok("GET", "/models")["models"]]
        model = next(i for i in ids if "qwen3-8b" in i and "q4_k_m" in i)  # быстрый квант
        report["model"] = model
        chat = client.ok("POST", "/chats", {"model_id": model})["chat"]

        client.ok("POST", "/settings", {"attach_max_files": 1})
        status, body = client.request(
            "POST", f"/chats/{chat['id']}/send", {"text": "?", "attachments": [FILE, FILE]}
        )
        report["chat_two_files_with_limit_1"] = {"status": status, "error": body.get("error")}

        client.ok("POST", "/settings", {name: 0 for name in names})
        started = time.monotonic()
        client.ok("POST", "/engine/load", {"model_id": model})
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            engine = client.ok("GET", "/status")["engine"]
            if engine.get("problem") or (
                engine.get("model_id") == model and not engine.get("loading")
            ):
                break
            time.sleep(1)
        report["engine"] = {**engine, "load_s": round(time.monotonic() - started, 1)}

        big = {"name": "big.txt", "text": "строка текста для проверки контекста " * 12000}
        events = client.stream(
            f"/chats/{chat['id']}/send", {"text": "Что в файле?", "attachments": [big]}
        )
        done = next((e for e in events if e.get("type") in ("done", "http")), {})
        report["chat_file_bigger_than_context"] = {
            "file_bytes": len(big["text"].encode()),
            "done": done.get("error") or done,
        }

        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        started = time.monotonic()
        client.ok(
            "POST",
            path,
            {
                "operation": "start",
                "arguments": {
                    "text": "Посмотри файл и скажи, какой там код двери.",
                    "model": model,
                    "files": [FILE],
                },
            },
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            job = client.ok("POST", path, {"operation": "poll"})["job"]
            if job and job["state"] != "running" and job.get("settled", True):
                break
            time.sleep(0.3)
        answer = " ".join(e["text"] for e in job["events"] if e.get("type") == "segment")
        report["character_with_file"] = {
            "state": job["state"],
            "error": job.get("error"),
            "answer": answer,
            "names_the_code": SECRET in answer,
            "seconds": round(time.monotonic() - started, 1),
            "engine_after": client.ok("GET", "/status")["engine"],
            "job": {k: v for k, v in job.items() if k != "events"},
        }
    finally:
        client.request("POST", "/settings", {name: saved.get(name, 0) for name in names})
        client.request("POST", "/engine/unload", {})
        server.close()
        if server.sections is not None:
            server.sections.host.close()  # процессы модулей (персонаж)
        engines.stop()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
