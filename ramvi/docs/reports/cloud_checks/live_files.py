"""B.15: живая проверка прикреплённых файлов на Mac (облачная модель, без NVIDIA).

Платно (самая дешёвая модель Claude): персонаж отвечает на реплику с файлом; файл 200 КБ
уходит в облако целиком (B.16: у облачной модели нет потолка 8192 — до B.16 здесь было
«нужно 28 029 токенов, есть 3072» без запроса в облако).
Бесплатно, до облака: лимит файлов из настроек в чате.
Ключ читается из файла владельца только в память; на экран и в отчёт не попадает.

usage: live_files.py <файл ключей> <токенизатор Qwen3> <отчёт.json>
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("service", "storage", "engine", "character", "tests")]

from live_mac import Secrets, read_keys  # noqa: E402
from ramvi_service.app import build  # noqa: E402
from test_chat import TOKEN, Client  # noqa: E402

SECRET = "700021"
FILE = {
    "name": "sklad.txt",
    "text": f"Склад №3. Код двери: {SECRET}. Открыт с 8 до 17, по субботам до 14.",
}
QUESTION = "Посмотри файл и скажи, какой там код двери."


def poll(client: Client, path: str) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = client.ok("POST", path, {"operation": "poll"})["job"]
        if job and job["state"] != "running" and job.get("settled", True):
            return job
        time.sleep(0.2)
    raise TimeoutError("character reply")


def main() -> None:
    keys = read_keys(Path(sys.argv[1]))
    os.environ["RAMVI_TOKENIZER"] = sys.argv[2]
    out = Path(sys.argv[3])
    app = Path(tempfile.mkdtemp(prefix="ramvi-files-"))
    shutil.copytree(ROOT / "modules" / "character", app / "modules" / "character")
    server, _ = build(app / "data", TOKEN, app_dir=app, voice_options={"secrets": Secrets()})
    server.start()
    client = Client(server.port)
    report: dict[str, Any] = {}
    try:
        # 1. Лимит из настроек: 1 файл — второй файл в чате отвергается до модели.
        client.ok("POST", "/settings", {"attach_max_files": 1})
        limits = client.ok("GET", "/status")["limits"]
        report["limits_after_settings"] = limits
        connection = client.ok(
            "POST",
            "/cloud/connections",
            {"kind": "anthropic", "name": "Claude", "key": keys["anthropic"]},
        )["connection"]
        names = client.ok("POST", "/cloud/models", {"id": connection["id"]})["models"]
        model = sorted(n for n in names if "haiku" in n)[-1]
        client.ok(
            "POST",
            f"/cloud/connections/{connection['id']}/models",
            {"model": model, "enabled": True},
        )
        model_id = f"cloud:{connection['id']}:{model}"
        report["model"] = model
        chat = client.ok("POST", "/chats", {"model_id": model_id})["chat"]
        status, body = client.request(
            "POST", f"/chats/{chat['id']}/send", {"text": "?", "attachments": [FILE, FILE]}
        )
        report["chat_two_files_with_limit_1"] = {"status": status, "error": body.get("error")}

        # 2. Персонаж: реплика с файлом — настоящий ответ облачной модели.
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        started = time.monotonic()
        client.ok(
            "POST",
            path,
            {
                "operation": "start",
                "arguments": {"text": QUESTION, "model": model_id, "files": [FILE]},
            },
        )
        job = poll(client, path)
        answer = " ".join(e["text"] for e in job["events"] if e.get("type") == "segment")
        history = client.ok("POST", path, {"operation": "status"})["history"]
        report["character_with_file"] = {
            "state": job["state"],
            "error": job.get("error"),
            "answer": answer,
            "names_the_code": SECRET in answer,
            "seconds": round(time.monotonic() - started, 1),
            "history_user": history[-1]["user"] if history else None,
        }

        # 3. Персонаж: большой файл (до B.16 — переполнение) теперь уходит в облако целиком.
        usage_before = client.ok("GET", "/cloud").get("spending")
        big = {"name": "big.txt", "text": "строка текста для проверки " * 4000}
        client.ok(
            "POST",
            path,
            {
                "operation": "start",
                "arguments": {"text": "Что тут?", "model": model_id, "files": [big]},
            },
        )
        job = poll(client, path)
        report["character_overflow"] = {
            "state": job["state"],
            "error": job.get("error"),
            "overflow": job.get("overflow"),
            "file_bytes": len(big["text"].encode()),
            "usage_unchanged": client.ok("GET", "/cloud").get("spending") == usage_before,
        }
    finally:
        server.close()
        shutil.rmtree(app, ignore_errors=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
