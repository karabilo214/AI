"""B.12: проверка комплекта для Mac — запускается встроенным Python из RAMVI.app.

Сервис собирается так же, как его запускает оболочка (папка комплекта, модули agent и
character, токенизатор комплекта); облачная модель — по одному короткому запросу в чате,
агенте и персонаже. В конце: загружен ли torch в процессе сервиса. Только стандартная
библиотека и код комплекта; ключи облака — из файла владельца, только в память.

usage: RAMVI.app/Contents/Resources/runtime/python/bin/python3 -P mac_bundle_check.py \\
           <RAMVI.app> <файл ключей> <отчёт.json>
"""

import json
import os
import platform
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

APP = Path(sys.argv[1]).resolve()
RES = APP / "Contents" / "Resources"
sys.path[:0] = [str(RES / "ramvi" / p) for p in ("service", "engine", "storage")]
os.environ.update(
    {
        "RAMVI_TOKENIZER": str(RES / "runtime" / "tokenizers" / "Qwen3-8B"),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join(
            str(RES / "ramvi" / p) for p in ("service", "engine", "storage")
        ),
    }
)

from ramvi_service.app import build  # noqa: E402

TOKEN = "t" * 32
MODEL = "cloud:claude:claude-sonnet-5-5"


class Secrets:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self.values.get((namespace, key))

    def set(self, namespace: str, key: str, value: str) -> None:
        self.values[(namespace, key)] = value

    def delete(self, namespace: str, key: str) -> None:
        self.values.pop((namespace, key), None)


def claude_key(path: Path) -> str:
    lines = [line.strip() for line in path.read_text("utf-8").splitlines()]
    index = lines.index("Claude API key")
    return next(line for line in lines[index + 1 :] if line)


class Api:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}/ui/v1"

    def call(self, method: str, path: str, body: Any = None, timeout: float = 120) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data,
            {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
        if raw.startswith("{") or raw.startswith("["):
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]


def main() -> None:
    key = claude_key(Path(sys.argv[2]))
    data = Path(tempfile.mkdtemp(prefix="ramvi-macbundle-"))
    server, engines = build(data, TOKEN, app_dir=RES, voice_options={"secrets": Secrets()})
    server.start()
    api = Api(server.port)
    report: dict[str, Any] = {
        "machine": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
    }
    try:
        report["system"] = {
            k: v for k, v in api.call("GET", "/system").items() if k in ("cloud_only", "checks")
        }
        report["sections"] = [s["id"] for s in api.call("GET", "/sections")["sections"]]
        api.call("POST", "/cloud/connections", {"kind": "anthropic", "name": "claude", "key": key})
        api.call(
            "POST",
            "/cloud/connections/claude/models",
            {"model": "claude-sonnet-5-5", "enabled": True},
        )

        chat = api.call("POST", "/chats", {"model_id": MODEL})["chat"]
        started = time.monotonic()
        events = api.call(
            "POST", f"/chats/{chat['id']}/send", {"text": "Ответь одним словом: столица Франции?"}
        )
        report["chat"] = {
            "answer": "".join(e.get("text", "") for e in events if e["type"] == "text"),
            "error": events[-1].get("error"),
            "seconds": round(time.monotonic() - started, 2),
        }
        print("chat", report["chat"], flush=True)

        folder = data / "workspace"
        folder.mkdir()
        (folder / "numbers.txt").write_text("3\n5\n7\n")
        api.call(
            "POST",
            "/agent/start",
            {
                "goal": "Прочитай numbers.txt и назови сумму чисел.",
                "workspace": str(folder),
                "model_id": MODEL,
            },
        )
        started = time.monotonic()
        state: dict[str, Any] = {}
        while time.monotonic() - started < 120:
            state = api.call("GET", "/agent/state")
            if (state.get("task") or {}).get("state") not in (None, "running", "queued"):
                break
            time.sleep(0.3)
        task = state.get("task") or {}
        report["agent"] = {
            "state": task.get("state"),
            "result": str(task.get("result"))[:300],
            "seconds": round(time.monotonic() - started, 2),
        }
        print("agent", report["agent"], flush=True)

        api.call("POST", "/sections/character/open")
        command = "/sections/character/commands/dialogue"
        api.call("POST", command, {"operation": "profile", "arguments": {"id": "sarcastic"}})
        started = time.monotonic()
        api.call(
            "POST",
            command,
            {
                "operation": "start",
                "arguments": {"text": "Привет! Представься одним предложением.", "model": MODEL},
            },
        )
        turn: dict[str, Any] = {}
        while time.monotonic() - started < 120:
            status = api.call("POST", command, {"operation": "status", "arguments": {}})
            history = status.get("history") or []
            if history and history[-1].get("status") not in (None, "running"):
                turn = history[-1]
                break
            time.sleep(0.5)
        job = api.call("POST", command, {"operation": "poll", "arguments": {}}).get("job") or {}
        report["character"] = {
            "status": turn.get("status"),
            "reply": str(turn.get("assistant"))[:300],
            "error": job.get("error"),
            "fallback": job.get("fallback"),
            "seconds": round(time.monotonic() - started, 2),
        }
        print("character", report["character"], flush=True)
        report["spending"] = {
            k: api.call("GET", "/cloud")["spending"][k] for k in ("today", "unknown_today")
        }
    finally:
        server.close()
        engines.stop()
        report["torch_in_service_process"] = "torch" in sys.modules
        text = json.dumps(report, ensure_ascii=False, indent=1).replace(key, "***")
        Path(sys.argv[3]).write_text(text)
        print("torch", report["torch_in_service_process"], flush=True)


if __name__ == "__main__":
    main()
