"""Поддельный движок для проверок чата без модели: тот же HTTP-контракт, что у serve.

Ответ: thinking из двух кусков, затем слова `[<модель>] echo: <начало вопроса>`,
`files=<число вложений>`. Слово LONG в вопросе — 400 слов (для Stop).
Задержка между кусками — FAKE_ENGINE_DELAY секунд (по умолчанию 0.01).

3.5: модель `model-<сбой>` (по имени файла из `--model`) падает при «загрузке»
настоящей трассировкой Python, как движок: oom, cuda-oom, incompatible, denied,
missing, crash (неизвестная причина).
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ramvi_engine.processes import ParentWatch
from ramvi_engine.protocol import API_VERSION

watch = ParentWatch.from_env()
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else ""


class NoMemoryPlan(RuntimeError):
    """Как ramvi_engine.planning.automatic.NoMemoryPlan."""


def load() -> None:
    kind = os.path.splitext(os.path.basename(MODEL))[0].partition("model-")[2]
    print(f"loading {MODEL}", flush=True)
    if kind == "oom":
        try:
            raise RuntimeError("CUDA error: out of memory")
        except RuntimeError as error:
            raise NoMemoryPlan("no estimated GPU/owned-RAM fit after 4 candidates") from error
    if kind == "cuda-oom":
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    if kind == "incompatible":
        raise NotImplementedError("архитектура qwen9 пока не поддержана")
    if kind == "denied":
        raise PermissionError(13, "Permission denied", MODEL)
    if kind == "missing":
        raise FileNotFoundError(2, "No such file or directory", MODEL)
    if kind == "crash":
        raise RuntimeError("weights corrupted at tensor 17")


load()
DELAY = float(os.environ.get("FAKE_ENGINE_DELAY", "0.01"))
cancelled: set[str] = set()
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                {"instance_id": os.environ["RAMVI_INSTANCE_ID"], "versions": {"api": API_VERSION}}
            )
        elif self.path == "/crash":
            os._exit(17)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.headers.get("Authorization") != "Bearer " + os.environ["RAMVI_API_KEY"]:
            self._json({"error": "unauthorized"}, 401)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/engine/cancel":
            with lock:
                cancelled.add(body.get("request_id", ""))
            self._json({"cancelled": True})
            return
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        served = os.path.basename(MODEL.rstrip("/\\")) or "fake"
        if body.get("model") not in (None, served):  # как validate_capabilities движка
            self._json({"error": {"message": f"загружена только модель {served}"}}, 400)
            return
        question = body["messages"][-1]["content"]
        words = (
            ["word"] * 400
            if "LONG" in question
            else [f"[{served}]", "echo:", question.split("\n")[0][:40]]
        )
        words.append(f"files={question.count('<file name=')}")
        request_id = self.headers.get("X-Request-ID", "")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks: list[dict[str, Any]] = [
            {"reasoning_content": "думаю "},
            {"reasoning_content": "ещё"},
        ]
        chunks += [{"content": (" " if i else "") + w} for i, w in enumerate(words)]
        try:
            for delta in chunks:
                with lock:
                    if request_id in cancelled:
                        return
                self._event({"choices": [{"delta": delta, "finish_reason": None}]})
                time.sleep(DELAY)
            self._event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            self._event({"choices": [], "usage": {"completion_tokens": len(words)}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _event(self, value: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _json(self, value: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


server = ThreadingHTTPServer(("127.0.0.1", int(os.environ["RAMVI_ENGINE_PORT"])), Handler)
server.daemon_threads = True
server.serve_forever()
