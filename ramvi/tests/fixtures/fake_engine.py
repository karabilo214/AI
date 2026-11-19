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
# 4.7: план слоёв как у настоящего движка; --resident задаёт слоёв в «видеокарте».
LAYERS = 40
RESIDENT = int(sys.argv[sys.argv.index("--resident") + 1]) if "--resident" in sys.argv else 30
MAXT = int(sys.argv[sys.argv.index("--maxt") + 1]) if "--maxt" in sys.argv else 2048
KIND = os.path.splitext(os.path.basename(MODEL))[0].partition("model-")[2]
requests_served = 0


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
last_body: dict[str, Any] = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                {"instance_id": os.environ["RAMVI_INSTANCE_ID"], "versions": {"api": API_VERSION}}
            )
        elif self.path == "/debug/last_body":  # 7.6: что именно дошло до движка
            with lock:
                self._json({"body": last_body})
        elif self.path == "/debug/cancelled":  # 7.1: проверка отмены по разрыву клиента
            with lock:
                self._json({"cancelled": sorted(cancelled)})
        elif self.path == "/crash":
            os._exit(17)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.headers.get("Authorization") != "Bearer " + os.environ["RAMVI_API_KEY"]:
            self._json({"error": "unauthorized"}, 401)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        global last_body
        with lock:
            last_body = body
        if self.path == "/engine/cancel":
            with lock:
                cancelled.add(body.get("request_id", ""))
            self._json({"cancelled": True})
            return
        served = os.path.basename(MODEL.rstrip("/\\")) or "fake"
        if self.path in ("/v1/messages", "/api/chat"):  # 7.1: другие форматы — одним ответом
            if body.get("model") not in (None, served):
                self._json({"error": {"message": f"загружена только модель {served}"}}, 400)
                return
            text = f"[{served}] {self.path} echo: {body['messages'][-1]['content'][:40]}"
            if self.path == "/v1/messages":
                self._json(
                    {
                        "type": "message",
                        "content": [{"type": "text", "text": text}],
                        "usage": {"input_tokens": 7, "output_tokens": 3},
                    }
                )
            else:
                self._json(
                    {
                        "message": {"role": "assistant", "content": text},
                        "done": True,
                        "prompt_eval_count": 7,
                        "eval_count": 3,
                    }
                )
            return
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        if body.get("model") not in (None, served):  # как validate_capabilities движка
            self._json({"error": {"message": f"загружена только модель {served}"}}, 400)
            return
        if body.get("temperature") != 0:  # чат RAMVI обязан просить жадный выбор (3.10)
            self._json({"error": {"message": "fake engine expects temperature=0"}}, 400)
            return
        question = body["messages"][-1]["content"]
        words = (
            ["word"] * 400
            if "LONG" in question
            else [f"[{served}]", "echo:", question.split("\n")[0][:40]]
        )
        words.append(f"files={question.count('<file name=')}")
        request_id = self.headers.get("X-Request-ID", "")
        global requests_served
        requests_served += 1
        delay = DELAY
        if KIND == "jitter":  # 4.7: нестабильно — каждый третий со второго в 5 раз медленнее
            delay = DELAY * (5 if requests_served % 3 == 2 else 1)
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
            if KIND == "runtime-oom" and RESIDENT > 20:  # 4.7: нехватка VRAM уже при генерации
                plan = {
                    "counts": {"gpu": RESIDENT, "cpu": LAYERS - RESIDENT},
                    "context_tokens": MAXT,
                }
                self._event(
                    {
                        "error": {"message": "CUDA out of memory. Tried to allocate 1 GiB"},
                        "ramvi_metrics": {"plan": plan},
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            for number, delta in enumerate(chunks):
                with lock:
                    if request_id in cancelled:
                        return
                if "CRASH_MID" in question and number == 4:  # 8.6: процесс падает посреди ответа
                    os._exit(17)
                self._event({"choices": [{"delta": delta, "finish_reason": None}]})
                time.sleep(delay)
            metrics = {
                "plan": {
                    "counts": {"gpu": RESIDENT, "cpu": LAYERS - RESIDENT},
                    "context_tokens": MAXT,
                },
                "memory": {"peak_ram_bytes": 2 * 2**30, "peak_vram_reserved_bytes": 6 * 2**30},
            }
            self._event(
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "ramvi_metrics": metrics}
            )
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
