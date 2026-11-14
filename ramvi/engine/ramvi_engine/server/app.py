"""HTTP-приложение RAMVI: состояние модели, авторизация и маршрутизация."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ramvi_engine.server.routes_anthropic import anthropic_routes
from ramvi_engine.server.routes_ollama import ollama_routes
from ramvi_engine.server.routes_openai import openai_routes
from ramvi_engine.server.streaming import (
    GenerationStream,
    StreamingResponse,
)
from ramvi_engine.server.streaming import _first_stop as _first_stop
from ramvi_engine.server.streaming import _held as _held

# Переходные экспорты прежнего server API; обработчики используют модули напрямую.
from ramvi_engine.server.streaming import _split as _split
from ramvi_engine.server.validation import (
    _maybe_json as _maybe_json,
)
from ramvi_engine.server.validation import (
    _schema_from_ollama as _schema_from_ollama,
)
from ramvi_engine.server.validation import (
    _schema_from_openai as _schema_from_openai,
)
from ramvi_engine.server.validation import (
    _text as _text,
)
from ramvi_engine.server.validation import (
    _tool_result as _tool_result,
)
from ramvi_engine.server.validation import (
    anthropic_messages as anthropic_messages,
)
from ramvi_engine.server.validation import (
    anthropic_tools as anthropic_tools,
)
from ramvi_engine.server.validation import (
    openai_messages as openai_messages,
)
from ramvi_engine.server.validation import (
    validate_capabilities,
    validate_request,
)


class App(GenerationStream):
    """Состояние модели и синхронизация запросов."""

    def __init__(
        self: Any,
        engine: Any,
        tokenizer: Any,
        name: Any,
        draft: Any = None,
        k: Any = None,
        api_key: Any = None,
        force_greedy: Any = False,
    ) -> None:
        """Подготовить состояние и ограничители запросов."""
        self.eng, self.tk, self.name, self.draft, self.k = (engine, tokenizer, name, draft, k)
        self.api_key, self.force_greedy = (api_key, force_greedy)
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(8)

    def metadata(self: Any) -> Any:
        """Вернуть идентификаторы модели, размещение и хеш исходников."""
        engine = self.eng
        model_path = Path(engine.g.path) if hasattr(engine, "g") else None
        weights_id = (
            model_path.name
            if model_path is not None and model_path.name.startswith("sha256-")
            else None
        )
        if weights_id is None and hasattr(engine, "ck") and hasattr(engine.ck, "dir"):
            weights_id = Path(engine.ck.dir).name
        digest = hashlib.sha256()
        source_root = Path(__file__).resolve().parents[1]
        for file in sorted(source_root.rglob("*.py")):
            digest.update(file.relative_to(source_root).as_posix().encode())
            digest.update(file.read_bytes())
        return {
            "weights_id": weights_id,
            "tokenizer": getattr(engine, "tokenizer_fingerprint", None),
            "maxt": getattr(engine, "maxt", None),
            "place": getattr(engine, "place", None),
            "code_sha256": digest.hexdigest(),
            "draft": self.draft is not None,
            "k": self.k,
            "force_greedy": self.force_greedy,
        }

    def build_ids(self: Any, messages: Any, tools: Any = None, think: Any = False) -> Any:
        """Преобразовать сообщения и инструменты в токены шаблона модели."""
        kw = {"tools": tools} if tools else {}
        text = self.tk.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think, **kw
        )
        return self.tk.encode(text, add_special_tokens=False)

    def chat_ids(self: Any, messages: Any, think: Any = False) -> Any:
        """Подготовить токены беседы через прежний интерфейс."""
        return self.build_ids(messages, None, think)

    def constraint(self: Any, schema_or_json: Any) -> Any:
        """schema_or_json: "json" — любой JSON; dict — JSON Schema; None — без ограничения."""
        if schema_or_json is None:
            return None
        from ramvi_engine.generation.constrain import Constraint

        V = getattr(self.eng, "V", None) or len(self.tk)
        return Constraint(self.tk, V, None if schema_or_json == "json" else schema_or_json)  # type: ignore[no-untyped-call]


def make_handler(app: Any) -> Any:
    """Создать HTTP-обработчик, связанный с одним приложением."""

    class H(StreamingResponse, BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self: Any) -> Any:
            """Установить ограничение ожидания входящих данных."""
            self.request.settimeout(15)
            super().setup()

        def log_message(self: Any, *a: Any) -> Any:
            """Оставить журнал HTTP отключённым, как в прежнем сервере."""
            pass

        def _auth(self: Any) -> Any:
            if not app.api_key:
                return True
            h = self.headers.get("Authorization", "")
            if h == f"Bearer {app.api_key}" or self.headers.get("x-api-key") == app.api_key:
                return True
            self.close_connection = True
            self._json(
                {"error": {"message": "invalid api key", "type": "authentication_error"}}, 401
            )
            return False

        def do_OPTIONS(self: Any) -> Any:
            """Отдать параметры CORS без тела ответа."""
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self: Any) -> Any:
            """Отдать состояние сервера и описания совместимых API."""
            p = self.path.split("?")[0]
            if p in ("/health", "/"):
                return self._json(
                    {"status": "ok", "model": app.name, "run_metadata": app.metadata()}
                )
            if not self._auth():
                return
            if p == "/v1/models":
                return self._json(
                    {
                        "object": "list",
                        "data": [{"id": app.name, "object": "model", "owned_by": "llm_app"}],
                    }
                )
            if p == "/api/tags":
                return self._json({"models": [{"name": app.name, "model": app.name}]})
            if p == "/api/version":
                return self._json({"version": "0.1.0-llm_app"})
            self._json({"error": "not found"}, 404)

        def do_POST(self: Any) -> Any:
            """Ограничить очередь запросов и освободить слот после обработки."""
            if not app.slots.acquire(blocking=False):
                self.close_connection = True
                return self._json({"error": {"message": "очередь запросов заполнена"}}, 503)
            try:
                return self._post()
            finally:
                app.slots.release()

        def _post(self: Any) -> Any:
            self._stream_started = False
            if not self._auth():
                return
            try:
                length = self.headers.get("Content-Length", "0")
                if (
                    not length.isascii()
                    or not length.isdecimal()
                    or self.headers.get("Transfer-Encoding")
                ):
                    raise ValueError("некорректный Content-Length или Transfer-Encoding")
                length = int(length)
                if length > 2 * 2**20:
                    self.close_connection = True
                    return self._json({"error": {"message": "тело запроса слишком велико"}}, 413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("неполное тело запроса")
                req = validate_request(json.loads(body.decode("utf-8") or "{}"))
            except (ValueError, UnicodeError, TypeError) as exc:
                self.close_connection = True
                return self._json({"error": {"message": str(exc)}}, 400)
            except OSError:
                self.close_connection = True
                return
            p = self.path.split("?")[0]
            try:
                validate_capabilities(req, app.name, getattr(app.eng, "maxt", 2048), p)
                if p == "/v1/chat/completions":
                    return self.openai_chat(req)
                if p == "/v1/completions":
                    return self.openai_completion(req)
                if p == "/v1/messages":
                    return self.anthropic(req)
                if p in ("/api/chat", "/api/generate"):
                    return self.ollama(req, p == "/api/chat")
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                return
            except TimeoutError as exc:
                return self._json({"error": {"message": str(exc), "type": "timeout_error"}}, 504)
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                AttributeError,
                NotImplementedError,
                RuntimeError,
            ) as e:
                return self._json(
                    {"error": {"message": str(e), "type": "invalid_request_error"}}, 400
                )
            self._json({"error": "not found"}, 404)

        openai_chat, openai_completion = openai_routes(app)
        anthropic = anthropic_routes(app)
        ollama = ollama_routes(app)

    return H


class LimitedHTTPServer(ThreadingHTTPServer):
    """HTTP-сервер с ограниченным числом соединений."""

    daemon_threads = True

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        """Подготовить состояние и ограничители запросов."""
        self.connections = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self: Any, request: Any, client_address: Any) -> Any:
        """Принять соединение, если лимит ещё не исчерпан."""
        if not self.connections.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self: Any, request: Any, client_address: Any) -> Any:
        """Освободить слот соединения после завершения потока."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def serve(app: Any, host: Any = "127.0.0.1", port: Any = 11435) -> Any:
    """Обслуживать HTTP и закрыть ресурсы моделей при остановке."""
    srv = LimitedHTTPServer((host, port), make_handler(app))
    print(
        f"сервер: http://{host}:{port} (OpenAI: /v1/chat/completions; Anthropic: /v1/messages; Ollama: /api/chat)"
        + ("; нужен ключ" if app.api_key else ""),
        flush=True,
    )
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        for engine in (app.eng, app.draft):
            if engine is not None and hasattr(engine, "close"):
                engine.close()
