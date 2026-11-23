"""HTTP-приложение RAMVI: состояние модели, авторизация и маршрутизация."""

import hashlib
import json
import os
import threading
import uuid
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from ramvi_engine.lifecycle import EngineSession, GenerationCancelled, ModelUnavailable
from ramvi_engine.metrics import MemoryMonitor, MetricsStore, plan_snapshot, with_error_metrics
from ramvi_engine.protocol import (
    EventContext,
    EventFactory,
    check_api_version,
    validate_id,
    versions,
)
from ramvi_engine.server.contracts import generation_context
from ramvi_engine.server.events import EventLog
from ramvi_engine.server.lifecycle_routes import PATHS, lifecycle_post
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

# Потолок тела запроса. Файлы пользователя идут в запросе целиком (их лимит — в настройках
# RAMVI, B.15); не помещается в контекст модели — ошибка с числами `context overflow`.
# Потолок лишь защищает процесс и много больше контекста любой модели.
MAX_REQUEST_BYTES = 256 * 2**20


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
        self.lock = threading.RLock()
        self.metrics = MetricsStore()
        self.events = EventLog()
        self.slots = threading.BoundedSemaphore(8)

    def _run(self: Any, *args: Any, **kwargs: Any) -> Any:
        meter = self.metrics.begin(self.name)
        context = kwargs.pop("_event_context", None) or EventContext(meter.id, response_id=meter.id)
        factory = EventFactory(context)
        self.events.publish(factory.emit("generation.started", {"model": self.name}))
        monitor = MemoryMonitor()
        stats: dict[str, Any] = {}
        plan: dict[str, Any] = {}
        recorded = False
        try:
            lease = (
                self.eng.request(request_id=context.request_id)
                if isinstance(self.eng, EngineSession)
                else nullcontext()
            )
            with lease, self.lock:
                if getattr(self.eng, "thinking_required", False):
                    if len(args) > 6:
                        args = (*args[:6], True, *args[7:])
                    else:
                        kwargs["think"] = True
                gen = None
                try:
                    monitor.start(self.eng)
                    kwargs["_metrics_stats"] = stats
                    kwargs["_request_id"] = context.request_id
                    gen = super()._run(*args, **kwargs)
                    for kind, value in gen:
                        meter.observe(kind, value)
                        if kind in ("text", "reasoning"):
                            for start in range(0, len(value), 8192):
                                self.events.publish(
                                    factory.emit(
                                        "generation.delta",
                                        {"kind": kind, "text": value[start : start + 8192]},
                                    )
                                )
                        if kind == "end":
                            plan = plan_snapshot(self.eng)
                            record = meter.finish(stats, value, monitor.stop(), plan, "completed")
                            record["request_id"] = context.request_id
                            record["response_id"] = context.response_id
                            self.metrics.finish(record)
                            recorded = True
                            value["metrics"] = record
                            self.events.publish(
                                factory.emit(
                                    "generation.completed",
                                    {"finish": value["finish"], "metrics": record},
                                )
                            )
                        yield kind, value
                finally:
                    try:
                        if gen is not None:
                            gen.close()
                    finally:
                        monitor.stop()
                        plan = plan_snapshot(self.eng)
            if not recorded:
                raise RuntimeError("генерация не выдала итоговое событие")
        except BaseException as exc:
            if not recorded:
                outcome = (
                    "cancelled"
                    if isinstance(exc, GenerationCancelled)
                    else ("aborted" if isinstance(exc, GeneratorExit) else "error")
                )
                record = meter.finish(
                    stats, None, monitor.stop(), plan, outcome, type(exc).__name__
                )
                record["request_id"] = context.request_id
                record["response_id"] = context.response_id
                self.metrics.finish(record)
                self.events.publish(
                    factory.emit(
                        "generation.failed", {"error_type": type(exc).__name__, "outcome": outcome}
                    )
                )
                if type(exc).__name__ == "OutOfMemoryError":
                    self.events.publish(
                        factory.emit("engine.oom", {"state": "error", "model": self.name})
                    )
                if isinstance(exc, Exception):
                    try:
                        error: Any = exc
                        error.ramvi_metrics = record
                    except (AttributeError, TypeError):
                        pass
            raise

    def effective_draft_loaded(self) -> bool:
        """Учитывать реально загруженный draft после планирования и restart."""
        if self.draft is None:
            return False
        if isinstance(self.eng, EngineSession):
            return bool(self.eng.status()["draft_loaded"])
        return True

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
            "draft": self.effective_draft_loaded(),
            "thinking_required": bool(getattr(engine, "thinking_required", False)),
            "chat_template_source": getattr(engine, "chat_template_source", "tokenizer"),
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
        if getattr(self.eng, "thinking_required", False):
            raise ValueError(
                "Thinking-only модель не поддерживает JSON-ограничение без reasoning adapter"
            )
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
            if callable(app.api_key):  # проверка ключей клиентов (API-02) снаружи движка
                token = h[7:] if h.startswith("Bearer ") else self.headers.get("x-api-key")
                if app.api_key(token, self.path):
                    return True
            elif h == f"Bearer {app.api_key}" or self.headers.get("x-api-key") == app.api_key:
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

        def _api_contract(self: Any) -> bool:
            self._request_id = uuid.uuid4().hex
            try:
                requested_id = self.headers.get("X-Request-ID")
                if requested_id is not None:
                    self._request_id = validate_id(requested_id)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return False
            try:
                requested = self.headers.get("X-RAMVI-API-Version")
                if requested is not None:
                    check_api_version(requested)
            except ValueError as exc:
                self._json({"error": str(exc), "versions": versions()}, 426)
                return False
            return True

        def do_GET(self: Any) -> Any:
            """Отдать состояние сервера и описания совместимых API."""
            if not self._api_contract():
                return
            p = self.path.split("?")[0]
            if p in ("/health", "/"):
                return self._json(
                    {
                        "status": "ok",
                        "model": app.name,
                        "run_metadata": app.metadata(),
                        "versions": versions(),
                        "instance_id": os.environ.get("RAMVI_INSTANCE_ID"),
                    }
                )
            if not self._auth():
                return
            if p == "/engine/versions":
                return self._json({"versions": versions()})
            if p == "/engine/events":
                try:
                    query = parse_qs(self.path.partition("?")[2], max_num_fields=4)
                    if set(query) - {"after", "request_id"} or any(
                        len(value) != 1 for value in query.values()
                    ):
                        raise ValueError("invalid event query")
                    request_id = query.get("request_id", [None])[0]
                    if request_id is not None:
                        validate_id(request_id)
                    return self._json(
                        app.events.read(int(query.get("after", ["0"])[0]), request_id)
                    )
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
            if p == "/engine/metrics":
                return self._json(
                    {
                        "metrics": app.metrics.snapshot(),
                        "plan": plan_snapshot(app.eng),
                        "lifecycle": app.eng.status()
                        if isinstance(app.eng, EngineSession)
                        else None,
                    }
                )
            if p == "/engine/status" and isinstance(app.eng, EngineSession):
                return self._json(app.eng.status())
            if p == "/capabilities":
                from ramvi_engine.capabilities import engine_capabilities

                return self._json(
                    engine_capabilities(app.eng, draft_loaded=app.effective_draft_loaded())
                )
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
            if not self._api_contract():
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
                if length > MAX_REQUEST_BYTES:
                    self.close_connection = True
                    return self._json({"error": {"message": "тело запроса слишком велико"}}, 413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("неполное тело запроса")
                req = validate_request(json.loads(body.decode("utf-8") or "{}"))
            except (ValueError, UnicodeError, TypeError) as exc:
                self.close_connection = True
                return self._json(with_error_metrics({"error": {"message": str(exc)}}, exc), 400)
            except OSError:
                self.close_connection = True
                return
            p = self.path.split("?")[0]
            try:
                if p in PATHS:
                    return self._json(lifecycle_post(app, p, req, generation_context(self)))
                if isinstance(app.eng, EngineSession) and app.eng.status()["state"] not in (
                    "ready",
                    "generating",
                ):
                    raise ModelUnavailable("модель не готова к запросу")
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
            except ModelUnavailable as exc:
                return self._json(
                    with_error_metrics(
                        {"error": {"message": str(exc), "type": "model_unavailable"}}, exc
                    ),
                    503,
                )
            except GenerationCancelled as exc:
                return self._json(
                    with_error_metrics({"error": {"message": str(exc), "type": "cancelled"}}, exc),
                    409,
                )
            except TimeoutError as exc:
                return self._json(
                    with_error_metrics(
                        {"error": {"message": str(exc), "type": "timeout_error"}}, exc
                    ),
                    504,
                )
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
                    with_error_metrics(
                        {"error": {"message": str(e), "type": "invalid_request_error"}}, e
                    ),
                    400,
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
        if isinstance(app.eng, EngineSession):
            app.eng.close()
        else:
            for engine in (app.eng, app.draft):
                if engine is not None and hasattr(engine, "close"):
                    engine.close()
