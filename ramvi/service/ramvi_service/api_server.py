"""7.1 (API-01): локальный API для внешних приложений и агентов (IDE и т. п.).

Внешний клиент говорит с **сервисом**, а не с движком напрямую:

1. ключ клиента проверяется (`KeyStore`, отдельный ключ на приложение, права по
   маршруту: генерация — `chat`, список моделей — `models`); без ключа — 401;
2. модель: отвечает загруженная модель; имя в запросе должно совпадать с ней (или
   отсутствовать) — иначе понятная ошибка со списком; модель посреди чужой работы не
   переключается;
3. запрос встаёт в **общую очередь модели** (`SharedLLM`) вместе с чатом и агентом:
   слот занят на всё время ответа, чужая генерация не прерывается;
4. тело уходит движку как есть (движок сам умеет форматы OpenAI, Anthropic и Ollama и
   поток), ответ возвращается клиенту по мере генерации. Клиент оборвал связь — запрос
   в движке отменяется.

Слушает только 127.0.0.1 (сеть — 7.4). Discovery — файл `<run>/api.json`: адрес,
порт, маршруты, PID; секретов в нём нет. Сервис можно запустить без окна
(`python -m ramvi_service.app --headless`), чтобы IDE работала без интерфейса.
"""

import contextlib
import http.client
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ramvi_service.client_keys import KeyStore, permission_for
from ramvi_service.resources import SharedLLM

GENERATION = frozenset(
    {"/v1/chat/completions", "/v1/completions", "/v1/messages", "/api/chat", "/api/generate"}
)
MAX_BODY = 8 * 1024 * 1024
DISCOVERY_VERSION = 1


@dataclass(frozen=True)
class ApiOptions:
    host: str = "127.0.0.1"
    port: int = 11435  # 0 — свободный порт
    queue_timeout: float = 600.0  # сколько ждать своей очереди к модели
    read_timeout: float = 3600.0  # ответ движка (длинная генерация)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **params: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.params = status, code, message, params


class ApiServer:
    """HTTP-сервер внешнего API поверх общего доступа к модели."""

    def __init__(
        self,
        keys: KeyStore,
        llm: SharedLLM,
        engines: Any,
        run_dir: Path,
        options: ApiOptions | None = None,
        *,
        on_usage: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """`engines` — `EngineManager` (текущая модель и адрес движка)."""
        self.keys, self.llm, self.engines = keys, llm, engines
        self.run_dir, self.options = Path(run_dir), options or ApiOptions()
        self.on_usage = on_usage
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- жизненный цикл -----------------------------------------------------------------

    @property
    def port(self) -> int:
        return 0 if self._server is None else int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.options.host}:{self.port}"

    @property
    def discovery_file(self) -> Path:
        return self.run_dir / "api.json"

    def start(self) -> None:
        """Занять порт и начать обслуживание; порт занят — OSError."""
        handler = _handler(self)
        server = ThreadingHTTPServer((self.options.host, self.options.port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="ramvi-api", daemon=True)
        self._thread.start()
        self._write_discovery()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        try:
            self.discovery_file.unlink()
        except FileNotFoundError:
            pass

    def _write_discovery(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": DISCOVERY_VERSION,
            "url": self.url,
            "host": self.options.host,
            "port": self.port,
            "pid": os.getpid(),
            "auth": "Authorization: Bearer <key> or x-api-key: <key>",
            "routes": {
                "openai": "/v1/chat/completions",
                "openai_models": "/v1/models",
                "anthropic": "/v1/messages",
                "ollama": "/api/chat",
                "ollama_models": "/api/tags",
            },
            "started_at": time.time(),
        }
        temp = self.discovery_file.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(temp, self.discovery_file)

    # --- запросы ------------------------------------------------------------------------

    def authorize(self, headers: Any, path: str) -> Any:
        token = None
        auth = headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        token = token or headers.get("x-api-key")
        client = self.keys.verify(token, permission_for(path)) if token else None
        if client is None:
            raise ApiError(401, "api.unauthorized", "missing, wrong, revoked or limited key")
        return client

    def loaded(self) -> tuple[str, Any]:
        model_id = self.engines.model_id
        if model_id is None or self.engines.status().get("state") != "ready":
            raise ApiError(503, "api.no_model", "no model is loaded in RAMVI")
        return model_id, self.engines.provider()

    @contextlib.contextmanager
    def request_owner(self, client_id: str) -> Iterator[Any]:
        """Свой владелец на каждый запрос: параллельные запросы одного клиента ждут
        в общей очереди, а не получают отказ по квоте владельца."""
        owner = self.llm.authority.issue("api", client_id, uuid.uuid4().hex)
        self.llm.retain(owner)
        try:
            yield owner
        finally:
            self.llm.release_owner(owner)
            self.llm.authority.revoke(owner)

    def models_payload(self, path: str) -> dict[str, Any]:
        model_id, _ = self.loaded()
        if path == "/api/tags":
            return {"models": [{"name": model_id, "model": model_id}]}
        if path == "/api/version":
            return {"version": "ramvi"}
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "ramvi"}],
        }

    def check_model(self, body: dict[str, Any], model_id: str, served: str) -> None:
        asked = body.get("model")
        names = {model_id.lower(), str(served).lower()}
        if asked not in (None, "") and str(asked).lower() not in names:
            raise ApiError(
                404,
                "api.model_not_loaded",
                f"model {asked!r} is not loaded; loaded: {model_id!r}",
                asked=asked,
                loaded=model_id,
            )
        body["model"] = served  # движок знает модель под своим именем


def _handler(api: ApiServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RAMVI"
        protocol_version = "HTTP/1.0"  # поток идёт до закрытия соединения

        def log_message(self, *args: Any) -> None:
            pass

        def _send_json(self, status: int, value: dict[str, Any]) -> None:
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error: ApiError) -> None:
            self._send_json(
                error.status,
                {"error": {"message": error.message, "type": error.code, "params": error.params}},
            )

        def do_GET(self) -> None:
            path = urlsplit(self.path).path.rstrip("/") or "/"
            try:
                if path == "/":
                    return self._send_json(200, {"service": "ramvi", "status": "ok"})
                api.authorize(self.headers, path)
                if path in ("/v1/models", "/api/tags", "/api/version"):
                    return self._send_json(200, api.models_payload(path))
                raise ApiError(404, "api.not_found", f"unknown route {path}")
            except ApiError as error:
                self._error(error)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path.rstrip("/")
            try:
                client = api.authorize(self.headers, path)
                if path not in GENERATION:
                    raise ApiError(404, "api.not_found", f"unknown route {path}")
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_BODY:
                    raise ApiError(
                        413 if length > MAX_BODY else 400, "api.bad_body", "bad body size"
                    )
                try:
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError):
                    raise ApiError(400, "api.bad_json", "body must be JSON") from None
                if not isinstance(body, dict):
                    raise ApiError(400, "api.bad_json", "body must be a JSON object")
                self._generate(client, path, body)
            except ApiError as error:
                self._error(error)
            except TimeoutError:
                self._error(ApiError(503, "api.busy", "the model queue is busy, try again"))
            except Exception as error:  # noqa: BLE001 — клиент получает ответ, а не обрыв
                with contextlib.suppress(OSError):
                    self._error(
                        ApiError(
                            500, "api.internal", type(error).__name__, detail=type(error).__name__
                        )
                    )

        def _generate(self, client: Any, path: str, body: dict[str, Any]) -> None:
            model_id, provider = api.loaded()
            api.check_model(body, model_id, provider.model)
            cancel = threading.Event()
            with (
                api.request_owner(client.id) as owner,
                api.llm.exclusive(owner, timeout=api.options.queue_timeout, cancel=cancel),
            ):
                model_id, provider = api.loaded()  # пока ждали, модель могли сменить
                api.check_model(body, model_id, provider.model)
                self._proxy(client, path, body, provider)

        def _proxy(self, client: Any, path: str, body: dict[str, Any], provider: Any) -> None:
            target = urlsplit(provider.base_url)
            request_id = uuid.uuid4().hex
            data = json.dumps(body).encode()
            connection = http.client.HTTPConnection(
                target.hostname or "127.0.0.1", target.port, timeout=api.options.read_timeout
            )
            started = time.monotonic()
            tail = b""
            sent = False
            try:
                connection.request(
                    "POST",
                    path,
                    data,
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {provider.api_key}",
                        "X-Request-ID": request_id,
                    },
                )
                upstream = connection.getresponse()
                sent = True
                self.send_response(upstream.status)
                self.send_header(
                    "Content-Type", upstream.getheader("Content-Type") or "application/json"
                )
                self.end_headers()
                while chunk := upstream.read1(65536):
                    tail = (tail + chunk)[-65536:]
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (TimeoutError, BrokenPipeError, ConnectionResetError):
                        self._cancel(provider, request_id)  # клиент ушёл — остановить генерацию
                        return
            except (OSError, http.client.HTTPException):
                self._cancel(provider, request_id)
                if not sent:  # ответ клиенту ещё не начат — понятная ошибка
                    raise ApiError(502, "api.engine_error", "the engine did not answer") from None
            finally:
                connection.close()
                if api.on_usage is not None:
                    api.on_usage(
                        client.id,
                        {"path": path, "seconds": time.monotonic() - started, "tail": tail},
                    )

        @staticmethod
        def _cancel(provider: Any, request_id: str) -> None:
            try:
                target = urlsplit(provider.base_url)
                connection = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
                connection.request(
                    "POST",
                    "/engine/cancel",
                    json.dumps({"request_id": request_id}).encode(),
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {provider.api_key}",
                    },
                )
                connection.getresponse().read()
                connection.close()
            except (OSError, http.client.HTTPException):
                pass

    return Handler
