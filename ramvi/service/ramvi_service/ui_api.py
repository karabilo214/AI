"""3.4: локальный HTTP API сервиса для оболочки (Avalonia).

Только 127.0.0.1, ключ в заголовке `Authorization: Bearer`, запросы из браузера
(с заголовком `Origin` или чужим `Host`) отклоняются — так страница в браузере
не может управлять чатом через DNS rebinding. Ответ модели — Server-Sent Events.
Ошибки: `{"error": {"code": "<ключ>", "params": {...}}}`; текст по ключу
показывает интерфейс на выбранном языке (3.14).
"""

import hmac
import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ramvi_service.app_config import AppConfig
from ramvi_service.chat import ChatError, ChatService, Generation
from ramvi_service.onboarding import Onboarding

MAX_BODY = 2 * 2**20
PREFIX = "/ui/v1"
_ID = r"([0-9a-f]{16})"


class UIServer:
    def __init__(
        self,
        chat: ChatService,
        token: str,
        *,
        port: int = 0,
        engine_status: Callable[[], dict[str, Any]] | None = None,
        onboarding: Onboarding | None = None,
        config: AppConfig | None = None,
    ) -> None:
        if len(token) < 16:
            raise ValueError("слишком короткий ключ интерфейса")
        self.chat, self._token = chat, token
        self._engine_status = engine_status or (lambda: {"state": "stopped"})
        self.onboarding = onboarding
        self.config = config
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "UIServer":
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="ramvi-ui-api", daemon=True
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(5)
        self._server.server_close()

    # --- маршруты ------------------------------------------------------------------

    def route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """JSON-ответ или Generation для потока."""
        chat = self.chat
        if method == "GET" and path == "/status":
            return {"engine": self._engine_status(), "limits": asdict(chat.limits)}
        if method == "GET" and path == "/models":
            return {"models": chat.models()}
        if method == "GET" and path == "/chats":
            return {"chats": chat.list_chats()}
        if method == "POST" and path == "/chats":
            model_id = _optional_str(body, "model_id")
            created = chat.create(model_id)
            self._remember(model_id)
            return {"chat": created}
        if path == "/settings":
            if self.config is None:
                raise ChatError("api.not_found")
            if method == "GET":
                return self.config.get()
            if method == "POST":
                return self.config.update(body)
        if path in ("/system", "/onboarding", "/models/import"):
            return self._onboarding_route(method, path, body)
        match = re.fullmatch(f"/chats/{_ID}(/[a-z]+)?", path)
        if match is None:
            raise ChatError("api.not_found")
        chat_id, action = match.group(1), match.group(2) or ""
        if method == "GET" and action == "/messages":
            return {"messages": chat.messages(chat_id)}
        if method == "DELETE" and action == "":
            chat.delete(chat_id)
            return {"deleted": True}
        if method == "POST":
            if action == "/send":
                attachments = body.get("attachments", [])
                return chat.send(chat_id, _required_str(body, "text"), attachments)
            if action == "/regenerate":
                return chat.regenerate(chat_id)
            if action == "/stop":
                return {"stopped": chat.stop(chat_id)}
            if action == "/clear":
                return {"removed": chat.clear(chat_id)}
            if action == "/model":
                model_id = _required_str(body, "model_id")
                changed = chat.set_model(chat_id, model_id)
                self._remember(model_id)
                return {"chat": changed}
            if action == "/title":
                return {"chat": chat.rename(chat_id, _required_str(body, "title"))}
        raise ChatError("api.not_found")

    def _onboarding_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """3.6: мастер первого запуска."""
        onboarding = self.onboarding
        if onboarding is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/system":
            return onboarding.system()
        if method == "GET" and path == "/onboarding":
            return onboarding.state()
        if method == "POST" and path == "/onboarding":
            return onboarding.complete(_optional_str(body, "model_id"))
        if method == "POST" and path == "/models/import":
            return {"model": onboarding.import_model(_required_str(body, "path"))}
        raise ChatError("api.not_found")

    def _remember(self, model_id: str | None) -> None:
        if model_id is not None and self.onboarding is not None:
            self.onboarding.remember(model_id)


def _required_str(body: dict[str, Any], name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str):
        raise ChatError("api.bad_request", field=name)
    return value


def _optional_str(body: dict[str, Any], name: str) -> str | None:
    value = body.get(name)
    if value is not None and not isinstance(value, str):
        raise ChatError("api.bad_request", field=name)
    return value


_STATUS = {
    "api.not_found": 404,
    "chat.not_found": 404,
    "chat.model_not_found": 404,
    "chat.busy": 409,
    "api.unauthorized": 401,
    "api.forbidden": 403,
    "api.too_large": 413,
}


def _handler(server: UIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "RAMVI"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            pass  # содержимое чатов не пишется в логи

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            try:
                self._check_access()
                body = self._body() if method == "POST" else {}
                path = self.path.split("?", 1)[0]
                if not path.startswith(PREFIX + "/"):
                    raise ChatError("api.not_found")
                result = server.route(method, path[len(PREFIX) :], body)
            except ChatError as error:
                self._json(
                    _STATUS.get(error.code, 400),
                    {"error": {"code": error.code, "params": error.params}},
                )
                return
            except Exception as error:  # сервис не падает из-за одного запроса
                self._json(
                    500,
                    {"error": {"code": "api.internal", "params": {"detail": str(error)}}},
                )
                return
            if isinstance(result, Generation):
                self._stream(result)
            else:
                self._json(200, result)

        def _check_access(self) -> None:
            port = server.port
            if self.headers.get("Origin") is not None:
                raise ChatError("api.forbidden")
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                raise ChatError("api.forbidden")
            auth = self.headers.get("Authorization", "")
            if not hmac.compare_digest(auth.encode(), ("Bearer " + server._token).encode()):
                raise ChatError("api.unauthorized")

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ChatError("api.bad_request", field="Content-Length") from None
            if not 0 <= length <= MAX_BODY:
                raise ChatError("api.too_large", limit=MAX_BODY)
            if length == 0:
                return {}
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ChatError("api.bad_request", field="body") from None
            if not isinstance(value, dict):
                raise ChatError("api.bad_request", field="body")
            return value

        def _json(self, status: int, value: Any) -> None:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _stream(self, events: Generation) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for event in events:
                    data = json.dumps(event, ensure_ascii=False, allow_nan=False)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # интерфейс закрыл поток: close() ниже остановит генерацию
            finally:
                events.close()

    return Handler
