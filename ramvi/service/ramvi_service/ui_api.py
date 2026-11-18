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
from pathlib import Path
from typing import Any

from ramvi_service.agent.desktop import AgentDesktop
from ramvi_service.api_server import ApiOptions
from ramvi_service.app_config import AppConfig
from ramvi_service.catalog import Catalog
from ramvi_service.chat import ChatError, ChatService, Generation
from ramvi_service.client_keys import ClientKeyError
from ramvi_service.control_run import ControlRun
from ramvi_service.download_manager import DownloadManager
from ramvi_service.engines import resolve_model
from ramvi_service.guard import Guard
from ramvi_service.i18n import to_json
from ramvi_service.model_control import ModelControl
from ramvi_service.model_fit import ModelFit
from ramvi_service.model_library import ModelLibrary
from ramvi_service.onboarding import Onboarding
from ramvi_service.recommend import rank
from ramvi_service.sections import ModuleDataApi, SectionRegistry
from ramvi_service.voice.service import VoiceService

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
        sections: SectionRegistry | None = None,
        module_data: ModuleDataApi | None = None,
        model_control: ModelControl | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        if len(token) < 16:
            raise ValueError("UI token is too short")
        self.chat, self._token = chat, token
        self.model_control = model_control
        if engine_status is None and model_control is not None:
            engine_status = model_control.status
        self._engine_status = engine_status or (lambda: {"state": "stopped"})
        self.onboarding = onboarding
        self.config = config
        self.sections = sections
        self.catalog = catalog
        self.downloads: DownloadManager | None = None
        self.fit: ModelFit | None = None  # 4.4: статусы совместимости
        self.control: ControlRun | None = None  # 4.7: контрольный прогон
        self.guard: Guard | None = None  # 4.8: подсказки при нехватке памяти
        self.library: ModelLibrary | None = None  # 4.11: по умолчанию / удалить
        self.voice: VoiceService | None = None  # 5.2: голосовой ввод
        self.agent_ui: AgentDesktop | None = None
        self.api: Any = None  # 7.1: внешний API (ApiServer), запускается в main
        self.app_dir: Path | None = None
        self.on_close: list[Callable[[], None]] = []  # после остановки HTTP (база и т.п.)
        self.module_data = module_data
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
        while self.on_close:
            self.on_close.pop()()

    # --- маршруты ------------------------------------------------------------------

    def route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """JSON-ответ или Generation для потока."""
        chat = self.chat
        if path.startswith("/agent/") and self.agent_ui is not None:
            return self.agent_ui.route(method, path, body)
        if method == "GET" and path == "/status":
            return {"engine": self._status(), "limits": asdict(chat.limits)}
        if method == "GET" and path == "/models":
            return {"models": self._with_fit(chat.models())}
        if method == "GET" and path == "/chats":
            return {"chats": chat.list_chats()}
        if method == "POST" and path == "/chats":
            model_id = _optional_str(body, "model_id")
            created = chat.create(model_id)
            self._remember(model_id)
            return {"chat": created}
        if path in ("/engine/load", "/engine/unload"):
            return self._engine_route(method, path, body)
        if path == "/modules" or path.startswith("/modules/"):
            return self._modules_route(method, path, body)
        if method == "GET" and path == "/i18n":
            texts = self.sections.texts() if self.sections is not None else {}
            return {"modules": texts}  # 3.14: переводы модулей; ядро — в каталоге оболочки
        if path == "/sections" or path.startswith("/sections/"):
            return self._sections_route(method, path)
        if path == "/settings":
            if self.config is None:
                raise ChatError("api.not_found")
            if method == "GET":
                return self.config.get()
            if method == "POST":
                result = self.config.update(body)
                if self.api is not None and {"api_port", "network_access", "network_allow"} & set(
                    body
                ):
                    # 7.4: порт и сеть применяются сразу, без перезапуска приложения.
                    error = self.api.apply(ApiOptions.from_settings(self.config.current()))
                    result = {**result, "api_error": error}
                return result
        if path == "/downloads" or path.startswith("/downloads/"):
            return self._downloads_route(method, path, body)
        if path == "/voice" or path.startswith("/voice/"):
            return self._voice_route(method, path, body)
        if path == "/connections" or path.startswith("/connections/"):
            return self._connections_route(method, path, body)
        if method == "GET" and path == "/recommendations":
            return self._recommendations()
        if path in ("/catalog", "/catalog/refresh"):
            return self._catalog_route(method, path)
        control = re.fullmatch(r"/models/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/control", path)
        if control is not None:
            return self._control_route(method, control.group(1))
        verb = re.fullmatch(r"/models/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/(default|delete)", path)
        if verb is not None and method == "POST" and self.library is not None:
            if verb.group(2) == "default":
                return self.library.set_default(verb.group(1))
            delete_file = body.get("delete_file", False)
            if not isinstance(delete_file, bool):
                raise ChatError("api.bad_request", field="delete_file")
            return self.library.delete(verb.group(1), delete_file)
        if path in ("/system", "/onboarding", "/models/import", "/models/scan"):
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

    def _engine_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """3.17: загрузить выбранную модель сразу; «Выгрузить модель» в настройках."""
        control = self.model_control
        if control is None or method != "POST":
            raise ChatError("api.not_found")
        if path == "/engine/load":
            model_id = _required_str(body, "model_id")
            if not self.chat.has_model(model_id):
                raise ChatError("chat.model_not_found", model_id=model_id)
            started = control.load(model_id)
            return {"started": started, "engine": self._status()}
        unloaded = control.unload()
        return {"unloaded": unloaded, "engine": self._status()}

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
        if method == "POST" and path == "/models/scan":
            return {**onboarding.scan_models_dir(), "models": self.chat.models()}
        raise ChatError("api.not_found")

    def _catalog_route(self, method: str, path: str) -> Any:
        """4.2: каталог — локальная копия без сети; обновление — по явному запросу."""
        catalog = self.catalog
        if catalog is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/catalog":
            return self._with_entry_fit(catalog.view())
        if method == "POST" and path == "/catalog/refresh":
            return self._with_entry_fit(catalog.refresh())
        raise ChatError("api.not_found")

    def _downloads_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """4.3: загрузки по записи каталога — очередь, пауза, продолжение, отмена."""
        downloads = self.downloads
        if downloads is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/downloads":
            return {"downloads": downloads.list()}
        if method == "POST" and path == "/downloads":
            return {"download": downloads.start(_required_str(body, "entry_id"))}
        match = re.fullmatch(f"/downloads/{_ID}/(pause|resume|cancel)", path)
        if method != "POST" or match is None:
            raise ChatError("api.not_found")
        action = {"pause": downloads.pause, "resume": downloads.resume, "cancel": downloads.cancel}
        return {"download": action[match.group(2)](match.group(1))}

    def _connections_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """7.2: клиенты внешнего API — список со счётчиками, новый ключ, отзыв."""
        api = self.api
        if api is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/connections":
            running = api._server is not None
            return {
                "api": {
                    "url": api.url if running else None,
                    "port": api.options.port,
                    "running": running,
                },
                "clients": api.clients(),
            }
        if method == "POST" and path == "/connections/keys":
            permissions = body.get("permissions")
            if not isinstance(permissions, list) or not all(
                isinstance(p, str) for p in permissions
            ):
                raise ChatError("api.bad_request", field="permissions")
            try:
                record, key = api.keys.create(_required_str(body, "name"), set(permissions))
            except ClientKeyError as error:
                raise ChatError(error.code, **error.params) from None
            # Ключ целиком — только в этом ответе; дальше в списке только подсказка.
            return {"client": {**record.json(), "running": 0}, "key": key}
        match = re.fullmatch(r"/connections/keys/([0-9a-f]{12})/revoke", path)
        if method == "POST" and match is not None:
            try:
                stopped = api.revoke(
                    match.group(1), cancel_running=body.get("cancel_running", True) is not False
                )
            except KeyError:
                raise ChatError("api.not_found") from None
            return {"revoked": match.group(1), "stopped": stopped}
        raise ChatError("api.not_found")

    def _voice_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """5.2: голосовой ввод — запись, распознавание, устройства, модель."""
        voice = self.voice
        if voice is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/voice":
            return voice.status()
        if method == "GET" and path == "/voice/devices":
            return voice.devices()
        if method == "POST" and path == "/voice/start":
            return voice.start(_optional_str(body, "device"))
        if method == "POST" and path == "/voice/stop":
            return voice.stop()
        if method == "POST" and path == "/voice/cancel":
            return voice.cancel()
        if method == "POST" and path == "/voice/model/download":
            return voice.download_model(_optional_str(body, "model"))
        if method == "POST" and path == "/voice/model/cancel":
            return voice.cancel_download()
        if method == "POST" and path == "/voice/accel/download":
            return voice.download_accel()  # 5.4: ускоритель на видеокарте
        if method == "POST" and path == "/voice/api/key":
            key = body.get("key", "")  # 5.7: в хранилище ОС; в ответе — только «задан/нет»
            if not isinstance(key, str):
                raise ChatError("api.bad_request", field="key")
            return voice.set_api_key(key)
        if method == "POST" and path == "/voice/api/check":
            return voice.check_api(_optional_str(body, "url"), _optional_str(body, "model"))
        raise ChatError("api.not_found")

    def _status(self) -> dict[str, Any]:
        """Статус движка; при нехватке памяти — проверенные подсказки, что сделать (4.8)."""
        status = self._engine_status()
        problem = status.get("problem")
        if self.guard is not None and isinstance(problem, dict):
            if problem.get("code") == "engine.no_memory" and "suggestions" not in problem:
                model_id = (problem.get("params") or {}).get("model_id")
                if model_id:
                    problem["suggestions"] = self.guard.suggestions(model_id)
        return status

    def _control_route(self, method: str, model_id: str) -> Any:
        """4.7: запустить контрольный прогон модели (POST) или узнать его состояние (GET)."""
        if self.control is None:
            raise ChatError("api.not_found")
        if not self.chat.has_model(model_id):
            raise ChatError("chat.model_not_found", model_id=model_id)
        if method == "POST":
            return {"control": self.control.start(model_id)}
        if method == "GET":
            return {"control": self.control.status(model_id)}
        raise ChatError("api.not_found")

    def _recommendations(self) -> dict[str, Any]:
        """4.6: лучшая модель для профиля, причина и альтернативы (каталог + установленные)."""
        if self.fit is None or self.catalog is None:
            raise ChatError("api.not_found")
        view = self._with_entry_fit(self.catalog.view())
        items = [
            {"id": e["id"], "name": e["name"], "installed": False, "compat": e["compat"]}
            for e in view["entries"]
        ]
        items += [
            {"id": m["id"], "name": m["id"], "installed": True, "compat": m["compat"]}
            for m in self._with_fit(self.chat.models())
        ]
        profile = self.fit.profile()
        return {"profile": profile.json(), **rank(items, profile.ranking)}

    def _with_entry_fit(self, view: dict[str, Any]) -> dict[str, Any]:
        """4.4: статус каждой записи каталога по сохранённому заголовку."""
        if self.fit is not None:
            view["profile"] = self.fit.profile().json()
            for entry in view["entries"]:
                entry["compat"] = self.fit.for_entry(entry)
        return view

    def _with_fit(self, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """4.4: статус каждой установленной модели (заголовок из файла, без сети)."""
        if self.fit is None:
            return models
        for item in models:
            stored = resolve_model(self.chat.models_repo.get(item["id"]), self.app_dir)
            item["compat"] = self.fit.for_file(stored.path, item["id"], item["id"])
        return models

    def _sections_route(self, method: str, path: str) -> Any:
        """3.11: разделы интерфейса; модуль загружается при первом открытии раздела."""
        registry = self.sections
        if registry is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/sections":
            return {"sections": registry.list()}
        match = re.fullmatch(r"/sections/([a-z][a-z0-9_]{0,63})/open", path)
        if method == "POST" and match:
            return {"section": registry.open(match.group(1))}
        raise ChatError("api.not_found")

    def _modules_route(self, method: str, path: str, body: dict[str, Any]) -> Any:
        """3.12: данные модулей — список, экспорт, сброс, удаление."""
        api = self.module_data
        if api is None:
            raise ChatError("api.not_found")
        if method == "GET" and path == "/modules":
            return {"modules": api.list()}
        match = re.fullmatch(r"/modules/([a-z][a-z0-9_]{0,63})/(export|reset|uninstall)", path)
        if method != "POST" or match is None:
            raise ChatError("api.not_found")
        module_id, action = match.groups()
        if action == "export":
            return {"export": api.export(module_id)}
        if action == "reset":
            return {"module": api.reset(module_id)}
        purge = body.get("purge", False)
        if not isinstance(purge, bool):
            raise ChatError("api.bad_request", field="purge")
        return {"uninstalled": api.uninstall(module_id, purge)}

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
    "download.not_found": 404,
    "download.unknown_entry": 404,
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
                    {"error": {"code": error.code, "params": to_json(error.params)}},
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
