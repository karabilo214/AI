"""Scoped Services и loopback RPC для изолированного module worker."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, Owner
from ramvi_service.events import EventBus
from ramvi_service.resources import ResourceBroker, SharedLLM
from ramvi_service.scoped import ScopedData, ScopedSecrets
from ramvi_service.transport import local_opener, loopback_url


class ScopedServices:
    def __init__(
        self,
        owner: Owner,
        authority: Authority,
        grants: GrantStore,
        data: ScopedData,
        secrets: ScopedSecrets,
        resources: ResourceBroker,
        events: EventBus,
        llm: SharedLLM | None = None,
        context: ContextProvider | None = None,
        timeout: float = 30,
    ) -> None:
        self.owner, self.authority, self.grants, self.data, self.secrets = (
            owner,
            authority,
            grants,
            data,
            secrets,
        )
        self.resources, self.events, self.llm, self.context = resources, events, llm, context
        self.active = False
        self.timeout = timeout
        self._condition = threading.Condition()
        self._calls = 0
        self._stopping = False
        self._cancel = threading.Event()

    def _permission(self, name: str) -> None:
        self.authority.require(self.owner)
        self.grants.require(self.owner, name)

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.authority.require(self.owner)
        with self._condition:
            if self._stopping:
                raise PermissionError("scope остановлен")
            self._calls += 1
        try:
            return self._dispatch(method, payload)
        finally:
            with self._condition:
                self._calls -= 1
                self._condition.notify_all()

    def _dispatch(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.authority.require(self.owner)
        # Владелец — scope, не входное поле JSON/вывод модели.
        if set(payload) & {"module_id", "owner", "principal_id", "session_id", "token"}:
            raise PermissionError("подмена владельца в payload запрещена")
        if method.startswith("settings."):
            self._permission("settings")
            if method == "settings.get":
                return {"value": self.data.read_json("settings.json", {})}
            if method == "settings.set":
                self.data.write_json("settings.json", payload["value"])
                return {"ok": True}
        if method.startswith("data."):
            self._permission("data")
            if method == "data.get":
                return {"value": self.data.read_json(payload["path"])}
            if method == "data.set":
                self.data.write_json(payload["path"], payload["value"])
                return {"ok": True}
        if method.startswith("secrets."):
            self._permission("secrets")
            if method == "secrets.get":
                return {"value": self.secrets.get(payload["key"])}
            if method == "secrets.set":
                self.secrets.set(payload["key"], payload["value"])
                return {"ok": True}
            if method == "secrets.delete":
                self.secrets.delete(payload["key"])
                return {"ok": True}
        if method.startswith("events."):
            self._permission("events")
            if method == "events.begin":
                return {"response_id": self.events.begin(self.owner)}
            if method == "events.emit":
                return {
                    "delivered": self.events.emit(
                        self.owner, payload["response_id"], payload["name"], payload["payload"]
                    )
                }
            if method == "events.cancel":
                self.events.cancel(self.owner, payload.get("response_id"))
                return {"ok": True}
            if method == "events.subscribe":
                return {
                    "subscription_id": self.events.subscribe(
                        self.owner, size=min(128, payload.get("size", 32))
                    )
                }
            if method == "events.poll":
                return {
                    "event": self.events.poll(
                        self.owner, payload["subscription_id"], min(1, payload.get("timeout", 0))
                    )
                }
            if method == "events.unsubscribe":
                self.events.unsubscribe(self.owner, payload["subscription_id"])
                return {"ok": True}
        if not self.active:
            raise PermissionError("модуль не активен")
        if method.startswith("resources."):
            self._permission("resources")
            if method == "resources.acquire":
                return {
                    "lease_id": self.resources.acquire(
                        self.owner,
                        payload["amounts"],
                        timeout=min(30, payload.get("timeout", 30)),
                        cancel=self._cancel,
                    )
                }
            if method == "resources.release":
                self.resources.release(self.owner, payload["lease_id"])
                return {"ok": True}
        if method == "llm.complete":
            self._permission("llm")
            if self.llm is None:
                raise NotImplementedError("LLM Provider не подключён")
            request = payload["request"]
            if not isinstance(request, dict):
                raise ValueError("LLM request должен быть объектом")
            capabilities = self.llm.provider.capabilities
            if request.get("tools") and not capabilities.tools:
                raise NotImplementedError("Provider не объявил tools")
            if request.get("response_format") and not capabilities.json_schema:
                raise NotImplementedError("Provider не объявил constrained JSON")
            if request.get("stream"):
                raise NotImplementedError(
                    "Scoped complete возвращает один ответ; streaming adapter отдельный"
                )
            rights: list[tuple[str, str]] = []
            if self.llm.provider.capabilities.external:
                rights.append(("egress.text", "*"))
                self.grants.require(self.owner, "egress.text")
            for message in request.get("messages", []):
                content = message.get("content") if isinstance(message, dict) else None
                for part in content if isinstance(content, list) else []:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                        if not self.llm.provider.capabilities.image_input:
                            raise NotImplementedError("провайдер не поддерживает изображения")
                        if self.llm.provider.capabilities.external:
                            rights.append(("egress.image", "*"))
                            self.grants.require(self.owner, "egress.image")
            with self._condition:
                if self._stopping:
                    raise PermissionError("scope остановлен")
                self.llm.retain(self.owner)
            cancel = self.grants.watch(self.owner, rights) if rights else threading.Event()
            try:
                result = self.llm.complete(
                    self.owner,
                    payload["request"],
                    timeout=min(30, payload.get("timeout", 30)),
                    cancel=cancel,
                )
                for permission, application in rights:
                    self.grants.require(self.owner, permission, application)
                return result
            finally:
                if rights:
                    self.grants.unwatch(cancel)
        if method == "context.capture":
            if self.context is None:
                raise NotImplementedError("Context Provider не подключён")
            return {
                "snapshot": asdict(
                    self.context.capture(
                        self.owner,
                        payload["application"],
                        payload["window_id"],
                        kind=payload.get("kind", "ui"),
                        capabilities=self.llm.provider.capabilities if self.llm else None,
                    )
                )
            }
        raise ValueError("неизвестный метод Scoped Services")

    def stop(self) -> None:
        with self._condition:
            self.active = False
            self._stopping = True
            self._cancel.set()
        if self.llm is not None:
            self.llm.release_owner(self.owner)
        self.events.release_owner(self.owner)
        self.resources.release_owner(self.owner)
        if self.context is not None:
            self.context.release_owner(self.owner)
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._calls:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Scoped Services операции не завершились")
                self._condition.wait(left)
        # Запрос мог получить аренду непосредственно перед установкой cancel.
        self.resources.release_owner(self.owner)


class ServiceGateway:
    """Loopback endpoint выдаёт только scope, найденный по случайному bearer token."""

    def __init__(self, authority: Authority) -> None:
        self.authority = authority
        self._scopes: dict[str, ScopedServices] = {}
        self._lock = threading.RLock()
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def bind(self, services: ScopedServices) -> None:
        with self._lock:
            self._scopes[services.owner.token] = services

    def unbind(self, owner: Owner) -> None:
        with self._lock:
            self._scopes.pop(owner.token, None)

    def start(self) -> str:
        if self._server is not None:
            return f"http://127.0.0.1:{self._server.server_port}"
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                self.request.settimeout(1)
                super().setup()

            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                status = 200
                response: dict[str, Any]
                try:
                    self.connection.settimeout(1)
                    if self.path != "/":
                        raise ValueError("неизвестный маршрут")
                    token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                    gateway.authority.resolve(token)
                    with gateway._lock:
                        scope = gateway._scopes[token]
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > 2**20:
                        raise ValueError("слишком большой запрос")
                    request = json.loads(self.rfile.read(length))
                    if (
                        not isinstance(request, dict)
                        or set(request) != {"method", "payload"}
                        or not isinstance(request["payload"], dict)
                        or not isinstance(request["method"], str)
                        or len(request["method"]) > 128
                    ):
                        raise ValueError("неверная форма RPC")
                    response = {"result": scope.call(request["method"], request["payload"])}
                except PermissionError as error:
                    status, response = 403, {"error": str(error), "error_type": "PermissionError"}
                except TimeoutError as error:
                    status, response = 408, {"error": str(error), "error_type": "TimeoutError"}
                except InterruptedError as error:
                    status, response = 409, {"error": str(error), "error_type": "InterruptedError"}
                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    OSError,
                    NotImplementedError,
                    RuntimeError,
                ) as error:
                    status, response = (
                        400,
                        {"error": str(error), "error_type": type(error).__name__},
                    )
                body = json.dumps(response, ensure_ascii=False, allow_nan=False).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass
                self.close_connection = True

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._limit = threading.BoundedSemaphore(32)
                self._active: set[Any] = set()
                self._condition = threading.Condition()
                super().__init__(*args, **kwargs)

            def process_request(self, request: Any, client_address: Any) -> None:
                if not self._limit.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                with self._condition:
                    self._active.add(request)
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    self._release(request)
                    raise

            def _release(self, request: Any) -> None:
                with self._condition:
                    self._active.discard(request)
                    self._limit.release()
                    self._condition.notify_all()

            def process_request_thread(self, request: Any, client_address: Any) -> None:
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self._release(request)

            def close_connections(self) -> None:
                with self._condition:
                    connections = list(self._active)
                for connection in connections:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    finally:
                        connection.close()
                deadline = time.monotonic() + 2
                with self._condition:
                    while self._active:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise TimeoutError("Scoped Services handlers ещё выполняются")
                        self._condition.wait(left)

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="ramvi-services")
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            try:
                self._server.close_connections()
            finally:
                self._server.server_close()
            assert self._thread is not None
            self._thread.join(5)
            if self._thread.is_alive():
                raise TimeoutError("Service Gateway не остановлен")
            self._server, self._thread = None, None


class ServiceClient:
    def __init__(self, url: str, token: str, *, timeout: float = 30) -> None:
        self.url, self.token, self.timeout = loopback_url(url), token, timeout
        self._opener = local_opener()

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            json.dumps({"method": method, "payload": payload}).encode(),
            {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(4 * 2**20 + 1)
                if len(raw) > 4 * 2**20:
                    raise ValueError("ответ Scoped Services превышает лимит")
                return dict(json.loads(raw)["result"])
        except urllib.error.HTTPError as error:
            detail = json.loads(error.read(4096))
            classes = {
                "PermissionError": PermissionError,
                "TimeoutError": TimeoutError,
                "InterruptedError": InterruptedError,
                "NotImplementedError": NotImplementedError,
                "ValueError": ValueError,
                "RuntimeError": RuntimeError,
            }
            exception = classes.get(detail.get("error_type"), RuntimeError)
            raise exception(detail.get("error", "Scoped Services failure")) from error
