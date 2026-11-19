"""Scoped Services и loopback RPC для изолированного module worker."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, Owner
from ramvi_service.events import EventBus
from ramvi_service.resources import ResourceBroker, SharedLLM
from ramvi_service.scoped import ScopedData, ScopedSecrets
from ramvi_service.transport import local_opener, loopback_url

Payload = dict[str, Any]
Handler = Callable[["ScopedServices", Payload], Payload]
Permission = Callable[[Payload], tuple[str, str]]


@dataclass(frozen=True)
class Route:
    """Обработчик метода Scoped Services и право, которое проверяется до него."""

    run: Handler
    permission: Permission
    active: bool


ROUTES: dict[str, Route] = {}


def handler(
    method: str, *, permission: str | Permission, active: bool = False
) -> Callable[[Handler], Handler]:
    """Регистрирует метод. Право — имя гранта (для любого приложения) или функция от payload.

    active=True — метод доступен только активному модулю."""
    if not permission:
        raise ValueError("the method needs a permission: " + method)
    check: Permission = (lambda _: (permission, "*")) if isinstance(permission, str) else permission

    def register(run: Handler) -> Handler:
        if method in ROUTES:
            raise ValueError("method is already registered: " + method)
        ROUTES[method] = Route(run, check, active)
        return run

    return register


def _context_permission(payload: Payload) -> tuple[str, str]:
    kind = payload.get("kind", "ui")
    if kind not in ("ui", "screen", "auto"):
        raise ValueError("kind, application and window are required")
    return "context." + ("ui" if kind == "auto" else kind), payload["application"]


STREAM_BUFFER = 256  # кусков в буфере на поток; дальше генерация ждёт чтения
STREAM_IDLE = 30.0  # секунд без чтения — поток отменяется и освобождает модель
MAX_STREAMS = 2
PASSED_ERRORS = (
    PermissionError,
    TimeoutError,
    InterruptedError,
    NotImplementedError,
    ValueError,
    RuntimeError,
    KeyError,
)


class LLMStream:
    """Ограниченный буфер кусков одного потокового ответа."""

    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.chunks: deque[dict[str, Any]] = deque()
        self.condition = threading.Condition()
        self.finished = False
        self.error: Exception | None = None
        self.last_read = time.monotonic()
        self.thread: threading.Thread | None = None


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
        database: Any = None,
        voice: Any = None,
        speech: Any = None,
        agent: Any = None,
    ) -> None:
        self.agent = agent  # 6.10: общий Agent Gateway или None
        self.speech = speech
        self.voice = voice  # 5.6: VoiceService (захват и распознавание) или None
        self.database = database  # 3.12: своя база модуля (ModuleDatabase) или None
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
        self._streams: dict[str, LLMStream] = {}
        self._context_history: dict[tuple[str, str], tuple[int, bool, bool]] = {}

    def _permission(self, name: str, application: str = "*") -> None:
        self.authority.require(self.owner)
        self.grants.require(self.owner, name, application)

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.authority.require(self.owner)
        with self._condition:
            if self._stopping:
                raise PermissionError("scope stopped")
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
            raise PermissionError("owner substitution in payload is forbidden")
        route = ROUTES.get(method)
        if route is None:
            raise ValueError("unknown Scoped Services method")
        if route.active and not self.active:
            raise PermissionError("module is not active")
        self._permission(*route.permission(payload))
        return route.run(self, payload)

    @handler("settings.get", permission="settings")
    def _settings_get(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"value": self.data.read_json("settings.json", {})}

    @handler("settings.set", permission="settings")
    def _settings_set(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.data.write_json("settings.json", payload["value"])
        return {"ok": True}

    @handler("data.get", permission="data")
    def _data_get(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"value": self.data.read_json(payload["path"])}

    @handler("data.set", permission="data")
    def _data_set(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.data.write_json(payload["path"], payload["value"])
        return {"ok": True}

    @handler("data.query", permission="data")
    def _db_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.database is None:
            raise RuntimeError("the module has no database of its own (no migrations)")
        return dict(self.database.query(payload["sql"], payload.get("params")))

    @handler("data.execute", permission="data")
    def _db_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.database is None:
            raise RuntimeError("the module has no database of its own (no migrations)")
        return dict(self.database.execute(payload["sql"], payload.get("params")))

    @handler("secrets.get", permission="secrets")
    def _secrets_get(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"value": self.secrets.get(payload["key"])}

    @handler("secrets.set", permission="secrets")
    def _secrets_set(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.secrets.set(payload["key"], payload["value"])
        return {"ok": True}

    @handler("secrets.delete", permission="secrets")
    def _secrets_delete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.secrets.delete(payload["key"])
        return {"ok": True}

    @handler("events.begin", permission="events")
    def _events_begin(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"response_id": self.events.begin(self.owner)}

    @handler("events.emit", permission="events")
    def _events_emit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "delivered": self.events.emit(
                self.owner, payload["response_id"], payload["name"], payload["payload"]
            )
        }

    @handler("events.cancel", permission="events")
    def _events_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.events.cancel(self.owner, payload.get("response_id"))
        return {"ok": True}

    @handler("events.subscribe", permission="events")
    def _events_subscribe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "subscription_id": self.events.subscribe(
                self.owner, size=min(128, payload.get("size", 32))
            )
        }

    @handler("events.poll", permission="events")
    def _events_poll(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": self.events.poll(
                self.owner, payload["subscription_id"], min(1, payload.get("timeout", 0))
            )
        }

    @handler("events.unsubscribe", permission="events")
    def _events_unsubscribe(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.events.unsubscribe(self.owner, payload["subscription_id"])
        return {"ok": True}

    @handler("resources.acquire", permission="resources", active=True)
    def _resources_acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "lease_id": self.resources.acquire(
                self.owner,
                payload["amounts"],
                timeout=min(30, payload.get("timeout", 30)),
                cancel=self._cancel,
            )
        }

    @handler("resources.release", permission="resources", active=True)
    def _resources_release(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.resources.release(self.owner, payload["lease_id"])
        return {"ok": True}

    def _llm_request(
        self, payload: dict[str, Any]
    ) -> tuple[SharedLLM, dict[str, Any], list[tuple[str, str]]]:
        """Общие проверки запроса к модели и дополнительные права, зависящие от содержимого."""
        if self.llm is None:
            raise NotImplementedError("LLM Provider is not connected")
        request = payload["request"]
        if not isinstance(request, dict):
            raise ValueError("LLM request must be an object")
        capabilities = self.llm.provider.capabilities
        if request.get("tools") and not capabilities.tools:
            raise NotImplementedError("Provider did not declare tools")
        if request.get("response_format") and not capabilities.json_schema:
            raise NotImplementedError("Provider did not declare constrained JSON")
        rights: list[tuple[str, str]] = []
        with self._condition:
            history = dict(self._context_history)
        for (application, source), (epoch, has_text, has_image) in history.items():
            permission = "context." + source
            if self.grants.epoch(self.owner, permission, application) != epoch:
                raise PermissionError(
                    "history contains revoked context; start a new module session"
                )
            rights.append((permission, application))
            if capabilities.external:
                if has_text:
                    self.grants.require(self.owner, "egress.text", application)
                    rights.append(("egress.text", application))
                if has_image:
                    self.grants.require(self.owner, "egress.image", application)
                    rights.append(("egress.image", application))
        if capabilities.external:
            rights.append(("egress.text", "*"))
            self.grants.require(self.owner, "egress.text")
        for message in request.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    if not capabilities.image_input:
                        raise NotImplementedError("the provider does not support images")
                    if capabilities.external:
                        rights.append(("egress.image", "*"))
                        self.grants.require(self.owner, "egress.image")
        with self._condition:
            if self._stopping:
                raise PermissionError("scope stopped")
            self.llm.retain(self.owner)
        return self.llm, request, rights

    @handler("llm.capabilities", permission="llm")
    def _llm_capabilities(self, payload: dict[str, Any]) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "available": self.llm is not None,
            "capabilities": asdict(self.llm.provider.capabilities) if self.llm else {},
        }

    @handler("llm.complete", permission="llm", active=True)
    def _llm_complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("request"), dict) and payload["request"].get("stream"):
            raise NotImplementedError(
                "llm.complete returns one response; use llm.stream for a stream"
            )
        llm, request, rights = self._llm_request(payload)
        cancel = self.grants.watch(self.owner, rights) if rights else threading.Event()
        try:
            result = llm.complete(
                self.owner,
                request,
                timeout=min(30, payload.get("timeout", 30)),
                cancel=cancel,
            )
            for permission, application in rights:
                self.grants.require(self.owner, permission, application)
            return result
        finally:
            if rights:
                self.grants.unwatch(cancel)

    @handler("llm.stream", permission="llm", active=True)
    def _llm_stream(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Начать потоковый ответ; куски забираются llm.read, отмена — llm.cancel."""
        if self.llm is None:
            raise NotImplementedError("LLM Provider is not connected")
        if not self.llm.provider.capabilities.streaming:
            raise NotImplementedError("Provider did not declare streaming")
        with self._condition:
            now = time.monotonic()
            for identifier, old in list(self._streams.items()):
                if old.finished and now - old.last_read > STREAM_IDLE:
                    del self._streams[identifier]
            if len(self._streams) >= MAX_STREAMS:
                raise RuntimeError("too many concurrent module streams")
        llm, request, rights = self._llm_request(payload)
        cancel = self.grants.watch(self.owner, rights) if rights else threading.Event()
        stream = LLMStream(cancel)
        identifier = uuid.uuid4().hex
        with self._condition:
            if self._stopping:
                if rights:
                    self.grants.unwatch(cancel)
                raise PermissionError("scope stopped")
            self._streams[identifier] = stream
        stream.thread = threading.Thread(
            target=self._produce,
            args=(stream, llm, request, min(30, payload.get("timeout", 30)), rights),
            name="ramvi-llm-stream",
            daemon=True,
        )
        stream.thread.start()
        return {"stream_id": identifier}

    def _produce(
        self,
        stream: "LLMStream",
        llm: SharedLLM,
        request: dict[str, Any],
        timeout: float,
        rights: list[tuple[str, str]],
    ) -> None:
        # Генератор закрывается явно: сохранённое исключение держит его кадр,
        # и без close() слот модели освободился бы только при сборке мусора.
        chunks: Generator[dict[str, Any]] = llm.stream(
            self.owner, request, timeout=timeout, cancel=stream.cancel
        )
        try:
            for chunk in chunks:
                with stream.condition:
                    # Backpressure: модуль не читает — генерация ждёт, но не дольше STREAM_IDLE.
                    while len(stream.chunks) >= STREAM_BUFFER and not stream.cancel.is_set():
                        if time.monotonic() - stream.last_read > STREAM_IDLE:
                            # Причина записывается до отмены, чтобы чтение не увидело «отменён».
                            stream.error = TimeoutError("the module does not read the stream")
                            stream.cancel.set()
                            raise stream.error
                        stream.condition.wait(0.05)
                    if stream.cancel.is_set():
                        raise InterruptedError("stream cancelled")
                    stream.chunks.append(chunk)
                    stream.condition.notify_all()
            for permission, application in rights:
                self.grants.require(self.owner, permission, application)
        except Exception as error:
            if stream.error is None:
                stream.error = error
        finally:
            try:
                chunks.close()
            except Exception as error:
                if stream.error is None:
                    stream.error = error
            if rights:
                self.grants.unwatch(stream.cancel)
            with stream.condition:
                stream.finished = True
                stream.condition.notify_all()

    def _stream(self, payload: dict[str, Any]) -> "LLMStream":
        with self._condition:
            stream = self._streams.get(payload["stream_id"])
        if stream is None:
            raise ValueError("unknown stream")
        return stream

    @handler("llm.read", permission="llm", active=True)
    def _llm_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ждёт до timeout (не больше 0,5 с) и отдаёт все накопленные куски."""
        stream = self._stream(payload)
        deadline = time.monotonic() + max(0.0, min(0.5, payload.get("timeout", 0.5)))
        with stream.condition:
            # Конец потока — только когда производитель завершился и освободил модель.
            while not stream.finished and (not stream.chunks or stream.cancel.is_set()):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                stream.condition.wait(left)
            if stream.cancel.is_set():
                stream.chunks.clear()  # после отмены старые куски не доставляются
            chunks = [stream.chunks.popleft() for _ in range(min(len(stream.chunks), 64))]
            stream.last_read = time.monotonic()
            stream.condition.notify_all()
            done = stream.finished and not stream.chunks
            error = stream.error
        if done and not chunks:
            with self._condition:
                if self._streams.get(payload["stream_id"]) is stream:
                    del self._streams[payload["stream_id"]]
            if error is not None:
                if isinstance(error, PASSED_ERRORS):
                    raise error
                raise RuntimeError(f"{type(error).__name__}: {error}")
        return {"chunks": chunks, "done": done and not chunks and error is None}

    @handler("llm.cancel", permission="llm")
    def _llm_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Отмена не ждёт остановки генерации; поток сразу перестаёт выдавать куски."""
        with self._condition:
            stream = self._streams.pop(payload["stream_id"], None)
        if stream is not None:
            stream.cancel.set()
            with stream.condition:
                stream.chunks.clear()
                stream.condition.notify_all()
        return {"ok": True}

    @handler("context.capture", permission=_context_permission, active=True)
    def _context_capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.context is None:
            raise NotImplementedError("Context Provider is not connected")
        capture = (
            self.context.capture_best if payload.get("kind") == "auto" else self.context.capture
        )
        snapshot = capture(
            self.owner,
            payload["application"],
            payload["window_id"],
            **({} if payload.get("kind") == "auto" else {"kind": payload.get("kind", "ui")}),
            capabilities=self.llm.provider.capabilities if self.llm else None,
        )
        epoch = self.grants.epoch(self.owner, "context." + snapshot.source, snapshot.application)
        self.context.validate_snapshot(self.owner, snapshot)
        with self._condition:
            self._context_history[snapshot.application, snapshot.source] = (
                epoch,
                snapshot.text is not None or snapshot.structure is not None,
                snapshot.image is not None,
            )
        return {"snapshot": asdict(snapshot)}

    def _agent_gateway(self) -> Any:
        if self.agent is None:
            raise NotImplementedError("Agent is not connected")
        return self.agent

    @handler("agent.create", permission="agent", active=True)
    def _agent_create(self, payload: Payload) -> Payload:
        if set(payload) != {"goal", "binding_id"}:
            raise ValueError("agent.create accepts only goal and binding_id")
        return {
            "task_id": self._agent_gateway().create(
                self.owner, payload["goal"], payload["binding_id"]
            )
        }

    @handler("agent.status", permission="agent", active=True)
    def _agent_status(self, payload: Payload) -> Payload:
        return dict(self._agent_gateway().status(self.owner, payload["task_id"]))

    @handler("agent.events", permission="agent", active=True)
    def _agent_events(self, payload: Payload) -> Payload:
        return dict(
            self._agent_gateway().events(self.owner, payload["task_id"], payload.get("after", 0))
        )

    @handler("agent.cancel", permission="agent")
    def _agent_cancel(self, payload: Payload) -> Payload:
        self._agent_gateway().cancel(self.owner, payload["task_id"])
        return {"ok": True}

    @handler("agent.forget", permission="agent")
    def _agent_forget(self, payload: Payload) -> Payload:
        self._agent_gateway().forget(self.owner, payload["task_id"])
        return {"ok": True}

    def _speech_service(self) -> Any:
        if self.speech is None:
            raise NotImplementedError("speech output is not connected")
        return self.speech

    @handler("speech.catalog", permission="speech")
    def _speech_catalog(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().catalog(self.data))

    @handler("speech.configure", permission="speech", active=True)
    def _speech_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().configure(self.owner, self.data, payload["settings"]))

    @handler("speech.key", permission="speech", active=True)
    def _speech_key(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().key(self.secrets, payload["provider"], payload["value"]))

    @handler("speech.start", permission="speech", active=True)
    def _speech_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().start(self.owner, self.data, self.secrets, payload))

    @handler("speech.status", permission="speech", active=True)
    def _speech_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().status(self.owner, payload.get("job_id")))

    @handler("speech.cancel", permission="speech")
    def _speech_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._speech_service().cancel(self.owner, payload.get("job_id")))

    # --- 5.6: голос для модулей (Character и др.) ---------------------------------------

    def _voice_service(self) -> Any:
        if self.voice is None:
            raise NotImplementedError("voice is not connected")
        return self.voice

    @handler("voice.capabilities", permission="voice")
    def _voice_capabilities(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"capabilities": self._voice_service().capabilities().to_dict()}

    @handler("voice.listen.start", permission="voice", active=True)
    def _voice_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        device = payload.get("device")
        if device is not None and not isinstance(device, str):
            raise ValueError("device must be a string")
        status = self._voice_service().start(device, owner=self.owner)
        return {"state": status["state"], "backend": status["backend"]}

    @handler("voice.listen.stop", permission="voice", active=True)
    def _voice_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._voice_service().stop(owner=self.owner))

    @handler("voice.listen.cancel", permission="voice", active=True)
    def _voice_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"state": self._voice_service().cancel(owner=self.owner)["state"]}

    @handler("voice.session.start", permission="voice", active=True)
    def _voice_session_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        device = payload.get("device")
        if device is not None and not isinstance(device, str):
            raise ValueError("device must be a string")
        return dict(
            self._voice_service().start_session(
                device, owner=self.owner, language=payload.get("language", "auto")
            )
        )

    @handler("voice.session.read", permission="voice", active=True)
    def _voice_session_read(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(
            self._voice_service().read_session(
                owner=self.owner,
                capture_id=payload.get("capture_id"),
                cursor=payload.get("cursor", 0),
            )
        )

    @handler("voice.session.cancel", permission="voice")
    def _voice_session_cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(
            self._voice_service().cancel_session(
                owner=self.owner, capture_id=payload.get("capture_id")
            )
        )

    @handler("voice.session.playback", permission="voice", active=True)
    def _voice_session_playback(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(
            self._voice_service().session_playback(
                payload["active"], owner=self.owner, capture_id=payload.get("capture_id")
            )
        )

    def _voice_release(self) -> None:
        """Остановленный модуль не держит микрофон: его запись отменяется."""
        if self.voice is not None:
            try:
                self.voice.cancel(owner=self.owner)
            except Exception:
                pass  # запись чужая или уже закончилась

    def stop(self) -> None:
        with self._condition:
            self.active = False
            self._stopping = True
            self._cancel.set()
            streams = list(self._streams.values())
            self._streams.clear()
        if self.speech is not None:
            self.speech.release(self.owner)
        for stream in streams:
            stream.cancel.set()
            with stream.condition:
                stream.condition.notify_all()
        if self.llm is not None:
            self.llm.release_owner(self.owner)
        if self.agent is not None:
            self.agent.release_owner(self.owner)
        self._voice_release()
        self.events.release_owner(self.owner)
        self.resources.release_owner(self.owner)
        if self.context is not None:
            self.context.release_owner(self.owner)
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._calls:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Scoped Services operations did not finish")
                self._condition.wait(left)
        for stream in streams:
            if stream.thread is not None:
                stream.thread.join(max(0.0, deadline - time.monotonic()))
                if stream.thread.is_alive():
                    raise TimeoutError("model stream did not stop")
        # An in-flight start may have passed its active check before stop.
        if self.speech is not None:
            self.speech.release(self.owner)
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
                        raise ValueError("unknown route")
                    token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                    gateway.authority.resolve(token)
                    with gateway._lock:
                        scope = gateway._scopes[token]
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > 2**20:
                        raise ValueError("request too large")
                    request = json.loads(self.rfile.read(length))
                    if (
                        not isinstance(request, dict)
                        or set(request) != {"method", "payload"}
                        or not isinstance(request["payload"], dict)
                        or not isinstance(request["method"], str)
                        or len(request["method"]) > 128
                    ):
                        raise ValueError("invalid RPC shape")
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
                            raise TimeoutError("Scoped Services handlers are still running")
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
                raise TimeoutError("Service Gateway did not stop")
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
                    raise ValueError("Scoped Services response exceeds the limit")
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

    def stream(self, request: dict[str, Any], *, timeout: float = 30) -> Iterator[dict[str, Any]]:
        """Куски ответа модели по мере генерации. Прерванный цикл отменяет генерацию."""
        stream_id = self.call("llm.stream", {"request": request, "timeout": timeout})["stream_id"]
        done = False
        try:
            while not done:
                result = self.call("llm.read", {"stream_id": stream_id, "timeout": 0.5})
                yield from result["chunks"]
                done = bool(result["done"])
        finally:
            if not done:
                try:
                    self.call("llm.cancel", {"stream_id": stream_id})
                except (OSError, PermissionError, ValueError, RuntimeError):
                    pass
