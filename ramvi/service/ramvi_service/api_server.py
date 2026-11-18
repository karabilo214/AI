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

По умолчанию слушает только 127.0.0.1. Сеть (7.4) — только по явному включению
`network_access`: тогда слушает все адреса, но не с этого компьютера пускаются
только адреса из списка `network_allow` и только ключи с правом `remote`; даже
«жив ли сервис» без ключа не отвечает. Трафик не шифруется — для внешней сети
перед RAMVI ставится прокси с TLS. Настройки применяются сразу (`apply`). Discovery — файл `<run>/api.json`: адрес,
порт, маршруты, PID; секретов в нём нет. Сервис можно запустить без окна
(`python -m ramvi_service.app --headless`), чтобы IDE работала без интерфейса.
"""

import contextlib
import http.client
import ipaddress
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
    host: str = "127.0.0.1"  # 0.0.0.0 — сеть, только по явному включению
    port: int = 11435  # 0 — свободный порт
    queue_timeout: float = 600.0  # сколько ждать своей очереди к модели
    read_timeout: float = 3600.0  # ответ движка (длинная генерация)
    allow: tuple[str, ...] = ()  # сети, откуда можно при включённой сети (CIDR)

    @classmethod
    def from_settings(cls, settings: Any) -> "ApiOptions":
        allow = tuple(p.strip() for p in settings.network_allow.split(",") if p.strip())
        host = "0.0.0.0" if settings.network_access else "127.0.0.1"
        return cls(host=host, port=settings.api_port, allow=allow)

    @property
    def network(self) -> bool:
        return self.host not in ("127.0.0.1", "localhost", "::1")


def is_local(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def allowed(address: str, allow: tuple[str, ...]) -> bool:
    """Адрес клиента входит в одну из разрешённых сетей."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    for network in allow:
        try:
            if ip in ipaddress.ip_network(network, strict=False):
                return True
        except ValueError:
            continue
    return False


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **params: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.params = status, code, message, params


def _numbers(value: Any, found: dict[str, int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _USAGE_KEYS and isinstance(item, int) and not isinstance(item, bool):
                found[key] = item
            else:
                _numbers(item, found)
    elif isinstance(value, list):
        for item in value:
            _numbers(item, found)


_PROMPT = ("prompt_tokens", "input_tokens", "prompt_eval_count")
_COMPLETION = ("completion_tokens", "output_tokens", "eval_count")
_USAGE_KEYS = frozenset(_PROMPT + _COMPLETION)


def parse_usage(head: bytes, tail: bytes) -> tuple[int, int]:
    """Токены запроса и ответа из ответа движка: JSON, поток SSE или NDJSON.

    OpenAI — `usage.prompt_tokens/completion_tokens`, Anthropic — `input_tokens`
    (в начале потока) и `output_tokens`, Ollama — `prompt_eval_count/eval_count`.
    Берётся последнее значение каждого поля: в потоке итог приходит в конце."""
    found: dict[str, int] = {}
    text = (head + b"\n" + tail).decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip().removeprefix("data:").strip()
        if not line.startswith("{"):
            continue
        try:
            _numbers(json.loads(line), found)
        except ValueError:
            continue
    for whole in (head, tail):  # ответ одним JSON на несколько строк
        try:
            _numbers(json.loads(whole.decode("utf-8", errors="replace").strip()), found)
        except ValueError:
            continue
    prompt = max((found.get(k, 0) for k in _PROMPT), default=0)
    completion = max((found.get(k, 0) for k in _COMPLETION), default=0)
    return prompt, completion


@dataclass
class _Active:
    """Идущий запрос клиента: отзыв ключа может его остановить."""

    client_id: str
    cancel: threading.Event
    provider: Any = None
    request_id: str | None = None


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
        self._active: list[_Active] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- жизненный цикл -----------------------------------------------------------------

    @property
    def port(self) -> int:
        return 0 if self._server is None else int(self._server.server_address[1])

    @property
    def url(self) -> str:
        """Адрес для программ на этом компьютере (и при включённой сети тоже)."""
        return f"http://127.0.0.1:{self.port}"

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

    def apply(self, options: ApiOptions) -> str | None:
        """Применить новые порт/сеть/список адресов сразу; ошибка — код сообщения."""
        if options == self.options and self._server is not None:
            return None
        self.close()
        self.options = options
        try:
            self.start()
        except OSError:
            return "api.port_busy"
        return None

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
            "network": self.options.network,
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

    def admit(self, address: str) -> bool:
        """Можно ли этому адресу вообще разговаривать с API."""
        return is_local(address) or (self.options.network and allowed(address, self.options.allow))

    def authorize(self, headers: Any, path: str, address: str = "127.0.0.1") -> Any:
        token = None
        auth = headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        token = token or headers.get("x-api-key")
        client = self.keys.verify(token, permission_for(path)) if token else None
        if client is None:
            raise ApiError(401, "api.unauthorized", "missing, wrong, revoked or limited key")
        if not is_local(address) and "remote" not in client.permissions:
            raise ApiError(401, "api.unauthorized", "this key works only on this computer")
        return client

    # --- клиенты (7.2) ------------------------------------------------------------------

    def track(self, client_id: str) -> _Active:
        active = _Active(client_id, threading.Event())
        with self._lock:
            self._active.append(active)
        return active

    def untrack(self, active: _Active) -> None:
        with self._lock:
            if active in self._active:
                self._active.remove(active)

    def running(self, client_id: str) -> int:
        with self._lock:
            return sum(1 for a in self._active if a.client_id == client_id)

    def revoke(self, key_id: str, *, cancel_running: bool = True) -> int:
        """Отозвать ключ: новые запросы сразу отклоняются; `cancel_running` —
        остановить и идущие (в очереди и в генерации). Другие клиенты не затронуты."""
        self.keys.revoke(key_id)
        if not cancel_running:
            return 0
        with self._lock:
            mine = [a for a in self._active if a.client_id == key_id]
        for active in mine:
            active.cancel.set()
            if active.provider is not None and active.request_id is not None:
                cancel_engine(active.provider, active.request_id)
        return len(mine)

    def clients(self) -> list[dict[str, Any]]:
        """Список для экрана «Подключения»: без ключей, со счётчиками и идущими запросами."""
        return [{**k.json(), "running": self.running(k.id)} for k in self.keys.list()]

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

        def _admit(self) -> str:
            address = str(self.client_address[0])
            if not api.admit(address):
                raise ApiError(403, "api.forbidden", "this address is not allowed")
            return address

        def do_GET(self) -> None:
            path = urlsplit(self.path).path.rstrip("/") or "/"
            try:
                address = self._admit()
                if path == "/" and is_local(address):
                    return self._send_json(200, {"service": "ramvi", "status": "ok"})
                api.authorize(self.headers, path, address)
                if path == "/":
                    return self._send_json(200, {"service": "ramvi", "status": "ok"})
                if path in ("/v1/models", "/api/tags", "/api/version"):
                    return self._send_json(200, api.models_payload(path))
                raise ApiError(404, "api.not_found", f"unknown route {path}")
            except ApiError as error:
                self._error(error)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path.rstrip("/")
            try:
                client = api.authorize(self.headers, path, self._admit())
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
            active = api.track(client.id)
            try:
                with (
                    api.request_owner(client.id) as owner,
                    api.llm.exclusive(
                        owner, timeout=api.options.queue_timeout, cancel=active.cancel
                    ),
                ):
                    if active.cancel.is_set():
                        raise InterruptedError("revoked")
                    model_id, provider = api.loaded()  # пока ждали, модель могли сменить
                    api.check_model(body, model_id, provider.model)
                    self._proxy(client, path, body, provider, active)
            except InterruptedError:
                raise ApiError(401, "api.unauthorized", "the key was revoked") from None
            finally:
                api.untrack(active)

        def _proxy(
            self, client: Any, path: str, body: dict[str, Any], provider: Any, active: _Active
        ) -> None:
            target = urlsplit(provider.base_url)
            request_id = uuid.uuid4().hex
            active.provider, active.request_id = provider, request_id
            data = json.dumps(body).encode()
            connection = http.client.HTTPConnection(
                target.hostname or "127.0.0.1", target.port, timeout=api.options.read_timeout
            )
            started = time.monotonic()
            head = tail = b""
            sent = False
            failed = False
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
                failed = upstream.status >= 400
                self.send_response(upstream.status)
                self.send_header(
                    "Content-Type", upstream.getheader("Content-Type") or "application/json"
                )
                self.end_headers()
                while chunk := upstream.read1(65536):
                    if len(head) < 16384:
                        head += chunk[: 16384 - len(head)]
                    tail = (tail + chunk)[-65536:]
                    if active.cancel.is_set():  # ключ отозван — оборвать ответ
                        cancel_engine(provider, request_id)
                        return
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (TimeoutError, BrokenPipeError, ConnectionResetError):
                        cancel_engine(provider, request_id)  # клиент ушёл — остановить генерацию
                        return
            except (OSError, http.client.HTTPException):
                failed = True
                cancel_engine(provider, request_id)
                if not sent:  # ответ клиенту ещё не начат — понятная ошибка
                    raise ApiError(502, "api.engine_error", "the engine did not answer") from None
            finally:
                connection.close()
                seconds = time.monotonic() - started
                prompt, completion = parse_usage(head, tail)
                api.keys.add_usage(client.id, prompt, completion, seconds, error=failed)
                if api.on_usage is not None:
                    api.on_usage(
                        client.id,
                        {
                            "path": path,
                            "seconds": seconds,
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "error": failed,
                        },
                    )

    return Handler


def cancel_engine(provider: Any, request_id: str) -> None:
    """Остановить запрос в движке (по X-Request-ID)."""
    try:
        target = urlsplit(provider.base_url)
        connection = http.client.HTTPConnection(
            target.hostname or "127.0.0.1", target.port, timeout=5
        )
        connection.request(
            "POST",
            "/engine/cancel",
            json.dumps({"request_id": request_id}).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {provider.api_key}"},
        )
        connection.getresponse().read()
        connection.close()
    except (OSError, http.client.HTTPException):
        pass
