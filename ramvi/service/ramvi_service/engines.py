"""3.4: какая модель сейчас в движке и смена модели без перезапуска интерфейса.

`EngineManager` держит один движок (одна модель за раз) под супервизором 3.3.
`ensure(model_id)` поднимает движок с нужной моделью или оставляет текущий.
`SwitchableProvider` — провайдер для `SharedLLM`: каждый запрос идёт в текущий
движок, поэтому смена модели и автоперезапуск (новый порт) не требуют нового
`SharedLLM` и не ломают аренды модулей.
"""

import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from ramvi_storage.repositories import Model, Models, NotFound

from ramvi_service.contracts.modules import ProviderCapabilities
from ramvi_service.providers import LocalEngineProvider
from ramvi_service.supervisor import RestartPolicy, Supervisor, Target


class EngineError(RuntimeError):
    """Ошибка с ключом сообщения для интерфейса (переводит UI, 3.14)."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params


class EngineTarget(Target, Protocol):
    """То, что нужно менеджеру от движка сверх `Target`: адрес и ключ."""

    @property
    def base_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...


def default_engine(model: Model) -> EngineTarget:
    """Настоящий движок для записи модели (тяжёлые импорты только при запуске)."""
    from ramvi_engine.engine_process import EngineProcess
    from ramvi_engine.model_loader import ModelConfig
    from ramvi_engine.planning.automatic import PlanningOptions

    return EngineProcess(ModelConfig(model.path, planning=PlanningOptions()))


class EngineManager:
    def __init__(
        self,
        models: Models,
        run_dir: Path,
        *,
        make_engine: Callable[[Model], EngineTarget] = default_engine,
        policy: Callable[[], RestartPolicy] = RestartPolicy,
        interval: float = 1.0,
        request_timeout: float = 600.0,
    ) -> None:
        self.models, self.run_dir = models, Path(run_dir)
        self._make_engine, self._policy, self._interval = make_engine, policy, interval
        self.request_timeout = request_timeout
        self._lock = threading.RLock()
        self._model_id: str | None = None
        self._engine: EngineTarget | None = None
        self._supervisor: Supervisor | None = None
        self._provider: LocalEngineProvider | None = None
        self._loading: str | None = None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def status(self) -> dict[str, Any]:
        with self._lock:
            sup = self._supervisor.status() if self._supervisor else None
            return {
                "model_id": self._model_id,
                "loading": self._loading,
                "state": "stopped" if sup is None else sup["state"],
                "last_failure": None if sup is None else sup["last_failure"],
            }

    def ensure(self, model_id: str, *, timeout: float = 600.0) -> None:
        """Движок с этой моделью готов; иначе EngineError с ключом причины."""
        with self._lock:
            if model_id == self._model_id and self._supervisor is not None:
                state = self._supervisor.status()["state"]
                if state == "ready":
                    return
            try:
                model = self.models.get(model_id)
            except NotFound:
                raise EngineError("engine.model_not_found", model_id=model_id) from None
            self._stop_locked()
            self._loading = model_id
            engine = self._make_engine(model)
            supervisor = Supervisor(
                engine, self.run_dir, policy=self._policy(), interval=self._interval
            )
            self._engine, self._supervisor, self._model_id = engine, supervisor, model_id
            try:
                supervisor.start()
            except Exception as error:
                self._stop_locked()
                raise EngineError("engine.start_failed", detail=str(error)) from error
        try:
            state = supervisor.wait("ready", "failed", timeout=timeout)
        finally:
            with self._lock:
                if self._loading == model_id:
                    self._loading = None
        if state["state"] != "ready":
            failure = state.get("last_failure") or {}
            raise EngineError(
                "engine.start_failed" if state["state"] == "failed" else "engine.start_timeout",
                detail=failure.get("error") or failure.get("reason") or "",
            )

    def provider(self) -> LocalEngineProvider:
        """Клиент текущего движка; пересоздаётся, если движок сменил адрес."""
        with self._lock:
            engine, supervisor = self._engine, self._supervisor
            if engine is None or supervisor is None or self._model_id is None:
                raise EngineError("engine.no_model")
            if supervisor.status()["state"] != "ready":
                raise EngineError("engine.not_ready")
            url = engine.base_url
            if self._provider is None or self._provider.base_url != url:
                self._provider = LocalEngineProvider(
                    url,
                    self._model_id,
                    api_key=engine.api_key,
                    timeout=self.request_timeout,
                )
            return self._provider

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
        self._engine = self._supervisor = self._provider = None
        self._model_id = None


MODEL_FIELD = "ramvi_model_id"


class SwitchableProvider:
    """Потоковый провайдер, который всегда обращается к текущему движку менеджера.

    Если в запросе есть `ramvi_model_id`, модель переключается здесь — уже внутри
    слота `SharedLLM`, то есть после окончания чужой генерации. Перед загрузкой
    поток отдаёт кусок `{"type": "status", "state": "loading", "model_id": ...}`.
    """

    capabilities = ProviderCapabilities(streaming=True)

    def __init__(self, manager: EngineManager) -> None:
        self.manager = manager
        self._active: LocalEngineProvider | None = None
        self._lock = threading.Lock()

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        request = dict(request)
        model_id = request.pop(MODEL_FIELD, None)
        if model_id is not None:
            self.manager.ensure(model_id)
        provider = self.manager.provider()
        with self._lock:
            self._active = provider
        try:
            return provider.complete(request, cancel)
        finally:
            with self._lock:
                self._active = None

    def stream(self, request: dict[str, Any], cancel: threading.Event) -> Iterator[dict[str, Any]]:
        request = dict(request)
        model_id = request.pop(MODEL_FIELD, None)
        if model_id is not None:
            if self.manager.model_id != model_id or self.manager.status()["state"] != "ready":
                yield {"type": "status", "state": "loading", "model_id": model_id}
            self.manager.ensure(model_id)
            if cancel.is_set():
                raise InterruptedError("запрос отменён")
        provider = self.manager.provider()
        with self._lock:
            self._active = provider
        try:
            yield from provider.stream(request, cancel)
        finally:
            with self._lock:
                self._active = None

    def cancel(self) -> None:
        with self._lock:
            provider = self._active
        if provider is not None:
            provider.cancel()

    def unload(self) -> None:
        self.manager.stop()
