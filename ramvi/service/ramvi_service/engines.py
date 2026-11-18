"""3.4: какая модель сейчас в движке и смена модели без перезапуска интерфейса.

`EngineManager` держит один движок (одна модель за раз) под супервизором 3.3.
`ensure(model_id)` поднимает движок с нужной моделью или оставляет текущий.
`SwitchableProvider` — провайдер для `SharedLLM`: каждый запрос идёт в текущий
движок, поэтому смена модели и автоперезапуск (новый порт) не требуют нового
`SharedLLM` и не ломают аренды модулей.

3.5 (UI-03): причина сбоя загрузки берётся из последней трассировки в выводе
движка (`classify_failure`). Однозначные причины — нехватка памяти,
несовместимость, нет доступа, нет файла — повторный запуск не исправит,
поэтому `ensure` сразу отдаёт код причины, не дожидаясь лимита перезапусков.
"""

import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
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


# Код причины → имена исключений (последний сегмент) или подстроки сообщения.
FAILURES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("engine.no_memory", ("NoMemoryPlan", "OutOfMemoryError", "MemoryError"), ("out of memory",)),
    ("engine.incompatible", ("NotImplementedError",), ()),
    ("engine.no_access", ("PermissionError",), ()),
    ("engine.model_missing", ("FileNotFoundError", "NotADirectoryError"), ()),
)
_EXCEPTION_LINE = re.compile(r"^(?P<name>[A-Za-z_][\w.]*)(?::\s?(?P<message>.*))?$")


def classify_failure(lines: list[str]) -> tuple[str, str] | None:
    """Код причины и сообщение по последней трассировке Python в выводе процесса.

    Как печатает Python: заголовок `Traceback ...`, кадры с отступом, затем одна
    строка исключения без отступа. Берётся последняя трассировка (при цепочке —
    итоговое исключение). Неизвестная причина — None: такой сбой стоит перезапустить.
    """
    start = max((i for i, line in enumerate(lines) if line.startswith("Traceback ")), default=-1)
    if start < 0:
        return None
    exception = next((line for line in lines[start + 1 :] if line and not line[0].isspace()), None)
    match = _EXCEPTION_LINE.match(exception or "")
    if match is None:
        return None
    name = match.group("name").rsplit(".", 1)[-1]
    message = (match.group("message") or "").strip()
    for code, names, phrases in FAILURES:
        if name in names or any(p in message.lower() for p in phrases):
            return code, message[:300]
    return None


def served_name(path: str) -> str:
    """Имя, под которым движок обслуживает модель (поле `model` запроса).

    Как в `ramvi_engine.cli`: последняя часть пути. Движок отвергает запрос
    с другим именем, а id модели в базе с ним не совпадает (импорт, slug).
    """
    return path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


class EngineTarget(Target, Protocol):
    """То, что нужно менеджеру от движка сверх `Target`: адрес и ключ."""

    @property
    def base_url(self) -> str: ...
    @property
    def api_key(self) -> str: ...


TOKENIZER_ENV = "RAMVI_TOKENIZER"


def resolve_model(model: Model, root: Path | None) -> Model:
    """Абсолютный путь модели: относительный — от папки приложения (PORT-01: смена
    буквы диска), без неё — от текущей папки, как разрешил бы движок."""
    path = Path(model.path)
    if path.is_absolute():
        return model
    return replace(model, path=str(((root or Path.cwd()) / path).resolve()))


DRAFT_RATIO = 8  # предсказатель хотя бы в 8 раз меньше основной модели


def default_engine(
    model: Model, draft: Model | None = None, maxt: int = 2048, resident: int | None = None
) -> EngineTarget:
    """Настоящий движок для записи модели (тяжёлые импорты только при запуске)."""
    from ramvi_engine.engine_process import EngineProcess
    from ramvi_engine.model_loader import ModelConfig
    from ramvi_engine.planning.automatic import PlanningOptions

    # 3.9: в комплекте токенизатор лежит рядом (runtime/tokenizers), без кеша Hugging Face.
    tokenizer = os.environ.get(TOKENIZER_ENV) or ModelConfig.tokenizer
    return EngineProcess(
        ModelConfig(
            model.path,
            draft=None if draft is None else draft.path,
            tokenizer=tokenizer,
            maxt=maxt,  # 4.5: контекст из профиля задачи
            resident=resident,  # 4.7: безопасный план после нехватки памяти в контрольном прогоне
            planning=PlanningOptions(),
        )
    )


class EngineManager:
    def __init__(
        self,
        models: Models,
        run_dir: Path,
        *,
        make_engine: Callable[..., EngineTarget] = default_engine,
        policy: Callable[[], RestartPolicy] = RestartPolicy,
        interval: float = 1.0,
        request_timeout: float = 600.0,
        model_root: Path | None = None,
        context: Callable[[], int] = lambda: 2048,
        resident_for: Callable[[Model, int], int | None] = lambda model, context: None,
        preflight: Callable[[Model], tuple[str, str] | None] = lambda model: None,
    ) -> None:
        self.models, self.run_dir = models, Path(run_dir)
        self._loaded_context = 0
        self.context = context  # 4.5: контекст из профиля задачи при каждой загрузке
        self.resident_for = resident_for  # 4.7: сохранённый безопасный план (или None)
        self.preflight = preflight  # 4.8: не запускать то, что точно не поместится
        self.model_root = model_root
        self._make_engine, self._policy, self._interval = make_engine, policy, interval
        self.request_timeout = request_timeout
        self._lock = threading.RLock()
        self._model_id: str | None = None
        self._engine: EngineTarget | None = None
        self._supervisor: Supervisor | None = None
        self._provider: LocalEngineProvider | None = None
        self._loading: str | None = None
        self._problem: tuple[str, str] | None = None
        self._problem_model: str | None = None
        self._served_name: str | None = None
        self._draft_id: str | None = None

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
                "problem": None if self._problem is None else self._problem_json(),
                "draft_model_id": self._draft_id if self._model_id is not None else None,
            }

    def _problem_json(self) -> dict[str, Any]:
        assert self._problem is not None
        return {"code": self._problem[0], "params": self._params(self._problem[1])}

    def _params(self, detail: str) -> dict[str, Any]:
        model_id = self._problem_model
        params: dict[str, Any] = {"model_id": model_id or "", "detail": detail}
        if model_id is not None:
            try:
                params["path"] = resolve_model(self.models.get(model_id), self.model_root).path
            except NotFound:
                pass
        return params

    def _on_change(self, engine: EngineTarget, status: dict[str, Any]) -> None:
        """Монитор супервизора: процесс упал — сохранить причину, пока вывод не сброшен.

        Без блокировки менеджера: `stop()` ждёт поток монитора под ней.
        """
        if status["state"] not in ("restarting", "failed") or engine is not self._engine:
            return
        diagnostics = getattr(engine, "diagnostics", None)
        lines = diagnostics() if callable(diagnostics) else []
        found = classify_failure(lines)
        if found is not None:
            self._problem = found

    def ensure(
        self,
        model_id: str,
        *,
        timeout: float = 600.0,
        resident: int | None = None,
        minimum_context: int = 0,
    ) -> None:
        """Движок с этой моделью готов; иначе EngineError с ключом причины.

        `resident` (4.7) — загрузить с этим числом слоёв в видеопамяти (безопасный план);
        иначе — сохранённый безопасный план модели (`resident_for`), если он есть."""
        if (
            isinstance(minimum_context, bool)
            or not isinstance(minimum_context, int)
            or minimum_context < 0
        ):
            raise ValueError("minimum_context must be a nonnegative integer")
        with self._lock:
            if model_id == self._model_id and self._supervisor is not None and resident is None:
                state = self._supervisor.status()["state"]
                if state == "ready" and self._loaded_context >= minimum_context:
                    return
            try:
                model = resolve_model(self.models.get(model_id), self.model_root)
            except NotFound:
                raise EngineError("engine.model_not_found", model_id=model_id) from None
            refused = self.preflight(model)  # до остановки текущей: рабочая модель остаётся
            if refused is not None:
                code, detail = refused
                raise EngineError(code, model_id=model_id, detail=detail, path=model.path)
            self._stop_locked()
            self._loading = model_id
            self._problem, self._problem_model = None, model_id
            self._served_name = served_name(model.path)
            try:
                draft = self.draft_for(model)
                self._draft_id = None if draft is None else draft.id
                context = max(self.context(), minimum_context)
                self._loaded_context = context
                if resident is None:
                    resident = self.resident_for(model, context)
                options: dict[str, Any] = {"maxt": context}
                if resident is not None:
                    options["resident"] = resident
                engine = self._make_engine(model, draft, **options)
                supervisor = Supervisor(
                    engine,
                    self.run_dir,
                    policy=self._policy(),
                    interval=self._interval,
                    on_change=lambda status: self._on_change(engine, status),
                )
            except Exception as error:  # иначе «загружается» висело бы вечно
                self._loading = None
                raise EngineError("engine.start_failed", detail=str(error)) from error
            self._engine, self._supervisor, self._model_id = engine, supervisor, model_id
            try:
                supervisor.start()
            except Exception as error:
                self._stop_locked()
                raise EngineError("engine.start_failed", detail=str(error)) from error
        try:
            state = self._wait(supervisor, timeout)
        finally:
            with self._lock:
                if self._loading == model_id:
                    self._loading = None
        if self._problem is not None and state["state"] != "ready":
            with self._lock:
                params = self._params(self._problem[1])
                if self._supervisor is supervisor:
                    self._stop_locked()  # перезапуск не поможет
            raise EngineError(self._problem[0], **params)
        if state["state"] != "ready":
            failure = state.get("last_failure") or {}
            raise EngineError(
                "engine.start_failed" if state["state"] == "failed" else "engine.start_timeout",
                detail=failure.get("error") or failure.get("reason") or "",
            )

    def draft_for(self, model: Model) -> Model | None:
        """Предсказатель для угадывания вперёд: самая маленькая другая модель той же
        плотной архитектуры qwen3 (общий токенизатор), не меньше чем в 8 раз меньше.
        Включать ли его на самом деле, решает план движка (`--draft-policy auto`)."""
        if model.architecture != "qwen3" or not model.bytes:
            return None
        candidates = [
            m
            for m in self.models.all()
            if m.id != model.id
            and m.architecture == "qwen3"
            and m.bytes
            and m.bytes * DRAFT_RATIO <= model.bytes
        ]
        if not candidates:
            return None
        return resolve_model(min(candidates, key=lambda m: (m.bytes or 0, m.id)), self.model_root)

    def _wait(self, supervisor: Supervisor, timeout: float) -> dict[str, Any]:
        """Ждать готовности; однозначная причина сбоя прерывает ожидание сразу."""
        deadline = time.monotonic() + timeout
        while True:
            state = supervisor.status()
            if (
                state["state"] in ("ready", "failed")
                or self._problem is not None
                or time.monotonic() >= deadline
            ):
                return state
            time.sleep(0.02)

    def engine_pid(self) -> int | None:
        """PID процесса движка (4.7: page faults во время контрольного прогона)."""
        with self._lock:
            if self._supervisor is None:
                return None
            pid = self._supervisor.status().get("target", {}).get("pid")
        return int(pid) if isinstance(pid, int) else None

    def provider(self) -> LocalEngineProvider:
        """Клиент текущего движка; пересоздаётся, если движок сменил адрес."""
        with self._lock:
            engine, supervisor = self._engine, self._supervisor
            if engine is None or supervisor is None or self._model_id is None:
                raise EngineError("engine.no_model")
            if supervisor.status()["state"] != "ready":
                if self._problem is not None:
                    raise EngineError(self._problem[0], **self._params(self._problem[1]))
                raise EngineError("engine.not_ready")
            url = engine.base_url
            if self._provider is None or self._provider.base_url != url:
                self._provider = LocalEngineProvider(
                    url,
                    self._served_name or self._model_id,
                    api_key=engine.api_key,
                    timeout=self.request_timeout,
                )
            return self._provider

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def unload(self) -> None:
        """«Выгрузить модель» (3.17): как stop, и прежняя причина сбоя больше не показывается."""
        with self._lock:
            self._stop_locked()
            self._problem = self._problem_model = None

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
        minimum = request.pop("ramvi_min_context", 0)
        if model_id is not None:
            self.manager.ensure(model_id, **({"minimum_context": minimum} if minimum else {}))
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
        minimum = request.pop("ramvi_min_context", 0)
        if model_id is not None:
            if self.manager.model_id != model_id or self.manager.status()["state"] != "ready":
                yield {"type": "status", "state": "loading", "model_id": model_id}
            self.manager.ensure(model_id, **({"minimum_context": minimum} if minimum else {}))
            if cancel.is_set():
                raise InterruptedError("request cancelled")
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
