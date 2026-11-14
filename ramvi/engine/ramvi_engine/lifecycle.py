"""Владелец модели: сериализация операций и кооперативная отмена без GPU-импортов."""

import gc
import math
import threading
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ramvi_engine.protocol import validate_id


@dataclass
class ModelBundle:
    engine: Any
    tokenizer: Any
    draft: Any = None


class LoadFailure(RuntimeError):
    """Сбой загрузки с ресурсами, которые ещё нельзя безопасно освободить."""

    def __init__(self, message: str, bundle: ModelBundle) -> None:
        super().__init__(message)
        self.bundle = bundle


class ModelUnavailable(RuntimeError):
    """Модель не готова к запросу."""


class GenerationCancelled(RuntimeError):
    """Ответ отменён; его нельзя выдавать как завершённый."""


class _Handle:
    def __init__(self, session: "EngineSession", target: str) -> None:
        self._session, self._target = session, target

    def __getattr__(self, name: str) -> Any:
        bundle = self._session._get_bundle()
        return getattr(getattr(bundle, self._target), name)


class EngineSession:
    """Один владелец native-движка; внешние ссылки на его тензоры запрещены.

    cancel проверяется между вызовами next(), CUDA kernel принудительно не прерывается.
    Неотзывчивый вызов требует остановки процесса будущим supervisor (2.8).
    """

    def __init__(
        self,
        factory: Callable[[], ModelBundle],
        *,
        cleanup: Callable[[], None] | None = None,
        timeout: float = 30.0,
        quiesce: Callable[[], None] | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout должен быть конечным положительным числом")
        self._factory, self._timeout = factory, timeout
        self._cleanup = cleanup or self._collect
        self._quiesce = quiesce or (lambda: None)
        self._operation = threading.RLock()
        self._guard = threading.Lock()
        self._cancel = threading.Event()
        self._bundle: ModelBundle | None = None
        self._state = "unloaded"
        self._owner: int | None = None
        self._leases = 0
        self._maintenance = 0
        self._active = False
        self._active_generation_id: str | None = None
        self._cancelled_requests: dict[str, float] = {}
        self._error: str | None = None
        self.tokenizer = _Handle(self, "tokenizer")
        self.draft_handle = _Handle(self, "draft")

    @staticmethod
    def _collect() -> None:
        gc.collect()

    def __getattr__(self, name: str) -> Any:
        # Совместимость App с maxt, ck, g, V, place и метаданными движка.
        with self._guard:
            bundle = self._bundle
        if bundle is None:
            raise AttributeError(name)
        return getattr(bundle.engine, name)

    def _get_bundle(self) -> ModelBundle:
        with self._guard:
            if self._bundle is None or self._state not in ("ready", "generating", "warming"):
                raise ModelUnavailable(f"модель не готова: {self._state}")
            return self._bundle

    def status(self) -> dict[str, Any]:
        with self._guard:
            return {
                "state": self._state,
                "loaded": self._bundle is not None,
                "draft_loaded": self._bundle is not None and self._bundle.draft is not None,
                "active": self._active,
                "request_id": self._active_generation_id,
                "cancel_requested": self._cancel.is_set(),
                "error": self._error,
            }

    def _set(self, state: str, *, error: str | None = None) -> None:
        with self._guard:
            self._state, self._error = state, error

    def _acquire(self, request_id: str | None = None) -> None:
        deadline = time.monotonic() + self._timeout
        while True:
            with self._guard:
                if (
                    request_id is not None
                    and self._cancelled_requests.get(request_id, 0) > time.monotonic()
                ):
                    raise GenerationCancelled("ожидающий запрос отменён")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("движок занят; операция не завершилась в срок")
            if self._operation.acquire(timeout=min(0.05, remaining)):
                return

    @contextmanager
    def request(self, request_id: str | None = None) -> Iterator[None]:
        """Удерживать модель до конца обработки текста/итогового события App."""
        self._acquire(request_id)
        entered = False
        try:
            with self._guard:
                if (
                    request_id is not None
                    and self._cancelled_requests.get(request_id, 0) > time.monotonic()
                ):
                    raise GenerationCancelled("запрос отменён до запуска")
                if self._maintenance or self._bundle is None or self._state != "ready":
                    raise ModelUnavailable(f"модель не готова: {self._state}")
                self._owner = threading.get_ident()
                self._leases += 1
                entered = True
            yield
        finally:
            with self._guard:
                if entered:
                    self._leases -= 1
                    if not self._leases and not self._active:
                        self._owner = None
            self._operation.release()

    def load(self) -> dict[str, Any]:
        self._acquire()
        try:
            self._load()
            return self.status()
        finally:
            self._operation.release()

    def _load(self) -> None:
        if self._bundle is not None:
            if self._state == "error":
                raise ModelUnavailable("сначала необходимо завершить unload")
            return
        self._set("loading")
        try:
            bundle = self._factory()
            with self._guard:
                self._bundle = bundle
            if bundle.engine is None or bundle.tokenizer is None:
                raise ValueError("factory не вернула двигатель и токенизатор")
            self._cancel.clear()
            self._set("ready")
        except Exception as exc:
            if isinstance(exc, LoadFailure):
                with self._guard:
                    self._bundle = exc.bundle
            self._set("error", error=str(exc))
            raise

    def cancel(self, request_id: str | None = None) -> dict[str, Any]:
        if request_id is not None:
            validate_id(request_id)
        with self._guard:
            if request_id is not None:
                now = time.monotonic()
                self._cancelled_requests = {
                    key: end for key, end in self._cancelled_requests.items() if end > now
                }
                self._cancelled_requests[request_id] = now + 60
                while len(self._cancelled_requests) > 512:
                    self._cancelled_requests.pop(next(iter(self._cancelled_requests)))
                if self._active and self._active_generation_id == request_id:
                    self._cancel.set()
                return {"requested": True, "scope": "request", "request_id": request_id}
            if self._active:
                self._cancel.set()
            return {"requested": self._active, "scope": "active_generation"}

    def generate(self, *args: Any, **kwargs: Any) -> Generator[int, None, None]:
        request_id = kwargs.pop("_generation_id", None)
        if request_id is not None:
            validate_id(request_id)
        self._acquire(request_id)
        gen = bundle = None
        started = False
        planning_outcome = "completed"
        try:
            with self._guard:
                if (
                    request_id is not None
                    and self._cancelled_requests.get(request_id, 0) > time.monotonic()
                ):
                    raise GenerationCancelled("запрос отменён до запуска")
                if self._maintenance or self._bundle is None or self._state != "ready":
                    raise ModelUnavailable(f"модель не готова: {self._state}")
                bundle = self._bundle
                self._cancel.clear()
                self._owner = threading.get_ident()
                self._active, self._state = True, "generating"
                self._active_generation_id = request_id
                started = True
            if isinstance(kwargs.get("draft"), _Handle):
                kwargs["draft"] = bundle.draft
            gen = bundle.engine.generate(*args, **kwargs)
            while not self._cancel.is_set():
                try:
                    token = next(gen)
                except StopIteration:
                    break
                if not self._cancel.is_set():
                    yield token
            if self._cancel.is_set():
                raise GenerationCancelled("генерация отменена")
        except (GenerationCancelled, ModelUnavailable):
            planning_outcome = "cancelled"
            raise
        except GeneratorExit:
            planning_outcome = "aborted"
            raise
        except Exception as exc:
            planning_outcome = "error"
            self._set("error", error=str(exc))
            raise
        finally:
            try:
                if gen is not None:
                    gen.close()
            except Exception as exc:
                self._set("error", error=str(exc))
                raise
            finally:
                if bundle is not None:
                    controller = getattr(bundle.engine, "planning_controller", None)
                    if controller is not None:
                        try:
                            controller.observe(bundle.engine, planning_outcome)
                        except Exception as observer_error:
                            # Observation cannot replace a model exception or prevent lock release.
                            controller.journal_error = type(observer_error).__name__
                gen = bundle = None
                kwargs.clear()
                with self._guard:
                    if started:
                        self._active = False
                        self._active_generation_id = None
                        if not self._leases:
                            self._owner = None
                        if self._bundle is not None and self._state == "generating":
                            self._state = "ready"
                self._operation.release()

    def warmup(self, tokens: list[int], *, max_new: int = 2) -> dict[str, Any]:
        if not tokens or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in tokens
        ):
            raise ValueError("warmup требует непустой список token IDs")
        if isinstance(max_new, bool) or not isinstance(max_new, int) or not 1 <= max_new <= 16:
            raise ValueError("warmup max_new от 1 до 16")
        with self.request():
            vocabulary = getattr(self._get_bundle().engine, "V", None)
            if vocabulary is not None and any(token >= vocabulary for token in tokens):
                raise ValueError("warmup token вне словаря модели")
            count = 0
            try:
                for _ in self.generate(tokens, max_new=max_new, draft=self.draft_handle):
                    count += 1
            except GenerationCancelled:
                pass
            return {"generated": count, "cancelled": self._cancel.is_set(), "status": self.status()}

    def _unload(self) -> None:
        bundle = self._bundle
        if bundle is None:
            self._set("unloaded")
            self._cancel.clear()
            self._cleanup()
            return
        self._set("unloading")
        try:
            # На ошибке close сохраняем владельца: DMA ещё может читать его буферы.
            seen: set[int] = set()
            for engine in (bundle.engine, bundle.draft):
                if engine is not None and id(engine) not in seen:
                    seen.add(id(engine))
                    if hasattr(engine, "close"):
                        engine.close()
            del engine
            self._quiesce()
            with self._guard:
                self._bundle = None
            del bundle
            self._cleanup()
            self._cancel.clear()
            self._set("unloaded")
        except Exception as exc:
            self._set("error", error=str(exc))
            raise

    def _maintenance_operation(self, restart: bool) -> dict[str, Any]:
        with self._guard:
            if self._owner == threading.get_ident() and (self._active or self._leases):
                raise TimeoutError("сначала закройте генератор/запрос в текущем потоке")
            self._maintenance += 1
            self._cancel.set()
        acquired = False
        try:
            self._acquire()
            acquired = True
            self._unload()
            if restart:
                self._load()
            return self.status()
        finally:
            with self._guard:
                self._maintenance -= 1
            if acquired:
                self._operation.release()

    def unload(self) -> dict[str, Any]:
        return self._maintenance_operation(False)

    def restart(self) -> dict[str, Any]:
        return self._maintenance_operation(True)

    def close(self) -> None:
        self.unload()
