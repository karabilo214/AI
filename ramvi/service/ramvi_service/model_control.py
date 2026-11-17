"""3.17 (UI-MODEL-LOAD-01): модель загружается при выборе и остаётся загруженной.

Модель в памяти меняется только тремя путями: выбрали другую модель (`load`),
нажали «Выгрузить модель» (`unload`) или закрыли программу (`EngineManager.stop`
при выходе сервиса). Выгрузки по простою, после ответа или при смене чата нет.

Загрузка и выгрузка занимают слот общей модели (`SharedLLM.exclusive`), как
обычный запрос: идущий ответ не обрывается сменой модели, а сообщение,
отправленное во время загрузки, ждёт её конца и уже не грузит модель заново.
"""

import threading
from collections.abc import Callable
from typing import Any

from ramvi_service.engines import EngineError, EngineManager
from ramvi_service.resources import ResourceBusy, SharedLLM

PRINCIPAL = "local_user"


class ModelControl:
    def __init__(
        self,
        engines: EngineManager,
        llm: SharedLLM,
        stop_generations: Callable[[], int],
        *,
        timeout: float = 600.0,
    ) -> None:
        self.engines, self.llm = engines, llm
        self._stop_generations = stop_generations
        self.timeout = timeout
        self._owner = llm.authority.issue("core", PRINCIPAL, "model_control")
        llm.retain(self._owner)
        self._lock = threading.Lock()
        self._pending: tuple[str, threading.Event] | None = None
        self._threads: list[threading.Thread] = []
        self._error: dict[str, Any] | None = None

    def load(self, model_id: str) -> bool:
        """Начать загрузку в фоне; False — эта модель уже загружена или грузится."""
        with self._lock:
            if self._pending is not None and self._pending[0] == model_id:
                return False
            if self._pending is None:
                status = self.engines.status()
                if status["model_id"] == model_id and status["state"] in (
                    "starting",
                    "ready",
                    "restarting",
                ):
                    return False
            else:
                self._pending[1].set()  # ещё ждёт очереди — больше не нужна
            cancel = threading.Event()
            self._pending, self._error = (model_id, cancel), None
            thread = threading.Thread(
                target=self._run, args=(model_id, cancel), name="ramvi-model-load", daemon=True
            )
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()
        return True

    def _run(self, model_id: str, cancel: threading.Event) -> None:
        error: dict[str, Any] | None = None
        try:
            with self.llm.exclusive(self._owner, timeout=self.timeout, cancel=cancel):
                if not cancel.is_set():
                    self.engines.ensure(model_id, timeout=self.timeout)
        except InterruptedError:
            pass  # выбрали другую модель или выгрузили, пока ждали очереди
        except EngineError as exc:
            error = {"code": exc.code, "params": exc.params}
        except TimeoutError:
            error = {"code": "chat.busy_timeout", "params": {}}
        except ResourceBusy:
            error = {"code": "api.busy", "params": {}}
        except Exception as exc:
            error = {"code": "engine.start_failed", "params": {"detail": str(exc)}}
        finally:
            with self._lock:
                if self._pending is not None and self._pending[1] is cancel:
                    self._pending, self._error = None, error

    def unload(self, *, timeout: float = 60.0) -> bool:
        """«Выгрузить модель»: остановить ответы, дождаться их конца, освободить память.

        True — модель была загружена (или грузилась) и выгружена.
        """
        with self._lock:
            had_pending = self._pending is not None
            if self._pending is not None:
                self._pending[1].set()
                self._pending = None
            self._error = None
        self._stop_generations()
        with self.llm.exclusive(self._owner, timeout=timeout, cancel=threading.Event()):
            loaded = self.engines.model_id is not None
            self.engines.unload()
        return loaded or had_pending

    def wait(self, timeout: float = 30.0) -> None:
        """Для проверок: дождаться фоновых загрузок."""
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        """Статус движка + модель, которая ждёт очереди на загрузку, и ошибка загрузки."""
        status = self.engines.status()
        with self._lock:
            pending, error = self._pending, self._error
        if status["loading"] is None and pending is not None:
            status["loading"] = pending[0]
        if status["problem"] is None and error is not None:
            failed = (error.get("params") or {}).get("model_id")
            # 4.8: отказ новой модели виден и тогда, когда прежняя продолжает работать.
            if status["state"] != "ready" or (failed and failed != status["model_id"]):
                status["problem"] = error
        return status
