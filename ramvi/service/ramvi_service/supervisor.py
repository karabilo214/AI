"""3.3: супервизор процесса — запуск, мониторинг, перезапуск, discovery.

Один общий механизм для движка и для worker модулей (2.14):

* `Supervisor` следит за целью (`Target`): процесс жив и отвечает на проверку;
  при падении или зависании перезапускает её с паузой, растущей вдвое;
  слишком частые падения (`RestartPolicy`) — состояние `failed`, дальше только
  ручной `restart()`, без бесконечного цикла.
* Файл состояния `<run_dir>/<name>.json` — discovery для интерфейса и других
  локальных клиентов: состояние, адрес, PID, число перезапусков. Секретов в нём
  нет. Рядом `<name>.lock` — блокировка ОС, которую держит владелец. Пока она
  занята, второй супервизор с тем же именем не запускается и ничего не трогает.
* Супервизор останавливает только процессы, которые сам запустил (через
  `ProcessSupervisor`: группа процессов / Job Object). PID из старого файла
  никогда не завершается: он мог достаться чужому процессу. Движок прежнего
  владельца уходит сам — по закрытию stdin (`ParentWatch`).
"""

import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

STATE_VERSION = 1
_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class Target(Protocol):
    """Управляемый процесс: движок (`EngineProcess`) или worker (`ProcessTarget`)."""

    def start(self) -> dict[str, Any]: ...
    def stop(self) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    def probe(self) -> bool: ...
    def describe(self) -> dict[str, Any]: ...


class OwnershipError(RuntimeError):
    """Имя уже занято другим живым владельцем."""

    def __init__(self, message: str, state: dict[str, Any] | None) -> None:
        super().__init__(message)
        self.state = state


@dataclass
class RestartPolicy:
    """Пауза перед перезапуском и защита от бесконечных падений."""

    max_restarts: int = 5  # не больше стольких автоперезапусков за окно
    window: float = 300.0
    base_delay: float = 1.0
    max_delay: float = 30.0
    _history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_restarts < 0 or not 0 < self.window <= 86400:
            raise ValueError("invalid restart limits")
        if not 0 <= self.base_delay <= self.max_delay <= 3600:
            raise ValueError("invalid restart delays")

    def next_delay(self, now: float) -> float | None:
        """Пауза до следующей попытки или None, если лимит за окно исчерпан."""
        while self._history and now - self._history[0] > self.window:
            self._history.popleft()
        if len(self._history) >= self.max_restarts:
            return None
        self._history.append(now)
        delay = float(min(self.max_delay, self.base_delay * 2.0**self._streak))
        self._streak += 1
        return delay

    def succeeded(self) -> None:
        self._streak = 0

    def reset(self) -> None:
        self._history.clear()
        self._streak = 0


class RunLock:
    """Неблокирующая блокировка ОС на файл; снимается ОС, если владелец умер."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: IO[bytes] | None = None

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")  # noqa: SIM115 — держим открытым всё время владения
        try:
            _lock(handle)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        if self._file is not None:
            try:
                _unlock(self._file)
            finally:
                self._file.close()
                self._file = None

    @classmethod
    def held(cls, path: Path) -> bool:
        """Занята ли блокировка кем-то (в том числе этим же процессом)."""
        if not path.exists():
            return False
        probe = cls(path)
        if probe.acquire():
            probe.release()
            return False
        return True


if sys.platform == "win32":
    import msvcrt

    def _lock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _check_name(name: str) -> str:
    if not name or len(name) > 64 or not set(name) <= _NAME_CHARS:
        raise ValueError("supervisor name: lowercase Latin letters, digits, _ and -")
    return name


def _write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: читатель держит файл открытым
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.01)


def discover(run_dir: Path, name: str) -> dict[str, Any] | None:
    """Прочитать файл состояния; `live` — жив ли владелец (держит блокировку).

    Устаревший файл (`live: false`) — только история: его PID и адрес могут
    принадлежать уже другому процессу.
    """
    path = Path(run_dir) / f"{_check_name(name)}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict) or value.get("version") != STATE_VERSION:
        return None
    value["live"] = RunLock.held(path.with_suffix(".lock"))
    return value


class Supervisor:
    """Запуск, мониторинг и перезапуск одной цели; один владелец на имя."""

    def __init__(
        self,
        target: Target,
        run_dir: Path,
        *,
        name: str = "engine",
        policy: RestartPolicy | None = None,
        interval: float = 1.0,
        failure_threshold: int = 3,
        stop_timeout: float = 30.0,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not 0 < interval <= 60 or not 1 <= failure_threshold <= 100:
            raise ValueError("invalid monitor settings")
        if not 0 < stop_timeout <= 3600:
            raise ValueError("invalid stop timeout")
        self.target, self.name = target, _check_name(name)
        self.run_dir = Path(run_dir)
        self.policy = policy or RestartPolicy()
        self.interval, self.failure_threshold = interval, failure_threshold
        self.stop_timeout, self.on_change = stop_timeout, on_change
        self.state_path = self.run_dir / f"{self.name}.json"
        self._lock_file = RunLock(self.run_dir / f"{self.name}.lock")
        self._owner = uuid.uuid4().hex
        self._mutex = threading.RLock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "stopped"
        self._restarts = 0
        self._last_failure: dict[str, Any] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=50)
        self._previous: dict[str, Any] | None = None
        self._started_at: float | None = None

    # --- публичные операции -------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Занять имя и запустить цель в фоне; не блокирует на время загрузки."""
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            if not self._lock_file.acquire():
                raise OwnershipError(
                    f"{self.name}: already managed by another owner",
                    discover(self.run_dir, self.name),
                )
            previous = discover(self.run_dir, self.name)
            self._previous = None if previous is None else _previous_summary(previous)
            self.policy.reset()
            self._halt.clear()
            self._wake.clear()
            self._started_at = time.time()
            self._set_phase("starting", "start")
            self._thread = threading.Thread(
                target=self._run, name=f"ramvi-supervisor-{self.name}", daemon=True
            )
            self._thread.start()
            return self.status()

    def restart(self) -> dict[str, Any]:
        """Ручной перезапуск: сбрасывает лимит падений (выход из `failed`)."""
        with self._mutex:
            if self._thread is None or not self._thread.is_alive():
                return self.start()
            self.policy.reset()
            self._set_phase("restarting", "manual restart")
            self._wake.set()
            return self.status()

    def stop(self) -> dict[str, Any]:
        """Остановить свою цель, монитор и освободить имя."""
        with self._mutex:
            thread = self._thread
            self._halt.set()
            self._wake.set()
        if thread is None:
            # Имя не занимали: файл состояния принадлежит другому владельцу.
            self._stop_target()
            return self.status()
        self._stop_target()  # прерывает долгий старт в потоке монитора
        if thread is not None:
            thread.join(self.stop_timeout)
        self._stop_target()  # на случай, если старт успел завершиться после остановки
        with self._mutex:
            alive = thread is not None and thread.is_alive()
            self._set_phase("stopped", "stop" if not alive else "stop timeout")
            if not alive:
                self._thread = None
                self._lock_file.release()
            return self.status()

    def close(self) -> None:
        self.stop()

    def wait(self, *states: str, timeout: float) -> dict[str, Any]:
        """Ждать одно из состояний не дольше `timeout` секунд."""
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if status["state"] in states or time.monotonic() >= deadline:
                return status
            time.sleep(0.02)

    def status(self) -> dict[str, Any]:
        with self._mutex:
            return {
                "version": STATE_VERSION,
                "name": self.name,
                "state": self._phase,
                "owner": {
                    "id": self._owner,
                    "pid": os.getpid(),
                    "started_at": self._started_at,
                },
                "target": self._describe(),
                "restarts": self._restarts,
                "last_failure": self._last_failure,
                "previous": self._previous,
                "events": list(self._events),
                "updated_at": time.time(),
            }

    # --- монитор -------------------------------------------------------------

    def _run(self) -> None:
        misses = 0
        while not self._halt.is_set():
            phase = self._phase
            if phase in ("starting", "restarting"):
                self._wake.clear()
                try:
                    if phase == "restarting":
                        self._stop_target()
                    self.target.start()
                except Exception as error:  # старт мог упасть по любой причине
                    if self._halt.is_set():
                        break
                    self._failure("start failed", error)
                    continue
                if self._halt.is_set():
                    break
                misses = 0
                self.policy.succeeded()
                with self._mutex:
                    if self._phase != phase:  # во время старта попросили restart
                        continue
                    self._set_phase("ready", "started")
            elif phase == "ready":
                if self._wake.wait(self.interval):
                    self._wake.clear()
                    if self._phase != "ready":
                        continue
                if self._halt.is_set():
                    break
                status = self.target.status()
                if status.get("state") not in ("ready", "running"):
                    self._failure("exited", None, status)
                    continue
                if self.target.probe():
                    misses = 0
                    continue
                misses += 1
                if misses >= self.failure_threshold:
                    self._failure(f"no response {misses} times", None)
            else:  # failed: ждём ручного restart или stop
                self._wake.wait()
                self._wake.clear()

    def _failure(
        self, reason: str, error: BaseException | None, status: dict[str, Any] | None = None
    ) -> None:
        with self._mutex:
            # Статус, по которому заметили сбой: EngineProcess после выхода
            # сразу убирает процесс, и код выхода есть только в last_exit.
            status = status or self.target.status()
            process = status.get("last_exit") or status.get("process") or status
            self._last_failure = {
                "reason": reason,
                "error": None if error is None else f"{type(error).__name__}: {error}",
                "process": process,
                "at": time.time(),
            }
        self._stop_target()
        delay = self.policy.next_delay(time.monotonic())
        if delay is None:
            self._set_phase("failed", reason + "; restart limit reached")
            return
        self._set_phase("restarting", f"{reason}; retry in {delay:g}s")
        with self._mutex:
            self._restarts += 1
        # Ручной restart/stop прерывает паузу.
        if self._wake.wait(delay):
            self._wake.clear()

    # --- служебное -----------------------------------------------------------

    def _stop_target(self) -> None:
        try:
            self.target.stop()
        except Exception as error:
            self._event("stop error", f"{type(error).__name__}: {error}")

    def _describe(self) -> dict[str, Any]:
        try:
            return self.target.describe()
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    def _event(self, phase: str, note: str) -> None:
        self._events.append({"at": time.time(), "state": phase, "note": note})

    def _set_phase(self, phase: str, note: str) -> None:
        with self._mutex:
            self._phase = phase
            self._event(phase, note)
            status = self.status()
            _write_json(self.state_path, status)
        if self.on_change is not None:
            try:
                self.on_change(status)
            except Exception:
                pass  # ошибка подписчика не останавливает монитор


def _previous_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Что осталось от прошлого владельца — только для диагностики, не для kill."""
    raw = state.get("target")
    target: dict[str, Any] = raw if isinstance(raw, dict) else {}
    raw = state.get("owner")
    owner: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return {
        "state": state.get("state"),
        "owner_pid": owner.get("pid"),
        "target_pid": target.get("pid"),
        "base_url": target.get("base_url"),
        "updated_at": state.get("updated_at"),
        "clean_exit": state.get("state") == "stopped",
    }


class ProcessTarget:
    """Адаптер `ProcessSupervisor` (worker модуля 2.14) к `Target`.

    `probe` по умолчанию проверяет только, что процесс жив; можно передать
    функцию, например RPC-запрос `ping` с коротким таймаутом.
    """

    def __init__(self, process: Any, probe: Callable[[Any], bool] | None = None) -> None:
        self.process = process
        self._probe = probe

    def start(self) -> dict[str, Any]:
        status: dict[str, Any] = self.process.start()
        return status

    def stop(self) -> dict[str, Any]:
        status: dict[str, Any] = self.process.stop()
        return status

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = self.process.status()
        return status

    def probe(self) -> bool:
        return self._probe(self.process) if self._probe else True

    def describe(self) -> dict[str, Any]:
        return {"pid": self.process.status().get("pid")}
