"""Общий FIFO-арбитр ресурсов и аренды одной модели, без второго планировщика."""

import contextlib
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Generator, Iterator
from typing import Any

from ramvi_service.contracts.modules import (
    Authority,
    LLMProvider,
    Owner,
    StreamingLLMProvider,
)


def amounts(values: dict[str, int]) -> dict[str, int]:
    for key, value in values.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError("resources must be non-negative integers")
    return dict(values)


class ResourceBusy(RuntimeError):
    """Очередь ресурса переполнена: повторить позже (UI-03 «API занят»)."""


class ResourceBroker:
    def __init__(self, authority: Authority, capacity: dict[str, int]) -> None:
        self.authority, self.capacity = authority, amounts(capacity)
        self._condition = threading.Condition()
        self._queue: deque[tuple[str, Owner]] = deque()
        self._leases: dict[str, tuple[Owner, dict[str, int]]] = {}
        self._quotas: dict[str, dict[str, int]] = {}

    @classmethod
    def with_model_estimate(
        cls, authority: Authority, *, ram: int, vram: int, estimate: Any, cpu_slots: int = 1
    ) -> "ResourceBroker":
        # ResourceEstimate — единый источник памяти, импорт движка не нужен.
        need = estimate.required
        if not estimate.assess(vram_available=vram, ram_available=ram)["estimated_fit"]:
            raise ValueError("the model exceeds the total budget")
        return cls(
            authority,
            {
                "ram_bytes": max(0, ram - need.ram - need.mapped),
                "vram_bytes": max(0, vram - need.vram),
                "cpu_slots": cpu_slots,
            },
        )

    def quota(self, owner: Owner, limits: dict[str, int]) -> None:
        self.authority.require(owner)
        with self._condition:
            self._quotas[owner.token] = amounts(limits)

    def _used(self, token: str | None = None) -> dict[str, int]:
        used: dict[str, int] = {}
        for owner, values in self._leases.values():
            if token is None or token == owner.token:
                for key, value in values.items():
                    used[key] = used.get(key, 0) + value
        return used

    def acquire(
        self,
        owner: Owner,
        values: dict[str, int],
        *,
        timeout: float = 30,
        cancel: threading.Event | None = None,
    ) -> str:
        self.authority.require(owner)
        request = amounts(values)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if any(value > self.capacity.get(key, 0) for key, value in request.items()):
            raise ValueError("the request exceeds the full budget")
        ticket = uuid.uuid4().hex
        end = time.monotonic() + timeout
        with self._condition:
            if len(self._queue) >= 256:
                raise ResourceBusy("resource queue is full")
            if sum(existing == owner for existing, _ in self._leases.values()) >= 64:
                raise PermissionError("lease count limit exceeded")
            self._queue.append((ticket, owner))
            try:
                while True:
                    self.authority.require(owner)
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("lease cancelled")
                    own, used = self._used(owner.token), self._used()
                    quota = self._quotas.get(owner.token, self.capacity)
                    if any(
                        own.get(k, 0) + v > quota.get(k, self.capacity.get(k, 0))
                        for k, v in request.items()
                    ):
                        raise PermissionError("owner quota exceeded")
                    fits = all(
                        used.get(k, 0) + v <= self.capacity.get(k, 0) for k, v in request.items()
                    )
                    if self._queue[0][0] == ticket and fits:
                        self._queue.popleft()
                        self._leases[ticket] = (owner, request)
                        self._condition.notify_all()
                        return ticket
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("resource wait timed out")
                    self._condition.wait(min(0.05, remaining))
            finally:
                self._queue = deque(row for row in self._queue if row[0] != ticket)
                self._condition.notify_all()

    def release(self, owner: Owner, lease: str) -> None:
        with self._condition:
            current = self._leases.get(lease)
            if current is not None and current[0] != owner:
                raise PermissionError("not your lease")
            self._leases.pop(lease, None)
            self._condition.notify_all()

    def release_owner(self, owner: Owner) -> None:
        with self._condition:
            self._leases = {key: value for key, value in self._leases.items() if value[0] != owner}
            self._quotas.pop(owner.token, None)
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "capacity": dict(self.capacity),
                "used": self._used(),
                "queued": len(self._queue),
                "leases": len(self._leases),
            }


class SharedLLM:
    """Модель общая; отмена владельца не выгружает чужую активную беседу."""

    def __init__(
        self,
        authority: Authority,
        provider: LLMProvider,
        on_last_release: Callable[[], None] | None = None,
    ) -> None:
        self.authority, self.provider = authority, provider
        self._broker = ResourceBroker(authority, {"llm": 1})
        self._owners: set[str] = set()
        self._lock = threading.RLock()
        self._finished = threading.Condition(self._lock)
        self._active: tuple[Owner, threading.Event] | None = None
        self._requests: dict[str, set[threading.Event]] = {}
        self._released = False
        self._on_last_release = on_last_release

    def retain(self, owner: Owner) -> None:
        self.authority.require(owner)
        with self._lock:
            self._owners.add(owner.token)
            self._released = False

    def _release_last(self) -> None:
        if not self._owners and self._active is None and not self._released:
            self._released = True
            if self._on_last_release:
                self._on_last_release()

    @contextlib.contextmanager
    def _slot(self, owner: Owner, timeout: float, cancel: threading.Event) -> Iterator[None]:
        """Очередь и аренда единственного слота модели на время одного запроса."""
        self.authority.require(owner)
        with self._lock:
            if owner.token not in self._owners:
                raise PermissionError("no model lease")
        with self._finished:
            # Перебивание: прошлый запрос владельца уже отменён и сейчас останавливается.
            # Новый ждёт его конца, а не получает отказ по квоте.
            end = time.monotonic() + timeout
            while (
                self._active is not None
                and self._active[0] == owner
                and self._active[1].is_set()
                and not cancel.is_set()
            ):
                left = end - time.monotonic()
                if left <= 0:
                    raise TimeoutError("the previous request did not stop")
                self._finished.wait(min(0.05, left))
            if owner.token not in self._owners:
                raise InterruptedError("owner stopped")
            self._requests.setdefault(owner.token, set()).add(cancel)
        lease: str | None = None
        try:
            lease = self._broker.acquire(owner, {"llm": 1}, timeout=timeout, cancel=cancel)
            with self._lock:
                self.authority.require(owner)
                if owner.token not in self._owners:
                    raise InterruptedError("owner stopped")
                if cancel.is_set():
                    raise InterruptedError("request cancelled")
                self._active = (owner, cancel)
            yield
        finally:
            with self._lock:
                if self._active is not None and self._active[1] is cancel:
                    self._active = None
                    self._finished.notify_all()
                requests = self._requests.get(owner.token)
                if requests is not None:
                    requests.discard(cancel)
                    if not requests:
                        self._requests.pop(owner.token, None)
                self._release_last()
            if lease is not None:
                self._broker.release(owner, lease)

    @contextlib.contextmanager
    def exclusive(
        self, owner: Owner, *, timeout: float = 30, cancel: threading.Event | None = None
    ) -> Iterator[None]:
        """Занять слот модели без запроса к ней — загрузить или выгрузить модель (3.17)
        после идущего ответа, а не посреди него."""
        with self._slot(owner, timeout, cancel if cancel is not None else threading.Event()):
            yield

    def complete(
        self,
        owner: Owner,
        request: dict[str, Any],
        *,
        timeout: float = 30,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        cancel = cancel if cancel is not None else threading.Event()
        with self._slot(owner, timeout, cancel):
            result = self.provider.complete(request, cancel)
            self.authority.require(owner)
            if cancel.is_set():
                raise InterruptedError("late response cancelled")
            return result

    def stream(
        self,
        owner: Owner,
        request: dict[str, Any],
        *,
        timeout: float = 30,
        cancel: threading.Event | None = None,
    ) -> Generator[dict[str, Any]]:
        """Как complete, но куски идут по мере генерации; слот занят до конца потока."""
        provider = self.provider
        if not provider.capabilities.streaming or not isinstance(provider, StreamingLLMProvider):
            raise NotImplementedError("Provider did not declare streaming")
        cancel = cancel if cancel is not None else threading.Event()
        with self._slot(owner, timeout, cancel):
            for chunk in provider.stream(request, cancel):
                self.authority.require(owner)
                if cancel.is_set():
                    raise InterruptedError("request cancelled")
                yield chunk

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            self._owners.discard(owner.token)
            for cancel in self._requests.get(owner.token, set()):
                cancel.set()
            if self._active is not None and self._active[0] == owner:
                self.provider.cancel()
            self._release_last()
        # Активный execution slot освобождает только finally complete().
        # Очередь увидит cancellation token, не дожидаясь чужой генерации.

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "owners": len(self._owners),
                "active": self._active is not None,
                "queue": self._broker.status(),
            }
