"""Host-owned envelope, ограниченные подписки и отбрасывание старого ответа."""

import copy
import math
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from ramvi_service.contracts.modules import Authority, Owner

EVENT_VERSION = 1  # версия формата событий (общий EventFactory движка, version=1)


@dataclass(frozen=True)
class EventSpec:
    required: dict[str, type]
    optional: dict[str, type] | None = None

    def validate(self, payload: dict[str, Any]) -> None:
        optional = self.optional or {}
        if (
            not isinstance(payload, dict)
            or set(self.required) - set(payload)
            or set(payload) - set(self.required) - set(optional)
        ):
            raise ValueError("invalid event fields")
        for key, value in payload.items():
            expected = self.required.get(key, optional.get(key))
            if (
                expected is None
                or not isinstance(value, expected)
                or (expected is int and isinstance(value, bool))
            ):
                raise ValueError("invalid event field type " + key)


class Subscription:
    def __init__(self, owner: Owner, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("queue size must be positive")
        self.owner = owner
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(size)
        self.closed = False
        self.dropped = 0

    def poll(self, timeout: float = 0) -> dict[str, Any] | None:
        if self.closed:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.closed = True
        self.clear()


class EventBus:
    def __init__(self, authority: Authority) -> None:
        self.authority = authority
        self._lock = threading.RLock()
        self._responses: dict[str, tuple[str, str, Any]] = {}
        self._schemas: dict[tuple[str, str], EventSpec] = {}
        self._subscriptions: dict[str, Subscription] = {}

    def schemas(self) -> list[dict[str, Any]]:
        """Объявленные события модулей (7.6): тип, версия, поля и их типы."""
        with self._lock:
            items = sorted(self._schemas.items())
        return [
            {
                "module_id": module,
                "name": name,
                "type": f"module.{module}.{name}",
                "version": EVENT_VERSION,
                "required": {k: v.__name__ for k, v in spec.required.items()},
                "optional": {k: v.__name__ for k, v in (spec.optional or {}).items()},
            }
            for (module, name), spec in items
        ]

    def register_schema(self, module: str, name: str, schema: EventSpec) -> None:
        from ramvi_service.contracts.modules import module_id

        module_id(module)
        with self._lock:
            self._schemas[module, name] = copy.deepcopy(schema)

    def begin(self, owner: Owner, request_id: str | None = None) -> str:
        self.authority.require(owner)
        from ramvi_engine.protocol import EventContext, EventFactory

        response = uuid.uuid4().hex
        context = EventContext(
            request_id or uuid.uuid4().hex,
            module_id=owner.module_id,
            session_id=owner.session_id,
            response_id=response,
        )
        with self._lock:
            for subscription in self._subscriptions.values():
                if subscription.owner == owner:
                    subscription.clear()
            self._responses[owner.token] = (response, context.request_id, EventFactory(context))
        return response

    def subscribe(self, owner: Owner, *, size: int = 32) -> str:
        self.authority.require(owner)
        identifier = uuid.uuid4().hex
        with self._lock:
            self._subscriptions[identifier] = Subscription(owner, size)
        return identifier

    def emit(self, owner: Owner, response: str, name: str, payload: dict[str, Any]) -> bool:
        self.authority.require(owner)
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_." for c in name):
            raise ValueError("invalid event type")
        with self._lock:
            current = self._responses.get(owner.token)
            if current is None or current[0] != response:
                return False
            schema = self._schemas.get((owner.module_id, name))
            if schema is None:
                raise ValueError("event schema is not registered by the Host")
            schema.validate(payload)
            event = current[2].emit(f"module.{owner.module_id}.{name}", payload)
            for subscription in self._subscriptions.values():
                if subscription.closed or subscription.owner != owner:
                    continue
                try:
                    subscription.queue.put_nowait(copy.deepcopy(event))
                except queue.Full:
                    subscription.dropped += 1
            return True

    def poll(self, owner: Owner, subscription_id: str, timeout: float = 0) -> dict[str, Any] | None:
        self.authority.require(owner)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 <= timeout <= 1:
            raise ValueError("invalid subscription timeout")
        with self._lock:
            subscription = self._subscriptions[subscription_id]
            if subscription.owner != owner:
                raise PermissionError("not your subscription")
        # Новое begin/cancel аннулирует и уже поставленные в очередь старые события.
        end = None
        while True:
            event = subscription.poll(timeout if end is None else 0)
            end = True
            if event is None:
                return None
            with self._lock:
                current = self._responses.get(owner.token)
                if current is not None and event["response_id"] == current[0]:
                    return event

    def cancel(self, owner: Owner, response: str | None = None) -> None:
        with self._lock:
            current = self._responses.get(owner.token)
            if current is not None and (response is None or current[0] == response):
                self._responses.pop(owner.token, None)
                for subscription in self._subscriptions.values():
                    if subscription.owner == owner:
                        subscription.clear()

    def unsubscribe(self, owner: Owner, identifier: str) -> None:
        with self._lock:
            subscription = self._subscriptions.get(identifier)
            if subscription is not None and subscription.owner != owner:
                raise PermissionError("not your subscription")
            if subscription is not None:
                subscription.close()
                self._subscriptions.pop(identifier)

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            self.cancel(owner)
            for identifier in [k for k, v in self._subscriptions.items() if v.owner == owner]:
                self.unsubscribe(owner, identifier)

    def unregister_module(self, module: str) -> None:
        with self._lock:
            self._schemas = {key: value for key, value in self._schemas.items() if key[0] != module}
