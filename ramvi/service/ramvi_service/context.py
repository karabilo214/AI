"""Контекст выбранного приложения: независимые read/action/egress гранты."""

import copy
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ramvi_service.contracts.modules import Authority, Owner, ProviderCapabilities


@dataclass(frozen=True)
class ContextSnapshot:
    snapshot_id: str
    application: str
    window_id: str
    monotonic_ns: int
    source: str
    text: str | None = None
    image: str | None = None
    structure: dict[str, Any] | None = None


class ContextAdapter(Protocol):
    def capture(
        self, application: str, window: str, kind: str, cancel: threading.Event
    ) -> dict[str, Any]: ...


class GrantStore:
    def __init__(self, authority: Authority) -> None:
        self.authority = authority
        self._lock = threading.RLock()
        self._grants: set[tuple[str, str, str]] = set()
        self._watches: dict[tuple[str, str, str], set[threading.Event]] = {}

    def allow(self, owner: Owner, permission: str, application: str = "*") -> None:
        self.authority.require(owner)
        if not permission or not application:
            raise ValueError("нужны permission/application")
        with self._lock:
            self._grants.add((owner.token, permission, application))

    def revoke(self, owner: Owner, permission: str, application: str = "*") -> None:
        with self._lock:
            key = (owner.token, permission, application)
            self._grants.discard(key)
            for cancel in self._watches.get(key, set()):
                cancel.set()

    def require(self, owner: Owner, permission: str, application: str = "*") -> None:
        self.authority.require(owner)
        with self._lock:
            if (owner.token, permission, application) not in self._grants:
                raise PermissionError(f"нет разрешения {permission} для {application}")

    def watch(self, owner: Owner, rights: list[tuple[str, str]]) -> threading.Event:
        self.authority.require(owner)
        with self._lock:
            for permission, application in rights:
                self.require(owner, permission, application)
            cancel = threading.Event()
            for permission, application in rights:
                self._watches.setdefault((owner.token, permission, application), set()).add(cancel)
            return cancel

    def unwatch(self, cancel: threading.Event) -> None:
        with self._lock:
            for key in list(self._watches):
                self._watches[key].discard(cancel)
                if not self._watches[key]:
                    self._watches.pop(key)

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            for key in list(self._watches):
                if key[0] == owner.token:
                    for cancel in self._watches.pop(key):
                        cancel.set()
            self._grants = {row for row in self._grants if row[0] != owner.token}


class ContextProvider:
    def __init__(
        self,
        authority: Authority,
        grants: GrantStore,
        adapter: ContextAdapter,
        *,
        max_bytes: int = 2**20,
        min_interval: float = 0.1,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("неверный лимит контекста")
        if isinstance(min_interval, bool) or not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("неверный интервал контекста")
        self.authority, self.grants, self.adapter = authority, grants, adapter
        self.max_bytes, self.min_interval = max_bytes, min_interval
        self._last: dict[str, float] = {}
        self._snapshots: dict[str, tuple[str, ContextSnapshot]] = {}
        self._lock = threading.Lock()

    def capture(
        self,
        owner: Owner,
        application: str,
        window: str,
        *,
        kind: str = "ui",
        capabilities: ProviderCapabilities | None = None,
    ) -> ContextSnapshot:
        if kind not in ("ui", "screen") or not application or not window:
            raise ValueError("нужны kind, application и window")
        self.grants.require(owner, "context." + kind, application)
        if kind == "screen" and (capabilities is None or not capabilities.image_input):
            raise NotImplementedError("провайдер не поддерживает изображения")
        with self._lock:
            now = time.monotonic()
            if now - self._last.get(owner.token, float("-inf")) < self.min_interval:
                raise TimeoutError("лимит частоты контекста")
            self._last[owner.token] = now
        cancel = self.grants.watch(owner, [("context." + kind, application)])
        try:
            data = self.adapter.capture(application, window, kind, cancel)
            if cancel.is_set():
                raise PermissionError("сбор контекста отменён отзывом права")
        finally:
            self.grants.unwatch(cancel)
        # Грант мог быть отозван, пока OS adapter собирал данные.
        self.grants.require(owner, "context." + kind, application)
        import json

        if not isinstance(data, dict):
            raise ValueError("Context Adapter должен вернуть объект")
        for field in ("text", "image"):
            if data.get(field) is not None and not isinstance(data[field], str):
                raise ValueError(field + " должен быть строкой")
        if data.get("structure") is not None and not isinstance(data["structure"], dict):
            raise ValueError("structure должен быть объектом")
        if len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode()) > self.max_bytes:
            raise ValueError("контекст превышает бюджет")
        if set(data) - {"text", "image", "structure"}:
            raise ValueError("неизвестные поля Context Adapter")
        if kind == "ui" and data.get("image") is not None:
            raise PermissionError("UI Context не даёт доступа к Screen")
        snapshot = ContextSnapshot(
            uuid.uuid4().hex, application, window, time.monotonic_ns(), kind, **copy.deepcopy(data)
        )
        with self._lock:
            # Только последний snapshot владельца; нет бесконечного журнала экрана.
            self._snapshots[owner.token] = (snapshot.snapshot_id, copy.deepcopy(snapshot))
        return snapshot

    def authorize_egress(self, owner: Owner, snapshot: ContextSnapshot) -> dict[str, Any]:
        """Повторно проверить read + независимое право внешней отправки."""
        with self._lock:
            original = self._snapshots.get(owner.token)
            if original is None or original[0] != snapshot.snapshot_id or original[1] != snapshot:
                raise PermissionError("snapshot не принадлежит владельцу или изменён")
        self.grants.require(owner, "context." + snapshot.source, snapshot.application)
        if snapshot.text is not None or snapshot.structure is not None:
            self.grants.require(owner, "egress.text", snapshot.application)
        if snapshot.image is not None:
            self.grants.require(owner, "egress.image", snapshot.application)
        return asdict(snapshot)

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            self._snapshots.pop(owner.token, None)
            self._last.pop(owner.token, None)

    def transmit(
        self,
        owner: Owner,
        snapshot: ContextSnapshot,
        sender: Callable[[dict[str, Any], threading.Event], Any],
    ) -> Any:
        """Передача выполняется адаптером с немедленным сигналом отзыва прав.

        Уже отправленные байты вернуть нельзя; sender обязан прекращать новые
        отправки при cancel. Core не запускает скрытую облачную передачу.
        """
        data = self.authorize_egress(owner, snapshot)
        rights = [("context." + snapshot.source, snapshot.application)]
        if snapshot.text is not None or snapshot.structure is not None:
            rights.append(("egress.text", snapshot.application))
        if snapshot.image is not None:
            rights.append(("egress.image", snapshot.application))
        cancel = self.grants.watch(owner, rights)
        try:
            result = sender(data, cancel)
            if cancel.is_set():
                raise PermissionError("передача отменена отзывом права")
            self.authorize_egress(owner, snapshot)
            return result
        finally:
            self.grants.unwatch(cancel)
