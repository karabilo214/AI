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
        self._epochs: dict[tuple[str, str, str], int] = {}
        self._next_epoch = 1  # не переиспользуется даже после release_owner
        self._watches: dict[tuple[str, str, str], set[threading.Event]] = {}

    def allow(self, owner: Owner, permission: str, application: str = "*") -> None:
        self.authority.require(owner)
        if not permission or not application:
            raise ValueError("permission and application are required")
        with self._lock:
            key = (owner.token, permission, application)
            if key not in self._grants:
                self._epochs[key] = self._next_epoch
                self._next_epoch += 1
            self._grants.add(key)

    def revoke(self, owner: Owner, permission: str, application: str = "*") -> None:
        with self._lock:
            key = (owner.token, permission, application)
            self._grants.discard(key)
            self._epochs.pop(key, None)
            for cancel in self._watches.get(key, set()):
                cancel.set()

    def require(self, owner: Owner, permission: str, application: str = "*") -> None:
        self.authority.require(owner)
        with self._lock:
            if (owner.token, permission, application) not in self._grants:
                raise PermissionError(f"no permission {permission} for {application}")

    def epoch(self, owner: Owner, permission: str, application: str) -> int:
        with self._lock:
            self.require(owner, permission, application)
            return self._epochs.get((owner.token, permission, application), 0)

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
            self._epochs = {
                key: value for key, value in self._epochs.items() if key[0] != owner.token
            }


class ContextProvider:
    def __init__(
        self,
        authority: Authority,
        grants: GrantStore,
        adapter: ContextAdapter,
        *,
        max_bytes: int = 2**20,
        min_interval: float = 0.1,
        max_bytes_per_minute: int | None = 4 * 2**20,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("invalid context limit")
        if isinstance(min_interval, bool) or not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("invalid context interval")
        if max_bytes_per_minute is not None and (
            isinstance(max_bytes_per_minute, bool)
            or not isinstance(max_bytes_per_minute, int)
            or max_bytes_per_minute < 1
        ):
            raise ValueError("invalid context budget")
        self.authority, self.grants, self.adapter = authority, grants, adapter
        self.max_bytes, self.min_interval = max_bytes, min_interval
        self.max_bytes_per_minute = max_bytes_per_minute
        self._budget: dict[str, tuple[float, int]] = {}
        self._last: dict[str, float] = {}
        self._snapshots: dict[str, tuple[str, ContextSnapshot, int]] = {}
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
            raise ValueError("kind, application and window are required")
        epoch = self.grants.epoch(owner, "context." + kind, application)
        if kind == "screen" and (capabilities is None or not capabilities.image_input):
            raise NotImplementedError("the provider does not support images")
        with self._lock:
            now = time.monotonic()
            if now - self._last.get(owner.token, float("-inf")) < self.min_interval:
                raise TimeoutError("context rate limit")
            start, used = self._budget.get(owner.token, (now, 0))
            if now - start >= 60:
                start, used = now, 0
            if self.max_bytes_per_minute is not None and used >= self.max_bytes_per_minute:
                raise TimeoutError("context byte budget exhausted")
            self._budget[owner.token] = (start, used)
            self._last[owner.token] = now
        cancel = self.grants.watch(owner, [("context." + kind, application)])
        try:
            data = self.adapter.capture(application, window, kind, cancel)
            if cancel.is_set():
                raise PermissionError("context collection cancelled: permission revoked")
        finally:
            self.grants.unwatch(cancel)
        # Грант мог быть отозван, пока OS adapter собирал данные.
        self.grants.require(owner, "context." + kind, application)
        import json

        if not isinstance(data, dict):
            raise ValueError("Context Adapter must return an object")
        for field in ("text", "image"):
            if data.get(field) is not None and not isinstance(data[field], str):
                raise ValueError(field + " must be a string")
        if data.get("structure") is not None and not isinstance(data["structure"], dict):
            raise ValueError("structure must be an object")
        size = len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode())
        if size > self.max_bytes:
            raise ValueError("context exceeds the budget")
        if set(data) - {"text", "image", "structure"}:
            raise ValueError("unknown Context Adapter fields")
        if kind == "ui" and data.get("image") is not None:
            raise PermissionError("UI Context does not grant Screen access")
        snapshot = ContextSnapshot(
            uuid.uuid4().hex, application, window, time.monotonic_ns(), kind, **copy.deepcopy(data)
        )
        with self._lock:
            # Только последний snapshot владельца; нет бесконечного журнала экрана.
            start, used = self._budget.get(owner.token, (time.monotonic(), 0))
            self._budget[owner.token] = (start, used + size)
            if self.max_bytes_per_minute is not None and used + size > self.max_bytes_per_minute:
                raise TimeoutError("context byte budget exhausted")
            if self.grants.epoch(owner, "context." + kind, application) != epoch:
                raise PermissionError("context permission was revoked")
            self._snapshots[owner.token] = (snapshot.snapshot_id, copy.deepcopy(snapshot), epoch)
        return snapshot

    def capture_best(
        self,
        owner: Owner,
        application: str,
        window: str,
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> ContextSnapshot:
        """Достаточный UI — без снимка. Если его нет, Screen запрашивается явно.

        Не делаем скрытую дорогую/облачную попытку. Отсутствие UI-гранта не заменяется
        автоматическим повышением до Screen; пользователь сам выбирает этот режим.
        """
        snapshot = self.capture(owner, application, window, kind="ui", capabilities=capabilities)
        if not snapshot.text and not snapshot.structure:
            raise NotImplementedError("no UI Context; request Screen explicitly if allowed")
        return snapshot

    def validate_snapshot(
        self, owner: Owner, snapshot: ContextSnapshot, *, max_age: float | None = None
    ) -> None:
        with self._lock:
            original = self._snapshots.get(owner.token)
            if original is None or original[0] != snapshot.snapshot_id or original[1] != snapshot:
                raise PermissionError("snapshot does not belong to the owner or was changed")
            epoch = original[2]
        if self.grants.epoch(owner, "context." + snapshot.source, snapshot.application) != epoch:
            raise PermissionError("snapshot permission was revoked")
        if max_age is not None and (time.monotonic_ns() - snapshot.monotonic_ns) / 1e9 > max_age:
            raise TimeoutError("snapshot is stale")

    def snapshot(self, owner: Owner, snapshot_id: str) -> ContextSnapshot:
        with self._lock:
            row = self._snapshots.get(owner.token)
            if row is None or row[0] != snapshot_id:
                raise PermissionError("unknown snapshot for this owner")
            snapshot = copy.deepcopy(row[1])
        self.validate_snapshot(owner, snapshot)
        return snapshot

    def release_snapshot(self, owner: Owner, snapshot_id: str) -> None:
        with self._lock:
            row = self._snapshots.get(owner.token)
            if row is not None and row[0] == snapshot_id:
                self._snapshots.pop(owner.token, None)

    def authorize_egress(self, owner: Owner, snapshot: ContextSnapshot) -> dict[str, Any]:
        """Повторно проверить read + независимое право внешней отправки."""
        self.validate_snapshot(owner, snapshot)
        if snapshot.text is not None or snapshot.structure is not None:
            self.grants.require(owner, "egress.text", snapshot.application)
        if snapshot.image is not None:
            self.grants.require(owner, "egress.image", snapshot.application)
        return asdict(snapshot)

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            self._snapshots.pop(owner.token, None)
            self._last.pop(owner.token, None)
            self._budget.pop(owner.token, None)

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
            self.validate_snapshot(owner, snapshot)
            if cancel.is_set():
                raise PermissionError("transfer cancelled: permission revoked")
            result = sender(data, cancel)
            if cancel.is_set():
                raise PermissionError("transfer cancelled: permission revoked")
            self.authorize_egress(owner, snapshot)
            return result
        finally:
            self.grants.unwatch(cancel)
