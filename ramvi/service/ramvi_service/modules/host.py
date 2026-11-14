"""Модульный Host: общий жизненный цикл без веток по Character/Chat/Agent."""

import copy
import math
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, Manifest, Owner
from ramvi_service.events import EventBus
from ramvi_service.modules.controller import ProcessModuleWorker, WorkerController
from ramvi_service.resources import ResourceBroker, SharedLLM
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets, SecretBackend
from ramvi_service.services import ScopedServices, ServiceGateway


@dataclass
class Entry:
    manifest: Manifest
    approved: tuple[str, ...]
    state: str = "registered"
    error: str | None = None
    owner: Owner | None = None
    services: ScopedServices | None = None
    worker: WorkerController | None = None
    application_grants: set[tuple[str, str]] = field(default_factory=set)
    lock: Any = field(default_factory=threading.RLock, repr=False)


class ModuleHost:
    def __init__(
        self,
        root: Path,
        *,
        principal_id: str = "local_user",
        timeout: float = 10,
        capacity: dict[str, int] | None = None,
        llm: SharedLLM | None = None,
        authority: Authority | None = None,
        grants: GrantStore | None = None,
        context: ContextProvider | None = None,
        secrets: SecretBackend | None = None,
        controller: Callable[[Manifest], WorkerController] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("timeout должен быть положительным")
        self.root, self.principal_id, self.timeout = (
            root.resolve() / "modules",
            principal_id,
            timeout,
        )
        if self.root.is_symlink():
            raise PermissionError("modules root не может быть symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = authority or (
            llm.authority
            if llm
            else grants.authority
            if grants
            else context.authority
            if context
            else Authority()
        )
        for service in (llm, grants, context):
            if service is not None and service.authority is not self.authority:
                raise ValueError("сервисы используют разных Authority")
        self.grants = grants or (context.grants if context else GrantStore(self.authority))
        self.resources = ResourceBroker(self.authority, capacity or {"cpu_slots": 1})
        self.events = EventBus(self.authority)
        self.llm, self.context = llm, context
        self.secrets = secrets or MemorySecrets()
        self.gateway = ServiceGateway(self.authority)
        self.controller = controller or (
            lambda m: ProcessModuleWorker(m.entrypoint, timeout=timeout)
        )
        self._entries: dict[str, Entry] = {}
        self._migrations: dict[str, dict[int, Callable[[ScopedData], None]]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        manifest: Manifest,
        *,
        approved_permissions: tuple[str, ...] = (),
        migrations: dict[int, Callable[[ScopedData], None]] | None = None,
    ) -> None:
        with self._lock:
            if manifest.module_id in self._entries:
                raise ValueError("module_id уже зарегистрирован")
            if set(approved_permissions) - set(manifest.permissions):
                raise ValueError("одобрение не запрошенного разрешения")
            self._entries[manifest.module_id] = Entry(copy.deepcopy(manifest), approved_permissions)
            self._migrations[manifest.module_id] = dict(migrations or {})
            try:
                self._check_graph()
            except Exception:
                self._entries.pop(manifest.module_id)
                self._migrations.pop(manifest.module_id)
                raise

    def _check_graph(self) -> None:
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visiting:
                raise ValueError("цикл обязательных зависимостей")
            if identifier in done or identifier not in self._entries:
                return
            visiting.add(identifier)
            for dependency in self._entries[identifier].manifest.required:
                visit(dependency)
            visiting.remove(identifier)
            done.add(identifier)

        for identifier in self._entries:
            visit(identifier)

    def _entry(self, identifier: str) -> Entry:
        with self._lock:
            if identifier not in self._entries:
                raise KeyError("модуль не установлен")
            return self._entries[identifier]

    def status(self, identifier: str | None = None) -> dict[str, Any]:
        with self._lock:
            entries = [self._entry(identifier)] if identifier else list(self._entries.values())
        result: dict[str, Any] = {}
        for entry in entries:
            with entry.lock:
                worker = entry.worker.status() if entry.worker else None
                if worker is not None and worker["state"] != "running":
                    detail = worker.get("error") or "module worker завершился"
                    self._stop(entry)
                    entry.state, entry.error = "error", str(detail)
                    worker = None
                result[entry.manifest.module_id] = {
                    "state": entry.state,
                    "error": entry.error,
                    "version": entry.manifest.version,
                    "api_version": entry.manifest.api_version,
                    "worker": worker,
                }
        return result

    def _release(self, entry: Entry) -> None:
        try:
            if entry.services:
                entry.services.stop()
        finally:
            if entry.owner:
                self.gateway.unbind(entry.owner)
                self.grants.release_owner(entry.owner)
                self.authority.revoke(entry.owner)
            entry.services, entry.owner = None, None

    def _stop(self, entry: Entry) -> None:
        try:
            if entry.worker:
                entry.worker.stop()
        finally:
            entry.worker = None
            self._release(entry)

    def _call(
        self, entry: Entry, method: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if entry.worker is None:
            raise RuntimeError("module worker не создан")
        try:
            return entry.worker.call(method, payload or {}, self.timeout)
        except Exception as error:
            entry.state, entry.error = "error", f"{type(error).__name__}: {error}"
            self._stop(entry)
            raise

    def _start(self, entry: Entry) -> None:
        if set(entry.manifest.permissions) - set(entry.approved):
            raise PermissionError("разрешения модуля не одобрены")
        available = {"settings", "data", "secrets", "resources", "events"}
        if self.llm:
            available.add("llm")
        if self.context:
            available.add("context")
        missing = set(entry.manifest.capabilities) - available
        if missing:
            raise NotImplementedError("недоступные сервисы: " + ", ".join(sorted(missing)))
        for dependency in entry.manifest.required:
            if dependency not in self._entries or self._entries[dependency].state == "disposed":
                raise RuntimeError("не установлена зависимость " + dependency)
        for dependency, version in entry.manifest.dependency_versions.items():
            installed = self._entries.get(dependency)
            if installed is not None and installed.manifest.version != version:
                raise RuntimeError("несовместимая версия зависимости " + dependency)
        data = ScopedData(self.root, entry.manifest.module_id)
        data.migrate(entry.manifest.data_schema, self._migrations[entry.manifest.module_id])
        owner = self.authority.issue(entry.manifest.module_id, self.principal_id, uuid.uuid4().hex)
        scope = ScopedServices(
            owner,
            self.authority,
            self.grants,
            data,
            ScopedSecrets(self.secrets, entry.manifest.module_id),
            self.resources,
            self.events,
            self.llm,
            self.context,
            timeout=self.timeout,
        )
        entry.owner, entry.services = owner, scope
        for permission in set(entry.approved) & set(entry.manifest.permissions):
            self.grants.allow(owner, permission)
        for permission, application in entry.application_grants:
            if permission in entry.manifest.permissions and permission in entry.approved:
                self.grants.allow(owner, permission, application)
        self.resources.quota(owner, entry.manifest.resource_limits)
        self.gateway.bind(scope)
        try:
            url = self.gateway.start()
            entry.worker = self.controller(entry.manifest)
            self._call(
                entry,
                "initialize",
                {
                    "context": {
                        "module_id": entry.manifest.module_id,
                        "session_id": owner.session_id,
                        "data_schema": entry.manifest.data_schema,
                        "checkpoint": data.read_json("checkpoint.json", {}),
                        "services": {"url": url, "token": owner.token},
                    }
                },
            )
            entry.state, entry.error = "initialized", None
        except Exception:
            self._stop(entry)
            raise

    def approve_application(self, identifier: str, permission: str, application: str) -> None:
        """Только доверенный UI/Host, не RPC модуля: отдельный выбор приложения."""
        entry = self._entry(identifier)
        if permission not in {
            "context.ui",
            "context.screen",
            "egress.text",
            "egress.image",
            "agent.control",
        }:
            raise ValueError("неверное право приложения")
        with entry.lock:
            if permission not in entry.approved or permission not in entry.manifest.permissions:
                raise PermissionError("разрешение не запрошено/не одобрено")
            entry.application_grants.add((permission, application))
            if entry.owner is not None:
                self.grants.allow(entry.owner, permission, application)

    def revoke_application(self, identifier: str, permission: str, application: str) -> None:
        entry = self._entry(identifier)
        with entry.lock:
            entry.application_grants.discard((permission, application))
            if entry.owner is not None:
                self.grants.revoke(entry.owner, permission, application)

    def initialize(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state in {"initialized", "active", "suspended"} and entry.worker is not None:
                return self.status(identifier)
            if entry.state == "disposed":
                raise RuntimeError("модуль disposed")
            try:
                self._start(entry)
            except Exception as error:
                entry.state, entry.error = "error", f"{type(error).__name__}: {error}"
                raise
            return self.status(identifier)

    def activate(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state == "active":
                return self.status(identifier)
            if entry.state == "suspended":
                raise RuntimeError("используйте resume")
            for dependency in entry.manifest.required:
                self.activate(dependency)
            self.initialize(identifier)
            assert entry.services is not None
            entry.services.active = True
            self._call(entry, "activate")
            entry.state = "active"
            return self.status(identifier)

    def _checkpoint(self, entry: Entry) -> None:
        if entry.worker is not None and entry.services is not None:
            value = self._call(entry, "checkpoint")
            entry.services.data.write_json("checkpoint.json", value)

    def deactivate(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state in {"registered", "inactive", "disposed"}:
                return self.status(identifier)
            try:
                if entry.services:
                    entry.services.active = False
                self._checkpoint(entry)
                if entry.worker:
                    self._call(entry, "deactivate")
            finally:
                self._stop(entry)
            if entry.state != "error":
                entry.state = "inactive"
            return self.status(identifier)

    def suspend(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state == "suspended":
                return self.status(identifier)
            if entry.state != "active":
                raise RuntimeError("suspend требует active")
            assert entry.services is not None
            entry.services.active = False
            self._checkpoint(entry)
            self._call(entry, "suspend")
            self._stop(entry)
            entry.state = "suspended"
            return self.status(identifier)

    def resume(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state == "active":
                return self.status(identifier)
            if entry.state != "suspended":
                raise RuntimeError("resume требует suspended")
            self._start(entry)
            assert entry.services is not None
            entry.services.active = True
            self._call(entry, "resume")
            entry.state = "active"
            return self.status(identifier)

    def invoke(
        self, identifier: str, method: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            if entry.state != "active":
                raise RuntimeError("module invoke требует active")
            return self._call(entry, "invoke", {"method": method, "arguments": arguments or {}})

    def dispose(self, identifier: str) -> dict[str, Any]:
        entry = self._entry(identifier)
        with entry.lock:
            try:
                if entry.worker:
                    self._checkpoint(entry)
                    self._call(entry, "dispose")
            finally:
                self._stop(entry)
                entry.state = "disposed"
            return self.status(identifier)

    def update(
        self,
        identifier: str,
        manifest: Manifest,
        *,
        migrations: dict[int, Callable[[ScopedData], None]] | None = None,
    ) -> dict[str, Any]:
        entry = self._entry(identifier)
        if manifest.module_id != identifier:
            raise ValueError("update не может менять module_id")
        with entry.lock:
            was_active, was_suspended = entry.state == "active", entry.state == "suspended"
            previous, old_migrations = entry.manifest, self._migrations[identifier]
            self.deactivate(identifier)
            data = ScopedData(self.root, identifier)
            if any(p.is_symlink() for p in data.root.rglob("*")):
                raise PermissionError("update namespace с symlink запрещён")
            with tempfile.TemporaryDirectory(prefix=".update-", dir=self.root) as temporary:
                backup = Path(temporary) / "backup"
                shutil.copytree(data.root, backup)
                try:
                    entry.manifest = copy.deepcopy(manifest)
                    self._migrations[identifier] = dict(migrations or {})
                    self._check_graph()
                    self.initialize(identifier)
                    if was_active:
                        self.activate(identifier)
                    elif was_suspended:
                        self.activate(identifier)
                        self.suspend(identifier)
                    else:
                        self.deactivate(identifier)
                except Exception:
                    self._stop(entry)
                    shutil.rmtree(data.root)
                    shutil.copytree(backup, data.root)
                    entry.manifest = previous
                    self._migrations[identifier] = old_migrations
                    entry.state, entry.error = "inactive", "обновление откатилось"
                    if was_active:
                        self.activate(identifier)
                    elif was_suspended:
                        self.activate(identifier)
                        self.suspend(identifier)
                    raise
            return self.status(identifier)

    def uninstall(self, identifier: str, *, purge_data: bool = False) -> None:
        self.dispose(identifier)
        with self._lock:
            self._entries.pop(identifier)
            self._migrations.pop(identifier)
        self.events.unregister_module(identifier)
        if purge_data:
            ScopedData(self.root, identifier).reset()

    def close(self) -> None:
        errors = []
        for identifier in list(self._entries):
            try:
                self.dispose(identifier)
            except Exception as error:
                errors.append(str(error))
        self.gateway.close()
        if errors:
            raise RuntimeError("; ".join(errors))
