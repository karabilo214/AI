"""Пути, настройки, секреты и миграции ограничены namespace модуля."""

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ramvi_service.contracts.modules import module_id


class ScopedData:
    def __init__(self, root: Path, module: str) -> None:
        self.base = root.resolve()
        self.module_id = module_id(module)
        candidate = self.base / self.module_id
        if candidate.is_symlink():
            raise PermissionError("namespace не может быть symlink")
        candidate.mkdir(parents=True, exist_ok=True)
        self.root = candidate.resolve()
        if not self.root.is_relative_to(self.base):
            raise PermissionError("namespace вне modules root")
        self._lock = threading.RLock()

    def path(self, relative: str) -> Path:
        value = Path(relative)
        if (
            not relative
            or relative == "."
            or value.is_absolute()
            or any(part in ("..", ".") for part in value.parts)
        ):
            raise PermissionError("нужен относительный путь внутри namespace")
        # Windows separators/drive syntax отклоняются также на POSIX.
        if "\\" in relative or ":" in relative or relative.startswith("/"):
            raise PermissionError("неверный scoped path")
        candidate = self.root / value
        for parent in (candidate, *candidate.parents):
            if parent == self.base:
                break
            if parent.is_symlink():
                raise PermissionError("symlink не разрешён в scoped data")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("путь вне namespace")
        return resolved

    def read_json(self, relative: str, default: Any = None) -> Any:
        with self._lock:
            path = self.path(relative)
            if not path.exists():
                return default
            if path.stat().st_size > 2**20:
                raise ValueError("scoped JSON превышает лимит")
            return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, relative: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 2**20:
            raise ValueError("scoped JSON превышает лимит")
        with self._lock:
            path = self.path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)

    @property
    def schema_version(self) -> int:
        value = self.read_json("schema.json", {"version": 1})
        if not isinstance(value, dict):
            raise ValueError("неверная форма версии данных")
        version = value.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("неверная версия данных модуля")
        return version

    def migrate(self, target: int, migrations: dict[int, Callable[["ScopedData"], None]]) -> None:
        """Миграция n -> n+1 в staging; атомарная замена только этого namespace."""
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError("неверная целевая версия данных")
        with self._lock:
            current = self.schema_version
            if target < current:
                raise ValueError("понижение схемы требует явного rollback backup")
            if current == target:
                return
            if any(path.is_symlink() for path in self.root.rglob("*")):
                raise PermissionError("миграция namespace с symlink запрещена")
            with tempfile.TemporaryDirectory(prefix=".migration-", dir=self.base) as folder:
                stage = ScopedData(Path(folder), self.module_id)
                shutil.copytree(self.root, stage.root, dirs_exist_ok=True)
                for version in range(current, target):
                    if version not in migrations:
                        raise ValueError(f"нет миграции {version} -> {version + 1}")
                    migrations[version](stage)
                    stage.write_json("schema.json", {"version": version + 1})
                backup = Path(folder) / "previous"
                os.replace(self.root, backup)
                try:
                    os.replace(stage.root, self.root)
                except BaseException:
                    os.replace(backup, self.root)
                    raise

    def reset(self) -> None:
        with self._lock:
            for child in self.root.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)


class SecretBackend(Protocol):
    """Backend будет OS credential store; Core не пишет секреты в обычный JSON."""

    def get(self, namespace: str, key: str) -> str | None: ...
    def set(self, namespace: str, key: str, value: str) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...


class MemorySecrets:
    """Непостоянный backend для portable bootstrap/тестов, не зашифрованное хранилище."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    def get(self, namespace: str, key: str) -> str | None:
        with self._lock:
            return self._values.get((namespace, key))

    def set(self, namespace: str, key: str, value: str) -> None:
        with self._lock:
            self._values[namespace, key] = value

    def delete(self, namespace: str, key: str) -> None:
        with self._lock:
            self._values.pop((namespace, key), None)


class ScopedSecrets:
    def __init__(self, backend: SecretBackend, module: str) -> None:
        self.backend, self.namespace = backend, module_id(module)

    def get(self, key: str) -> str | None:
        return self.backend.get(self.namespace, module_id(key))

    def set(self, key: str, value: str) -> None:
        if not isinstance(value, str) or len(value.encode()) > 16384:
            raise ValueError("secret должен быть строкой не длиннее 16 KiB")
        self.backend.set(self.namespace, module_id(key), value)

    def delete(self, key: str) -> None:
        self.backend.delete(self.namespace, module_id(key))
