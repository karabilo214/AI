"""5.7: постоянное хранилище секретов (ключи сторонних API) на средствах ОС.

`SecretBackend` из `scoped.py`; в открытом виде секрет не пишется никуда:
- Windows — DPAPI (`CryptProtectData`, привязка к пользователю Windows); зашифрованные
  записи лежат в `secrets.dpapi` папки данных. На другом компьютере не расшифруются —
  ключ вводится заново (так и задумано);
- macOS — Связка ключей через `security -i`: команда идёт в stdin, секрет не виден в
  списке процессов;
- Linux — Secret Service (`secret-tool`, секрет через stdin).
Нет хранилища — `SecretsUnavailable` (сервис показывает понятную ошибку).
"""

import base64
import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

SERVICE = "RAMVI"
B64 = "b64:"  # метка значения в base64 (Связка ключей macOS)


class SecretsUnavailable(RuntimeError):
    """В системе нет хранилища секретов."""


Runner = Callable[[list[str], str | None], subprocess.CompletedProcess[str]]


def _run(args: list[str], stdin: str | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, input=stdin, capture_output=True, text=True, timeout=15, check=False
    )


def _quote(value: str) -> str:
    """Аргумент для командной строки `security -i` (кавычки и обратные косые экранируются)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MacKeychain:
    def __init__(self, run: Runner = _run) -> None:
        self.run = run

    def _account(self, namespace: str, key: str) -> str:
        return f"{namespace}/{key}"

    def get(self, namespace: str, key: str) -> str | None:
        out = self.run(
            [
                "security",
                "find-generic-password",
                "-s",
                SERVICE,
                "-a",
                self._account(namespace, key),
                "-w",
            ],
            None,
        )
        if out.returncode != 0:
            return None
        stored = out.stdout.rstrip("\n")
        # Значение хранится в base64: `security -w` печатает не-ASCII шестнадцатеричной строкой.
        if stored.startswith(B64):
            try:
                return base64.b64decode(stored[len(B64) :]).decode("utf-8")
            except ValueError:
                return None
        return stored

    def set(self, namespace: str, key: str, value: str) -> None:
        if "\n" in value:
            raise ValueError("secret cannot contain line breaks")
        sealed = B64 + base64.b64encode(value.encode("utf-8")).decode("ascii")
        command = (
            f"add-generic-password -U -s {_quote(SERVICE)} "
            f"-a {_quote(self._account(namespace, key))} -w {_quote(sealed)}\n"
        )
        out = self.run(["security", "-i"], command)
        if out.returncode != 0:
            raise SecretsUnavailable(out.stderr.strip()[:200] or "keychain write failed")

    def delete(self, namespace: str, key: str) -> None:
        self.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                SERVICE,
                "-a",
                self._account(namespace, key),
            ],
            None,
        )


class LinuxSecretService:
    def __init__(self, run: Runner = _run) -> None:
        self.run = run
        if shutil.which("secret-tool") is None:
            raise SecretsUnavailable("secret-tool not found")

    def _attrs(self, namespace: str, key: str) -> list[str]:
        return ["service", SERVICE, "account", f"{namespace}/{key}"]

    def get(self, namespace: str, key: str) -> str | None:
        out = self.run(["secret-tool", "lookup", *self._attrs(namespace, key)], None)
        return out.stdout if out.returncode == 0 and out.stdout else None

    def set(self, namespace: str, key: str, value: str) -> None:
        label = f"{SERVICE} {namespace}/{key}"
        out = self.run(
            ["secret-tool", "store", "--label", label, *self._attrs(namespace, key)], value
        )
        if out.returncode != 0:
            raise SecretsUnavailable(out.stderr.strip()[:200] or "secret service write failed")

    def delete(self, namespace: str, key: str) -> None:
        self.run(["secret-tool", "clear", *self._attrs(namespace, key)], None)


def _dpapi(data: bytes, protect: bool) -> bytes:  # pragma: no cover - только Windows
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined,unused-ignore]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    buffer = ctypes.create_string_buffer(data, len(data))
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    result = Blob()
    call = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    # CRYPTPROTECT_UI_FORBIDDEN = 0x1: без окон
    if not call(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(result)):
        raise SecretsUnavailable(f"DPAPI error {ctypes.GetLastError()}")  # type: ignore[attr-defined,unused-ignore]
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)


class WindowsDpapi:
    """Записи `{namespace/key: base64(DPAPI(value))}` в одном файле; запись — атомарная."""

    def __init__(
        self,
        folder: Path,
        protect: Callable[[bytes], bytes] = lambda d: _dpapi(d, True),
        unprotect: Callable[[bytes], bytes] = lambda d: _dpapi(d, False),
    ) -> None:
        self.path = Path(folder) / "secrets.dpapi"
        self.protect, self.unprotect = protect, unprotect
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _save(self, values: dict[str, str]) -> None:
        from ramvi_service.settings import write_json

        write_json(self.path.parent, self.path.name, values)

    def get(self, namespace: str, key: str) -> str | None:
        with self._lock:
            blob = self._load().get(f"{namespace}/{key}")
        if blob is None:
            return None
        try:
            return self.unprotect(base64.b64decode(blob)).decode("utf-8")
        except (SecretsUnavailable, ValueError):
            return None  # другой пользователь или компьютер: ключ вводится заново

    def set(self, namespace: str, key: str, value: str) -> None:
        sealed = base64.b64encode(self.protect(value.encode("utf-8"))).decode("ascii")
        with self._lock:
            values = self._load()
            values[f"{namespace}/{key}"] = sealed
            self._save(values)

    def delete(self, namespace: str, key: str) -> None:
        with self._lock:
            values = self._load()
            if values.pop(f"{namespace}/{key}", None) is not None:
                self._save(values)


def os_secrets(data_dir: Path, platform: str = sys.platform) -> Any:
    """Хранилище ОС; нет — `SecretsUnavailable` при первом обращении."""
    try:
        if platform == "win32":
            return WindowsDpapi(data_dir)
        if platform == "darwin" and shutil.which("security"):
            return MacKeychain()
        if platform.startswith("linux"):
            return LinuxSecretService()
    except SecretsUnavailable as error:
        return _Unavailable(str(error))
    return _Unavailable(f"no OS secret store on {platform}")


class _Unavailable:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def get(self, namespace: str, key: str) -> str | None:
        return None

    def set(self, namespace: str, key: str, value: str) -> None:
        raise SecretsUnavailable(self.reason)

    def delete(self, namespace: str, key: str) -> None:
        pass


__all__ = [
    "LinuxSecretService",
    "MacKeychain",
    "SecretsUnavailable",
    "WindowsDpapi",
    "os_secrets",
]
