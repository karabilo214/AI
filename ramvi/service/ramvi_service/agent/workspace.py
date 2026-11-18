"""6.3 (AG-03): границы доступа агента — рабочая папка как явный grant.

Агент видит только папку, которую выбрал пользователь. Путь от модели всегда
относительный; каждый путь проходит одну проверку `Workspace.resolve`:

* абсолютные пути, диски, `..`, имена из одних точек/пробелов, `:` и имена устройств
  Windows отклоняются до обращения к диску;
* ни один компонент пути не может быть symlink, junction или иной точкой повторного
  разбора — даже если он указывает внутрь папки; итоговый реальный путь обязан лежать
  внутри корня;
* файл с несколькими жёсткими ссылками отклоняется: его содержимое может
  принадлежать файлу вне папки;
* секреты (`.env`, ключи, `.ssh` …) недоступны ни на чтение, ни на запись, на любой
  глубине; запись внутрь `.git` запрещена (хуки = исполнение кода);
* права read / write / terminal / network раздельные; `revoke()` закрывает все новые
  обращения сразу.

Ошибки — `AccessDenied` с ключом сообщения для интерфейса, как в чате (3.14).
Проверка и открытие файла — разные системные вызовы; `open_read` сверяет открытый
файл с проверенным (устройство и inode), чтобы подмену между ними не прочитать.
"""

import fnmatch
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO, Literal

Permission = Literal["read", "write", "terminal", "network"]
PERMISSIONS: tuple[Permission, ...] = ("read", "write", "terminal", "network")

# Имена (на любой глубине, без учёта регистра); совпавшая папка закрывает всё внутри.
DEFAULT_SECRETS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    "_netrc",
    ".pypirc",
    ".npmrc",
    ".git-credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    ".ssh",
    ".aws",
    ".gnupg",
    ".docker",
    ".kube",
)
WINDOWS_DEVICES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)
MAX_PATH_CHARS = 4096
_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT


class AccessDenied(PermissionError):
    """Отказ в доступе с ключом сообщения (`agent.access.*`) и параметрами."""

    def __init__(self, code: str, **params: Any) -> None:
        super().__init__(code)
        self.code, self.params = code, params


@dataclass(frozen=True)
class Permissions:
    """Раздельные права grant. По умолчанию — только чтение."""

    read: bool = True
    write: bool = False
    terminal: bool = False
    network: bool = False

    def allows(self, permission: Permission) -> bool:
        if permission not in PERMISSIONS:
            raise ValueError(f"unknown permission: {permission}")
        return bool(getattr(self, permission))


def _is_link(info: os.stat_result, path: str) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    if getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
        return True
    return os.path.isjunction(path)


def _within(path: str, root: str) -> bool:
    path, root = os.path.normcase(path), os.path.normcase(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # разные диски Windows
        return False


class Workspace:
    """Рабочая папка агента: единственная точка проверки путей и прав."""

    def __init__(
        self,
        root: str | Path,
        permissions: Permissions | None = None,
        *,
        secret_patterns: tuple[str, ...] = DEFAULT_SECRETS,
    ) -> None:
        given = Path(root)
        if not given.is_absolute():
            raise AccessDenied("agent.access.root_relative")
        try:
            real = os.path.realpath(given, strict=True)
        except OSError:
            raise AccessDenied("agent.access.root_missing") from None
        if not os.path.isdir(real):
            raise AccessDenied("agent.access.root_not_directory")
        if Path(real) == Path(Path(real).anchor):
            raise AccessDenied("agent.access.root_is_drive")
        self.root = Path(real)
        self.permissions = permissions or Permissions()
        self._secrets = tuple(p.lower() for p in secret_patterns)
        self._revoked = threading.Event()

    # --- права ------------------------------------------------------------------------

    @property
    def revoked(self) -> bool:
        return self._revoked.is_set()

    def revoke(self) -> None:
        """Отозвать grant: все следующие проверки отказывают."""
        self._revoked.set()

    def require(self, permission: Permission) -> None:
        if self._revoked.is_set():
            raise AccessDenied("agent.access.revoked")
        if not self.permissions.allows(permission):
            raise AccessDenied("agent.access.permission", permission=permission)

    def is_secret(self, name: str) -> bool:
        """Имя одного компонента пути попадает под исключения для секретов."""
        name = name.rstrip(" .").lower()
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self._secrets)

    # --- пути -------------------------------------------------------------------------

    def _parts(self, path: str) -> list[str]:
        if not isinstance(path, str) or len(path) > MAX_PATH_CHARS or "\0" in path:
            raise AccessDenied("agent.access.invalid")
        if path.startswith(("/", "\\")) or PureWindowsPath(path).drive:
            raise AccessDenied("agent.access.absolute")
        parts: list[str] = []
        for part in path.replace("\\", "/").split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                raise AccessDenied("agent.access.parent")
            if not part.rstrip(" ."):
                raise AccessDenied("agent.access.invalid")
            if ":" in part or part.split(".")[0].rstrip(" ").lower() in WINDOWS_DEVICES:
                raise AccessDenied("agent.access.invalid")
            parts.append(part)
        return parts

    def resolve(
        self, path: str, *, permission: Permission = "read", must_exist: bool = True
    ) -> Path:
        """Проверить относительный путь модели и вернуть абсолютный путь внутри папки.

        `must_exist=False` — для создания: недостающий хвост пути разрешён, проверяется
        существующая часть. Права проверяются при каждом вызове (отзыв действует сразу).
        """
        self.require(permission)
        parts = self._parts(path)
        for part in parts:
            if self.is_secret(part):
                raise AccessDenied("agent.access.secret", path=path)
        if permission == "write" and any(p.rstrip(" .").lower() == ".git" for p in parts):
            raise AccessDenied("agent.access.git_write", path=path)
        existing = str(self.root)  # самая длинная проверенная существующая часть
        info: os.stat_result | None = os.lstat(existing)
        missing: list[str] = []
        for index, part in enumerate(parts):
            assert info is not None
            if not stat.S_ISDIR(info.st_mode):
                raise AccessDenied("agent.access.not_directory", path=path)
            candidate = os.path.join(existing, part)
            try:
                info = os.lstat(candidate)
            except FileNotFoundError:
                if must_exist:
                    raise AccessDenied("agent.access.not_found", path=path) from None
                missing, info = parts[index:], None
                break
            except OSError:
                raise AccessDenied("agent.access.invalid", path=path) from None
            if _is_link(info, candidate):
                raise AccessDenied("agent.access.symlink", path=path)
            existing = candidate
        current = os.path.join(existing, *missing)
        if not _within(os.path.realpath(existing), str(self.root)):
            raise AccessDenied("agent.access.outside", path=path)
        if info is not None and stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise AccessDenied("agent.access.hardlink", path=path)
        return Path(current)

    def relative(self, path: Path) -> str:
        """Путь внутри папки для журнала и интерфейса (через `/`, без корня)."""
        rel = path.relative_to(self.root).as_posix()
        return "." if rel in ("", ".") else rel

    def open_read(self, path: str) -> BinaryIO:
        """Открыть обычный файл на чтение; подмена после проверки не читается."""
        target = self.resolve(path)
        checked = os.lstat(target)
        if not stat.S_ISREG(checked.st_mode):
            raise AccessDenied("agent.access.not_file", path=path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(target, flags)
        except OSError:
            raise AccessDenied("agent.access.invalid", path=path) from None
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (checked.st_dev, checked.st_ino) or (
            opened.st_nlink > 1
        ):
            os.close(fd)
            raise AccessDenied("agent.access.changed", path=path)
        return os.fdopen(fd, "rb")
