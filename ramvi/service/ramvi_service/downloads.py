"""4.3 MOD-02/MOD-04: загрузка файлов моделей.

Пауза = остановка с сохранением `.part`; продолжение — HTTP Range с места
остановки (и после переадресации на CDN). Отмена = остановка и удаление `.part`.
Файл появляется под своим именем только после проверки SHA-256 (атомарный
os.replace). Имя файла не может выйти за папку моделей. Адреса — только HTTPS
(`ramvi_service.web`).

3.14: ошибка загрузки — ключ `download.*` и параметры (`DownloadError.code/params`).
"""

import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from ramvi_service import web
from ramvi_service.i18n import UserError

CHUNK = 1024 * 1024
SPACE_RESERVE = 512 * 1024 * 1024  # не заполнять диск до нуля


class DownloadError(UserError):
    pass


class Paused(DownloadError):
    pass


class Cancelled(DownloadError):
    pass


class Interrupted(DownloadError):
    """Соединение закрылось раньше конца файла; `.part` цел — можно продолжить."""


@dataclass(frozen=True)
class Progress:
    done: int
    total: int | None


def safe_target(models_dir: str | os.PathLike[str], name: str) -> Path:
    """Только относительный путь без `..`, абсолютных путей, дисков и ссылок наружу."""
    if not name or "\x00" in name:
        raise DownloadError("download.bad_name")
    for pure in (PurePosixPath(name), PureWindowsPath(name)):
        if pure.is_absolute() or pure.drive or ".." in pure.parts:
            raise DownloadError("download.bad_path", name=name)
    root = Path(models_dir).resolve()
    target = (root / name).resolve()
    if root not in target.parents:
        raise DownloadError("download.outside_models", name=name)
    return target


def _hash_file(path: Path) -> "hashlib._Hash":
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(CHUNK), b""):
            digest.update(block)
    return digest


class Download:
    def __init__(
        self,
        url: str,
        models_dir: str | os.PathLike[str],
        name: str,
        sha256: str,
        size: int | None = None,
        limit_bps: int | None = None,
        on_progress: Callable[[Progress], None] | None = None,
        timeout: float = 30.0,
    ):
        if not web.allowed(url):
            raise DownloadError("download.only_https")
        if len(sha256) != 64:
            raise DownloadError("download.need_sha256")
        self.url, self.sha256, self.size = url, sha256.lower(), size
        self.target = safe_target(models_dir, name)
        self.part = self.target.with_name(self.target.name + ".part")
        self.limit_bps, self.on_progress, self.timeout = limit_bps, on_progress, timeout
        self._pause = threading.Event()
        self._cancel = threading.Event()

    def pause(self) -> None:
        self._pause.set()

    def cancel(self) -> None:
        """Остановить; `.part` удаляет `discard()` после выхода из `run()`."""
        self._cancel.set()
        self._pause.set()

    @property
    def stopped(self) -> str | None:
        """«cancelled», «paused» или None — запрошенная остановка ещё не обработана."""
        if self._cancel.is_set():
            return "cancelled"
        return "paused" if self._pause.is_set() else None

    def discard(self) -> None:
        self.part.unlink(missing_ok=True)

    def _check_space(self, need: int) -> None:
        free = shutil.disk_usage(self.target.parent).free
        if need + SPACE_RESERVE > free:
            raise DownloadError("download.no_space", need=need, free=free)

    def run(self) -> Path:
        """Скачать (или продолжить) и активировать. Paused — `.part` сохранён."""
        self._pause.clear()
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists() and _hash_file(self.target).hexdigest() == self.sha256:
            return self.target
        done = self.part.stat().st_size if self.part.exists() else 0
        if self.size is not None:
            if done > self.size:
                self.part.unlink()
                done = 0
            self._check_space(self.size - done)
        request = web.request(self.url, {"Range": f"bytes={done}-"} if done else None)
        try:
            response = web.OPENER.open(request, timeout=self.timeout)
        except web.Refused:
            raise DownloadError("download.only_https") from None
        with response:
            if done and response.status != 206:  # сервер не умеет Range — заново
                done = 0
            length = response.headers.get("Content-Length")
            total = done + int(length) if length else self.size
            if self.size is None and total is not None:
                self._check_space(total - done)
            digest = _hash_file(self.part) if done else hashlib.sha256()
            started, got = time.monotonic(), 0
            with self.part.open("ab" if done else "wb") as file:
                while True:
                    if self._pause.is_set():
                        if self._cancel.is_set():
                            raise Cancelled("download.cancelled")
                        raise Paused("download.paused")
                    block = response.read(min(CHUNK, self.limit_bps or CHUNK))
                    if not block:
                        break
                    file.write(block)
                    digest.update(block)
                    got += len(block)
                    if self.on_progress:
                        self.on_progress(Progress(done + got, total))
                    if self.limit_bps:
                        ahead = got / self.limit_bps - (time.monotonic() - started)
                        if ahead > 0:
                            time.sleep(ahead)
                file.flush()
                os.fsync(file.fileno())
        if total is not None and self.part.stat().st_size != total:
            raise Interrupted("download.interrupted")
        if digest.hexdigest() != self.sha256:
            self.part.unlink()
            raise DownloadError("download.hash_mismatch")
        os.replace(self.part, self.target)
        return self.target
