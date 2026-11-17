"""Менеджер загрузок моделей (4.3 MOD-02/MOD-04): очередь поверх `downloads.Download`.

Скачивается только запись каталога (4.2) — адрес, размер и SHA-256 берутся из неё,
а не от клиента. Состояния хранятся в таблице `downloads`:
queued → running → verifying → done; paused, failed, cancelled.

- Одна загрузка за раз (ограничение параллелизма MOD-04), остальные ждут в очереди.
- Обрыв сети: несколько повторов с паузой; потом «пауза» с причиной
  `download.interrupted` — продолжение по Range с места остановки.
- После перезапуска приложения незаконченные загрузки стоят на паузе: без явного
  действия пользователя гигабайты не качаются.
- Готовый файл (хеш уже проверен до активации) подключается как модель через ту же
  проверку заголовка GGUF, что и импорт; не прошёл — `failed`, в список моделей не
  попадает.
"""

import re
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_storage.database import Database
from ramvi_storage.repositories import DownloadRecord, Downloads, Models, NotFound

from ramvi_service.catalog import Catalog, CatalogEntry
from ramvi_service.chat import ChatError
from ramvi_service.downloads import (
    Cancelled,
    Download,
    DownloadError,
    Interrupted,
    Paused,
    safe_target,
)
from ramvi_service.i18n import UserError, msg

RETRIES = (2.0, 5.0, 15.0)  # секунд между повторами после обрыва сети
SAVE_EVERY = 1.0  # как часто писать прогресс в базу


class DownloadsApiError(ChatError):
    """Ошибка запроса к загрузкам — ключ и параметры для интерфейса."""


def file_name(entry: CatalogEntry) -> str:
    """Имя файла в папке моделей: у Hugging Face — исходное, у Ollama — по тегу."""
    if entry.file_name.endswith(".gguf"):
        return entry.file_name
    slug = re.sub(r"[^a-z0-9._-]+", "-", f"{entry.source}-{entry.quantization}".lower())
    return slug.strip("-.")[:96] + ".gguf"


class DownloadManager:
    """Очередь загрузок с одним рабочим потоком."""

    def __init__(
        self,
        db: Database,
        catalog: Catalog,
        models_dir: Callable[[], Path],
        register: Callable[[Path, DownloadRecord], str],
        *,
        retries: tuple[float, ...] = RETRIES,
        limit_bps: int | None = None,
    ) -> None:
        self.repo, self.models = Downloads(db), Models(db)
        self.catalog, self.models_dir, self.register = catalog, models_dir, register
        self.retries, self.limit_bps = retries, limit_bps
        self._cv = threading.Condition()
        self._job: tuple[str, Download] | None = None
        self._wake = threading.Event()  # прерывает паузу между повторами
        self._progress: dict[str, int] = {}
        self._errors: dict[str, dict[str, Any]] = {}
        self._installed: dict[str, str] = {}
        self._closed = False
        for record in self.repo.unfinished():  # после перезапуска — пауза
            if record.state != "paused":
                self.repo.update(record.id, state="paused")
        self._thread = threading.Thread(target=self._loop, name="ramvi-downloads", daemon=True)
        self._thread.start()

    # --- действия ------------------------------------------------------------------

    def start(self, entry_id: str) -> dict[str, Any]:
        entry = self._entry(entry_id)
        with self._cv:
            for record in self.repo.unfinished():
                if record.model_id == entry.id:  # уже есть — не дублировать
                    if record.state == "paused":
                        self.repo.update(record.id, state="queued")
                        self._errors.pop(record.id, None)
                        self._cv.notify_all()
                    return self._json(self.repo.get(record.id))
            try:
                target = safe_target(self.models_dir(), file_name(entry))
            except DownloadError as error:
                raise DownloadsApiError(error.code, **error.params) from None
            record = self.repo.create(
                entry.id,
                entry.url,
                str(target) + ".part",
                entry.checksum.removeprefix("sha256:"),
                entry.bytes,
            )
            self._cv.notify_all()
        return self._json(record)

    def pause(self, download_id: str) -> dict[str, Any]:
        with self._cv:
            record = self._get(download_id)
            if record.state == "queued":
                self.repo.update(download_id, state="paused")
            elif record.state == "running" and self._job and self._job[0] == download_id:
                self._job[1].pause()
                self._wake.set()
            return self._json(self.repo.get(download_id))

    def resume(self, download_id: str) -> dict[str, Any]:
        with self._cv:
            record = self._get(download_id)
            if record.state in ("paused", "failed"):
                self.repo.update(download_id, state="queued")
                self._errors.pop(download_id, None)
                self._cv.notify_all()
            return self._json(self.repo.get(download_id))

    def cancel(self, download_id: str) -> dict[str, Any]:
        with self._cv:
            record = self._get(download_id)
            if record.state == "running" and self._job and self._job[0] == download_id:
                self._job[1].cancel()  # рабочий поток удалит .part и отметит cancelled
                self._wake.set()
            elif record.state not in ("done", "cancelled"):
                Path(record.temp_path).unlink(missing_ok=True)
                self.repo.update(download_id, state="cancelled")
            return self._json(self.repo.get(download_id))

    def list(self) -> list[dict[str, Any]]:
        rows = self.repo.db.all("SELECT id FROM downloads ORDER BY rowid")
        return [self._json(self.repo.get(row["id"])) for row in rows]

    def close(self) -> None:
        with self._cv:
            self._closed = True
            if self._job is not None:
                self._job[1].pause()  # .part остаётся — продолжится после запуска
            self._wake.set()
            self._cv.notify_all()
        self._thread.join(10)

    # --- рабочий поток -------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._cv:
                queued = self._next()
                while not self._closed and queued is None:
                    self._cv.wait()
                    queued = self._next()
                if self._closed or queued is None:
                    return
                record = self.repo.update(queued.id, state="running")
                target = Path(record.temp_path).with_suffix("")
                job = Download(
                    record.source,
                    target.parent,
                    target.name,
                    record.checksum,
                    record.total_bytes,
                    limit_bps=self.limit_bps,
                    on_progress=self._on_progress(record.id),
                )
                self._job = (record.id, job)
                self._wake.clear()
            try:
                self._run(record, job)
            except Exception as error:  # поток очереди не должен умирать из-за одной загрузки
                detail = type(error).__name__
                self._finish(record.id, "failed", msg("download.failed", detail=detail).json())
            finally:
                with self._cv:
                    self._job = None
                    self._progress.pop(record.id, None)

    def _next(self) -> DownloadRecord | None:
        return next((r for r in self.repo.unfinished() if r.state == "queued"), None)

    def _run(self, record: DownloadRecord, job: Download) -> None:
        for delay in (*self.retries, None):
            try:
                path = job.run()
                break
            except Cancelled:
                job.discard()
                self._finish(record.id, "cancelled")
                return
            except Paused:
                self._finish(record.id, "paused", bytes_done=self._part_size(job))
                return
            except Interrupted as error:  # соединение закрылось раньше конца файла
                interrupted = error.message.json()
            except DownloadError as error:
                self._finish(record.id, "failed", error=error.message.json())
                return
            except OSError:  # сеть
                interrupted = msg("download.interrupted").json()
            # Обрыв: повторить с места остановки, после последнего повтора — пауза.
            stopped = job.stopped
            if delay is not None and stopped is None:
                self._errors[record.id] = interrupted
                self._wake.wait(delay)  # прерывается паузой, отменой или выходом
                stopped = job.stopped  # run() сбросил бы флаг паузы — проверить здесь
                if stopped is None:
                    continue
            if stopped == "cancelled":
                job.discard()
                self._finish(record.id, "cancelled")
            else:
                self._finish(record.id, "paused", interrupted, self._part_size(job))
            return
        self.repo.update(record.id, state="verifying", bytes_done=record.total_bytes)
        try:
            model_id = self.register(path, record)
        except (UserError, ChatError) as error:  # файл не модель или не поддерживается
            self._finish(record.id, "failed", error=msg(error.code, **error.params).json())
            return
        self._installed[record.id] = model_id
        self._finish(record.id, "done")

    def _finish(
        self,
        download_id: str,
        state: str,
        error: dict[str, Any] | None = None,
        bytes_done: int | None = None,
    ) -> None:
        with self._cv:
            self.repo.update(download_id, state=state, bytes_done=bytes_done)
            if error is None:
                self._errors.pop(download_id, None)
            else:
                self._errors[download_id] = error

    def _on_progress(self, download_id: str) -> Callable[[Any], None]:
        saved = [time.monotonic()]

        def report(progress: Any) -> None:
            self._progress[download_id] = progress.done
            if time.monotonic() - saved[0] >= SAVE_EVERY:
                saved[0] = time.monotonic()
                self.repo.update(download_id, bytes_done=progress.done)

        return report

    @staticmethod
    def _part_size(job: Download) -> int:
        return job.part.stat().st_size if job.part.exists() else 0

    # --- вспомогательное -----------------------------------------------------------

    def _entry(self, entry_id: str) -> CatalogEntry:
        for raw in self.catalog.view()["entries"]:
            if raw["id"] == entry_id and raw.get("unsupported") is None:
                data = {k: v for k, v in raw.items() if k != "unsupported"}
                return CatalogEntry.from_json(data)
        raise DownloadsApiError("download.unknown_entry", entry_id=entry_id)

    def _get(self, download_id: str) -> DownloadRecord:
        try:
            return self.repo.get(download_id)
        except NotFound:
            raise DownloadsApiError("download.not_found", download_id=download_id) from None

    def _json(self, record: DownloadRecord) -> dict[str, Any]:
        done = self._progress.get(record.id, record.bytes_done)
        return {
            "id": record.id,
            "entry_id": record.model_id,
            "state": record.state,
            "bytes_done": done,
            "total_bytes": record.total_bytes,
            "file": Path(record.temp_path).with_suffix("").name,
            "error": self._errors.get(record.id),
            "model_id": self._installed.get(record.id) or self._model_for(record),
        }

    def _model_for(self, record: DownloadRecord) -> str | None:
        """Подключённая модель этой загрузки (и после перезапуска сервиса)."""
        if record.state != "done":
            return None
        for model in self.models.all():
            if model.capabilities.get("catalog_id") == record.model_id:
                return model.id
        return None


def register_downloaded(
    import_model: Callable[[str], dict[str, Any]], models: Models, catalog: Catalog
) -> Callable[[Path, DownloadRecord], str]:
    """Подключить скачанный файл как импорт и записать происхождение из каталога."""

    def register(path: Path, record: DownloadRecord) -> str:
        model = models.get(import_model(str(path))["id"])
        entry = next((e for e in catalog.view()["entries"] if e["id"] == record.model_id), None)
        model = replace(
            model,
            provider=entry["provider"] if entry else model.provider,
            sha256=record.checksum,
            license=entry["license"] if entry else model.license,
            capabilities={
                **model.capabilities,
                "imported": False,
                "catalog_id": record.model_id,
                **({"version": entry["version"]} if entry else {}),
            },
        )
        models.upsert(model)
        return model.id

    return register
