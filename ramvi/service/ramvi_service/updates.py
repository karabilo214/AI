"""8.3 (PORT-04): обновления — подписанный пакет, подготовка отдельно, применение с откатом.

Пакет обновления — zip (он же офлайн-архив; по сети скачивается такой же):
`manifest.json` (схема `ramvi.update/1`: версия, файлы с SHA-256 и размером, что удалить),
`manifest.sig` (подпись Ed25519 точных байтов манифеста, base64) и `payload/<путь>`.

1. **Проверка** — подпись одним из доверенных ключей выпуска (`TRUSTED_KEYS`; закрытый
   ключ есть только у машины сборки), версия новее текущей, пути — только внутри
   разрешённых частей комплекта (`ramvi/`, `runtime/`, `modules/`, `licenses/`, `RAMVI.exe`,
   `BUILD.json`, `LICENSE`, `THIRD_PARTY_NOTICES.txt`). Данные, модели, `settings.json` и сама папка `updates`
   пакету недоступны — неудачное обновление не может удалить историю чатов и модели.
2. **Подготовка** (`stage`) — распаковка в `updates/staging/<версия>` со сверкой SHA-256 и
   размера каждого файла; рабочие файлы приложения не трогаются.
3. **Применение** (`apply`, при остановленном сервисе — его делает оболочка перед
   запуском или CLI) — резервная копия базы; каждый заменяемый файл не удаляется, а
   переносится в `updates/backup/<версия>` (переименование работает и для занятых файлов
   Windows); журнал перемещений пишется на диск до каждого шага. Ошибка посреди
   применения — сразу откат.
4. **Проверка запуска** — `mark_good` после успешного старта; не стартовало — `rollback`.
5. **Откат** — файлы возвращаются по журналу; база восстанавливается из копии, только если
   новая версия успела её изменить (новые миграции), а изменённая база сохраняется рядом.

Проверка по сети — только по согласию (`update_check` в настройках, только HTTPS).
"""

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ramvi_service import __version__, ed25519, web
from ramvi_service.i18n import UserError

SCHEMA = "ramvi.update/1"
PRODUCT = "RAMVI"
# Открытые ключи выпусков RAMVI (hex, 32 байта). Закрытая часть хранится только у
# владельца вне репозитория; пакет, подписанный другим ключом, не устанавливается.
TRUSTED_KEYS: tuple[str, ...] = (
    "15d704139a14739f953f61ef2c6e58647efae41c92b8a4c22a50fcd68fd019be",  # выпуск RAMVI, 2026-11-19
)
ALLOWED_DIRS = ("ramvi", "runtime", "modules", "licenses")
ALLOWED_FILES = ("RAMVI.exe", "BUILD.json", "LICENSE", "THIRD_PARTY_NOTICES.txt")
MAX_MANIFEST = 8 * 1024 * 1024
MAX_FILES = 100_000
MAX_TOTAL = 16 * 1024**3
DB_FILE = "ramvi.sqlite3"


class UpdateError(UserError):
    """Ошибка обновления с ключом сообщения (`update.*`)."""


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Manifest:
    version: str
    files: tuple[FileEntry, ...]
    remove: tuple[str, ...] = ()
    notes: str = ""


def parse_version(text: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(p) for p in text.strip().split("."))
    except ValueError:
        raise UpdateError("update.bad_version", version=text) from None
    if not parts or len(parts) > 4 or any(p < 0 for p in parts):
        raise UpdateError("update.bad_version", version=text)
    return parts


def safe_path(path: str) -> str:
    """Путь файла из пакета: относительный, без `..`, только разрешённые части комплекта."""
    if not isinstance(path, str) or not path or "\\" in path or "\0" in path or ":" in path:
        raise UpdateError("update.bad_path", path=str(path)[:200])
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise UpdateError("update.bad_path", path=path)
    top = pure.parts[0]
    allowed = (top in ALLOWED_DIRS and len(pure.parts) > 1) or (
        len(pure.parts) == 1 and top in ALLOWED_FILES
    )
    if not allowed:
        raise UpdateError("update.forbidden_path", path=path)
    return pure.as_posix()


def verify_manifest(raw: bytes, signature: str, keys: tuple[str, ...]) -> Manifest:
    """Подпись одним из доверенных ключей и строгий разбор манифеста."""
    if not keys:
        raise UpdateError("update.no_trusted_key")
    if len(raw) > MAX_MANIFEST:
        raise UpdateError("update.bad_manifest", detail="too large")
    try:
        sig = base64.b64decode(signature.strip(), validate=True)
    except ValueError:
        raise UpdateError("update.bad_signature") from None
    if not any(ed25519.verify(bytes.fromhex(key), raw, sig) for key in keys):
        raise UpdateError("update.bad_signature")
    try:
        data = json.loads(raw)
    except ValueError:
        raise UpdateError("update.bad_manifest", detail="json") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("product") != PRODUCT:
        raise UpdateError("update.bad_manifest", detail="schema")
    version = str(data.get("version", ""))
    parse_version(version)
    files, total, seen = [], 0, set()
    for item in data.get("files") or []:
        if not isinstance(item, dict):
            raise UpdateError("update.bad_manifest", detail="file")
        path = safe_path(item.get("path", ""))
        digest, size = str(item.get("sha256", "")), item.get("size")
        if len(digest) != 64 or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise UpdateError("update.bad_manifest", detail=path)
        if path.lower() in seen:
            raise UpdateError("update.bad_manifest", detail="duplicate " + path)
        seen.add(path.lower())
        files.append(FileEntry(path, digest.lower(), size))
        total += size
    if not files or len(files) > MAX_FILES or total > MAX_TOTAL:
        raise UpdateError("update.bad_manifest", detail="files")
    remove = tuple(safe_path(p) for p in data.get("remove") or [])
    return Manifest(version, tuple(files), remove, str(data.get("notes", ""))[:4000])


def current_version(app_dir: Path) -> str:
    """Версия установленного комплекта: BUILD.json, иначе версия кода сервиса."""
    try:
        version = json.loads((app_dir / "BUILD.json").read_text(encoding="utf-8")).get("version")
        if isinstance(version, str) and version:
            return version
    except (OSError, ValueError, AttributeError):
        pass
    return __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_version(db_file: Path) -> int:
    """Последняя применённая миграция ядра (0 — базы нет или таблицы миграций нет)."""
    if not db_file.exists():
        return 0
    connection = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT max(version) FROM schema_migrations WHERE owner = 'core'"
        ).fetchone()
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0
    finally:
        connection.close()


class Updater:
    """Подготовка, применение и откат обновлений одного комплекта."""

    def __init__(self, app_dir: Path, data_dir: Path, keys: tuple[str, ...] = TRUSTED_KEYS) -> None:
        self.app_dir, self.data_dir = Path(app_dir).resolve(), Path(data_dir).resolve()
        self.keys = keys
        self.root = self.app_dir / "updates"

    # --- состояние --------------------------------------------------------------------

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    def status(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return {"current": current_version(self.app_dir), **state}

    def _save(self, state: dict[str, Any]) -> None:
        state = {k: v for k, v in state.items() if k != "current"}  # версия — из BUILD.json
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.state_file.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temp, self.state_file)

    # --- подготовка -------------------------------------------------------------------

    def stage(self, archive: Path) -> dict[str, Any]:
        """Проверить пакет и распаковать в staging; рабочие файлы не меняются."""
        try:
            package = zipfile.ZipFile(archive)
        except (OSError, zipfile.BadZipFile):
            raise UpdateError("update.bad_archive") from None
        with package:
            try:
                raw = package.read("manifest.json")
                signature = package.read("manifest.sig").decode("ascii")
            except (KeyError, UnicodeDecodeError):
                raise UpdateError("update.bad_archive") from None
            manifest = verify_manifest(raw, signature, self.keys)
            current = current_version(self.app_dir)
            if parse_version(manifest.version) <= parse_version(current):
                raise UpdateError("update.not_newer", version=manifest.version, current=current)
            staging = self.root / "staging" / manifest.version
            shutil.rmtree(staging, ignore_errors=True)
            for entry in manifest.files:
                target = staging / "payload" / entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                try:
                    with package.open("payload/" + entry.path) as source, open(target, "wb") as out:
                        while chunk := source.read(1024 * 1024):
                            size += len(chunk)
                            if size > entry.size:
                                break
                            digest.update(chunk)
                            out.write(chunk)
                except KeyError:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise UpdateError("update.missing_file", path=entry.path) from None
                if size != entry.size or digest.hexdigest() != entry.sha256:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise UpdateError("update.bad_file", path=entry.path)
            (staging / "manifest.json").write_bytes(raw)
            (staging / "manifest.sig").write_text(signature, encoding="ascii")
        # Новый цикл: отметки прошлого обновления (откат, журнал, проверка) больше не к месту.
        stale = (
            "rolled_back",
            "rollback_failed",
            "journal",
            "applied",
            "applying",
            "health",
            "previous",
            "db_schema",
            "applied_at",
        )
        state = {k: v for k, v in self.status().items() if k not in stale}
        self._save({**state, "staged": manifest.version, "notes": manifest.notes})
        return {"staged": manifest.version, "files": len(manifest.files), "current": current}

    # --- сеть (только по согласию) ----------------------------------------------------

    def check_online(
        self, consent: bool, url: str, fetch: Callable[[str], bytes] | None = None
    ) -> dict[str, Any] | None:
        """Есть ли новая версия: `<url>/latest.json` + `.sig`; без согласия — не ходит в сеть.

        Возвращает версию и адрес пакета (поле `archive`), если она новее текущей."""
        if not consent:
            raise UpdateError("update.no_consent")
        base = url.rstrip("/")
        web.check(base)
        fetch = fetch or _fetch
        raw = fetch(base + "/latest.json")
        signature = fetch(base + "/latest.json.sig").decode("ascii")
        if len(raw) > MAX_MANIFEST:
            raise UpdateError("update.bad_manifest", detail="too large")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise UpdateError("update.bad_manifest", detail="latest")
        # latest.json подписан тем же ключом; сам пакет проверяется ещё раз при подготовке.
        if not any(
            ed25519.verify(bytes.fromhex(k), raw, base64.b64decode(signature)) for k in self.keys
        ):
            raise UpdateError("update.bad_signature")
        version = str(data.get("version", ""))
        if parse_version(version) <= parse_version(current_version(self.app_dir)):
            return None
        archive = str(data.get("archive", ""))
        web.check(archive)
        return {"version": version, "archive": archive, "notes": str(data.get("notes", ""))}

    # --- применение и откат -----------------------------------------------------------

    def _backup_database(self, backup: Path) -> int:
        db_file = self.data_dir / DB_FILE
        if not db_file.exists():
            return 0
        source = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
        target = sqlite3.connect(backup / DB_FILE)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return _schema_version(backup / DB_FILE)

    def apply(self) -> dict[str, Any]:
        """Заменить файлы подготовленной версией; при ошибке — откат. Сервис остановлен."""
        state = self.status()
        version = state.get("staged")
        if not version:
            raise UpdateError("update.nothing_staged")
        staging = self.root / "staging" / version
        manifest = verify_manifest(
            (staging / "manifest.json").read_bytes(),
            (staging / "manifest.sig").read_text(encoding="ascii"),
            self.keys,
        )
        backup = self.root / "backup" / version
        shutil.rmtree(backup, ignore_errors=True)
        (backup / "files").mkdir(parents=True)
        previous = current_version(self.app_dir)
        journal: list[dict[str, Any]] = []
        state = {
            **state,
            "applying": version,
            "previous": previous,
            "journal": journal,
            "db_schema": self._backup_database(backup),
            "health": "pending",
        }
        self._save(state)
        try:
            moves = [(e.path, True) for e in manifest.files] + [(p, False) for p in manifest.remove]
            for path, replace in moves:
                target = self.app_dir / path
                saved = backup / "files" / path
                entry = {"path": path, "had_old": target.exists(), "new": replace}
                journal.append(entry)
                self._save(state)  # журнал на диске до шага: после сбоя ясно, что откатывать
                if target.exists():
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, saved)
                if replace:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / "payload" / path, target)
        except OSError as error:
            self.rollback()
            raise UpdateError("update.apply_failed", detail=type(error).__name__) from None
        state.pop("applying", None)
        state.pop("staged", None)
        self._save({**state, "applied": version, "applied_at": time.time()})
        shutil.rmtree(staging, ignore_errors=True)
        return {"applied": version, "previous": previous, "files": len(manifest.files)}

    def mark_good(self) -> None:
        """Новая версия запустилась — откат больше не предлагается автоматически."""
        self._save({**self.status(), "health": "good"})

    def rollback(self) -> dict[str, Any]:
        """Вернуть прежние файлы и, если новая версия изменила базу, — прежнюю базу."""
        state = self.status()
        version = state.get("applying") or state.get("applied")
        journal = state.get("journal") or []
        if not version or not journal:
            raise UpdateError("update.nothing_to_roll_back")
        backup = self.root / "backup" / version
        rejected = backup / "rejected"
        failed: list[str] = []
        for entry in reversed(journal):
            target = self.app_dir / safe_path(entry["path"])
            saved = backup / "files" / entry["path"]
            try:
                if (
                    entry.get("new")
                    and target.exists()
                    and (saved.exists() or not entry["had_old"])
                ):
                    # Новый файл может быть занят (откат запускает новый python.exe с новым
                    # python312.dll): удалить занятый файл Windows не даст, а перенести — даст.
                    aside = rejected / entry["path"]
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    if aside.exists():
                        aside.unlink()
                    os.replace(target, aside)
                if saved.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(saved, target)
            except OSError:
                failed.append(entry["path"])  # остальные файлы всё равно возвращаются
        restored_db = False
        db_file, db_copy = self.data_dir / DB_FILE, backup / DB_FILE
        if db_copy.exists() and _schema_version(db_file) > int(state.get("db_schema") or 0):
            # Новая версия успела применить миграции: прежний код такую базу не откроет.
            keep = (
                self.data_dir / "backups" / f"{DB_FILE}.after-update-{version}-{int(time.time())}"
            )
            keep.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", "-wal", "-shm"):
                part = Path(str(db_file) + suffix)
                if part.exists():
                    os.replace(part, Path(str(keep) + suffix))
            shutil.copy2(db_copy, db_file)
            restored_db = True
        if failed:
            self._save({**state, "rollback_failed": failed})
            raise UpdateError("update.rollback_incomplete", files=", ".join(failed))
        for key in ("applying", "applied", "journal", "staged", "rollback_failed"):
            state.pop(key, None)
        self._save({**state, "rolled_back": version, "health": "rolled_back"})
        return {"rolled_back": version, "database_restored": restored_db}


def _fetch(url: str) -> bytes:
    with web.OPENER.open(web.request(url), timeout=60) as response:
        data: bytes = response.read(MAX_MANIFEST + 1)
    return data


def main(argv: list[str] | None = None) -> None:
    """CLI для оболочки: `python -m ramvi_service.updates status|stage|apply|rollback|mark-good`."""
    import argparse

    parser = argparse.ArgumentParser(prog="ramvi-update")
    parser.add_argument("action", choices=["status", "stage", "apply", "rollback", "mark-good"])
    parser.add_argument("--app-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    updater = Updater(args.app_dir, args.data_dir)
    try:
        if args.action == "status":
            result: Any = updater.status()
        elif args.action == "stage":
            if args.archive is None:
                parser.error("--archive is required for stage")
            result = updater.stage(args.archive)
        elif args.action == "apply":
            result = updater.apply()
        elif args.action == "rollback":
            result = updater.rollback()
        else:
            updater.mark_good()
            result = updater.status()
    except UpdateError as error:
        print(
            json.dumps({"error": {"code": error.code, "params": error.params}}, ensure_ascii=False)
        )
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
