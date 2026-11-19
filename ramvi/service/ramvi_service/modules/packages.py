"""8.8 (CORE-MOD-08): пакеты модулей — установка, обновление, отключение, удаление, откат.

Пакет `ramvi.module/1` — ZIP:

    package.json   {"schema": "ramvi.module/1", "module_id", "version", "module_api",
                    "files": {"<путь>": {"sha256", "size"}}}
    manifest.json  манифест модуля (тот же, что читает SectionRegistry)
    code/          Python-код модуля (добавляется в путь процесса модуля)
    migrations/    SQL-миграции его пространства данных
    i18n/          его переводы
    resources/     прочие файлы

Устанавливается в `<приложение>/modules/<id>/` — ту же раскладку читает поиск модулей при
запуске. Данные модуля живут отдельно (`<данные>/modules/<id>/`), поэтому установка кода,
его замена и удаление данных не касаются; стереть данные можно только явным выбором.

Проверки: схема пакета, версия Module API (мажорная совпадает с ядром, минорная не новее),
module_id и версия, SHA-256 и размер каждого файла, нет лишних файлов и путей наружу.
Обновление: новая версия готовится рядом, прежняя уходит в `.backup/<id>/<версия>`; если
модуль новой версии не зарегистрировался, миграции упали или процесс не стартовал — старый
код возвращается на место, а данные возвращает `ModuleHost.update` из своей копии.
Другие модули (Chat, Agent) при этом не перезапускаются и не меняются.
"""

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import ID, MODULE_API_VERSION, VERSION, Manifest
from ramvi_service.module_data import load_migrations

SCHEMA = "ramvi.module/1"
PACKAGE_FILE = "package.json"
DISABLED = ".disabled"
TOP = ("manifest.json", "code", "migrations", "i18n", "resources")
MAX_FILES = 20_000
MAX_BYTES = 2 * 2**30


def api_compatible(version: str, core: str = MODULE_API_VERSION) -> bool:
    """Мажорная версия Module API та же, минорная — не новее ядра."""
    try:
        major, minor = (int(x) for x in str(version).split("."))
        core_major, core_minor = (int(x) for x in core.split("."))
    except ValueError:
        return False
    return major == core_major and minor <= core_minor


def _safe(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or ":" in path.parts[0]
        or path.parts[0] not in TOP
    ):
        raise ChatError("module.package_invalid", detail=f"bad path: {name}")
    return path


def read_package(archive_path: Path) -> dict[str, Any]:
    """Прочитать и проверить пакет (без распаковки на диск); описание пакета."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ChatError("module.package_invalid", detail=str(error)) from None
    with archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if PACKAGE_FILE not in names:
            raise ChatError("module.package_invalid", detail="no package.json")
        try:
            meta = json.loads(archive.read(PACKAGE_FILE))
        except ValueError:
            raise ChatError("module.package_invalid", detail="package.json is not JSON") from None
        if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
            raise ChatError("module.package_invalid", detail=f"schema is not {SCHEMA}")
        module_id, version = meta.get("module_id"), meta.get("version")
        if not isinstance(module_id, str) or not ID.fullmatch(module_id):
            raise ChatError("module.package_invalid", detail="bad module_id")
        if not isinstance(version, str) or not VERSION.fullmatch(version):
            raise ChatError("module.package_invalid", detail="bad version")
        if not api_compatible(meta.get("module_api", "")):
            raise ChatError(
                "module.api_incompatible",
                module_id=module_id,
                need=str(meta.get("module_api")),
                core=MODULE_API_VERSION,
            )
        files = meta.get("files")
        if not isinstance(files, dict) or not files or len(files) > MAX_FILES:
            raise ChatError("module.package_invalid", detail="no file list")
        listed = set(files)
        actual = set(names) - {PACKAGE_FILE}
        if listed != actual:
            extra = sorted(actual - listed)[:3] or sorted(listed - actual)[:3]
            raise ChatError("module.integrity", module_id=module_id, file=", ".join(extra))
        total = 0
        for name, info in files.items():
            _safe(name)
            if not isinstance(info, dict) or not isinstance(info.get("sha256"), str):
                raise ChatError("module.integrity", module_id=module_id, file=name)
            data = archive.read(name)
            total += len(data)
            if total > MAX_BYTES or len(data) != info.get("size"):
                raise ChatError("module.integrity", module_id=module_id, file=name)
            if hashlib.sha256(data).hexdigest() != info["sha256"].lower():
                raise ChatError("module.integrity", module_id=module_id, file=name)
        if "manifest.json" not in files:
            raise ChatError("module.package_invalid", detail="no manifest.json")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8-sig"))
        if manifest.get("module_id") != module_id or manifest.get("version") != version:
            raise ChatError(
                "module.package_invalid", detail="manifest.json does not match package.json"
            )
        if not api_compatible(manifest.get("api_version", MODULE_API_VERSION)):
            raise ChatError(
                "module.api_incompatible",
                module_id=module_id,
                need=str(manifest.get("api_version")),
                core=MODULE_API_VERSION,
            )
    return {
        "module_id": module_id,
        "version": version,
        "module_api": meta["module_api"],
        "files": len(files),
    }


def build_package(folder: Path, target: Path) -> dict[str, Any]:
    """Собрать пакет из папки модуля (manifest.json, code/, migrations/, i18n/, resources/)."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
    files: dict[str, dict[str, Any]] = {}
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.relative_to(folder).parts[0] in TOP
    )
    for path in paths:
        data = path.read_bytes()
        files[path.relative_to(folder).as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    meta = {
        "schema": SCHEMA,
        "module_id": manifest["module_id"],
        "version": manifest["version"],
        "module_api": manifest.get("api_version", MODULE_API_VERSION),
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PACKAGE_FILE, json.dumps(meta, ensure_ascii=False, indent=2))
        for path in paths:
            archive.write(path, path.relative_to(folder).as_posix())
    return meta


def _manifest(folder: Path) -> Manifest:
    from ramvi_service.sections import load_manifest

    return load_manifest(folder / "manifest.json")


class ModulePackages:
    """Пакеты модулей в `<приложение>/modules`; регистрация — через SectionRegistry/ModuleHost."""

    def __init__(self, modules_dir: Path, registry: Any) -> None:
        self.dir = modules_dir
        self.registry = registry
        self.host = registry.host
        self.backups = modules_dir / ".backup"

    def code_path(self, module_id: str) -> None:
        """Папку code/ установленного модуля — в путь его процесса."""
        code = self.dir / module_id / "code"
        if code.is_dir():
            self.host.code_paths[module_id] = [str(code)]
        else:
            self.host.code_paths.pop(module_id, None)

    def list(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for folder in sorted(p for p in self.dir.glob("*") if (p / "manifest.json").is_file()):
            try:
                manifest = _manifest(folder)
            except Exception as error:  # noqa: BLE001 — сломанный пакет показывается, а не роняет список
                result.append(
                    {"module_id": folder.name, "error": f"{type(error).__name__}: {error}"}
                )
                continue
            backups = self.backups / manifest.module_id
            result.append(
                {
                    "module_id": manifest.module_id,
                    "version": manifest.version,
                    "api_version": manifest.api_version,
                    "disabled": (folder / DISABLED).exists(),
                    "packaged": (folder / PACKAGE_FILE).exists(),
                    "previous": sorted(p.name for p in backups.glob("*"))
                    if backups.is_dir()
                    else [],
                }
            )
        return result

    def _registered(self, module_id: str) -> bool:
        return module_id in self.host.status()

    def _register(self, folder: Path) -> None:
        from ramvi_service.sections import load_texts

        manifest = _manifest(folder)
        self.code_path(manifest.module_id)
        self.registry.install(
            manifest, load_migrations(folder / "migrations"), load_texts(folder, manifest.module_id)
        )

    def data_backup(self, module_id: str, version: str) -> Path:
        """Копия данных версии `version` (в папке данных модулей, рядом с пространствами)."""
        return Path(self.host.root) / ".backup" / module_id / version

    def install(self, archive_path: Path, *, data_from: Path | None = None) -> dict[str, Any]:
        """Установить или обновить модуль из пакета; при ошибке — прежняя версия на месте."""
        info = read_package(Path(archive_path))
        module_id, version = info["module_id"], info["version"]
        target = self.dir / module_id
        self.dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=self.dir) as staging:
            stage = Path(staging) / module_id
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    destination = stage / name  # имена проверены read_package
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
            if not target.exists():
                stage.replace(target)
                try:
                    self._register(target)
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    self.host.code_paths.pop(module_id, None)
                    raise
                return {
                    "module_id": module_id,
                    "version": version,
                    "installed": True,
                    "previous": None,
                }
            previous = _manifest(target).version
            backup = self.backups / module_id / previous
            if backup.exists():
                shutil.rmtree(backup)
            backup.parent.mkdir(parents=True, exist_ok=True)
            disabled = (target / DISABLED).exists()
            target.replace(backup)
            stage.replace(target)
            if disabled:
                (target / DISABLED).touch()
            try:
                self._swap_in(
                    target, keep_backup=self.data_backup(module_id, previous), data_from=data_from
                )
            except Exception as error:
                shutil.rmtree(target, ignore_errors=True)
                backup.replace(target)
                self.code_path(module_id)
                # Хост откатил данные и поднял модуль, пока в папке ещё был новый код, —
                # перезапустить его с прежним кодом и прежними миграциями.
                if self._registered(module_id):
                    self.host.update(
                        module_id,
                        _manifest(target),
                        sql_migrations=load_migrations(target / "migrations"),
                    )
                raise ChatError(
                    "module.update_failed",
                    module_id=module_id,
                    version=version,
                    previous=previous,
                    detail=f"{type(error).__name__}: {error}",
                ) from error
        return {"module_id": module_id, "version": version, "installed": True, "previous": previous}

    def _swap_in(self, folder: Path, keep_backup: Path, data_from: Path | None) -> None:
        """Новая версия кода уже в папке модуля — обновить регистрацию (данные — с откатом)."""
        from ramvi_service.sections import load_texts

        manifest = _manifest(folder)
        self.code_path(manifest.module_id)
        texts = load_texts(folder, manifest.module_id)
        if not self._registered(manifest.module_id):  # был отключён — просто зарегистрировать
            if not (folder / DISABLED).exists():
                self.registry.install(manifest, load_migrations(folder / "migrations"), texts)
            return
        self.host.update(
            manifest.module_id,
            manifest,
            sql_migrations=load_migrations(folder / "migrations"),
            keep_backup=keep_backup,
            data_from=data_from,
        )
        self.registry.remove(manifest.module_id)
        self.registry.restore(manifest, texts)

    def rollback(self, module_id: str) -> dict[str, Any]:
        """Вернуть предыдущую версию из резервной копии (последнюю по номеру версии)."""
        backups = self.backups / module_id
        versions = sorted(
            (p for p in backups.glob("*") if VERSION.fullmatch(p.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
        )
        if not versions:
            raise ChatError("module.no_backup", module_id=module_id)
        chosen = versions[-1]
        data = self.data_backup(module_id, chosen.name)
        with tempfile.TemporaryDirectory(prefix=".rollback-", dir=self.dir) as temporary:
            package = Path(temporary) / "previous.zip"
            build_package(chosen, package)
            # Данные той версии — из её копии (новая схема базы старому коду не подходит);
            # данные текущей версии сохраняются в её копию при этой же замене.
            result = self.install(package, data_from=data if data.is_dir() else None)
        shutil.rmtree(chosen, ignore_errors=True)
        shutil.rmtree(data, ignore_errors=True)
        return result | {"rolled_back": True}

    def disable(self, module_id: str) -> dict[str, Any]:
        """Отключить: процесс остановлен, раздел скрыт, код и данные на месте."""
        folder = self._folder(module_id)
        if self._registered(module_id):
            self.host.uninstall(module_id, purge_data=False)
            self.registry.remove(module_id)
        (folder / DISABLED).touch()
        return {"module_id": module_id, "disabled": True}

    def enable(self, module_id: str) -> dict[str, Any]:
        folder = self._folder(module_id)
        (folder / DISABLED).unlink(missing_ok=True)
        if not self._registered(module_id):
            self._register(folder)
        return {"module_id": module_id, "disabled": False}

    def uninstall(self, module_id: str, purge_data: bool = False) -> dict[str, Any]:
        """Удалить код модуля; данные — только при явном purge_data."""
        folder = self._folder(module_id)
        if self._registered(module_id):
            self.host.uninstall(module_id, purge_data=purge_data)
            self.registry.remove(module_id)
        elif purge_data:
            from ramvi_service.scoped import ScopedData

            ScopedData(self.host.root, module_id).reset()
        shutil.rmtree(folder)
        shutil.rmtree(self.backups / module_id, ignore_errors=True)
        if purge_data:
            shutil.rmtree(self.host.root / ".backup" / module_id, ignore_errors=True)
        self.host.code_paths.pop(module_id, None)
        return {"module_id": module_id, "uninstalled": True, "purged": purge_data}

    def _folder(self, module_id: str) -> Path:
        if not ID.fullmatch(module_id) or not (self.dir / module_id / "manifest.json").is_file():
            raise ChatError("module.not_installed", module_id=module_id)
        return self.dir / module_id
