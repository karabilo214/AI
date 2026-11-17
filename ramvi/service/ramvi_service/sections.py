"""3.11 (CORE-UI-01): реестр UI-разделов и ленивое подключение модулей.

- «Чат» — встроенный раздел, есть всегда; Agent и Character — разделы модулей
  (этапы 6 и 9), появляются, только если модуль установлен.
- Раздел модуля описывается в манифесте: `ui = {"section": {...}}`. Оболочка не
  загружает код модулей — у неё встроенные виды (`VIEWS`), модуль даёт описание.
- Установленные модули ищутся в `<папка приложения>/modules/<id>/manifest.json`.
- Ленивость: процесс модуля стартует при первом открытии раздела, а не при запуске.
  Повторное открытие ничего не перезапускает, поэтому переключение разделов не меняет
  состояние и разрешения модуля. Ошибка модуля — статус его раздела, остальные работают.
- 3.14: переводы модуля — `<папка модуля>/i18n/<язык>.json`. Ключи только свои
  (`section.<id>`, `module.<id>.*`), на всех языках ядра, с одинаковыми параметрами;
  иначе модуль не устанавливается. Оболочка получает их через `GET /i18n`.
"""

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ramvi_storage.migrations import Migration

from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Manifest
from ramvi_service.i18n import LANGUAGES, check_catalogs, load_catalog
from ramvi_service.module_data import load_migrations
from ramvi_service.modules.host import ModuleHost

VIEWS = ("chat", "models", "agent", "character")  # виды, которые умеет показывать оболочка
CORE_VIEWS = ("chat", "models")  # встроенные разделы; модулям недоступны
_ICON = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_TITLE = re.compile(r"section\.[a-z0-9_.]{1,64}")


@dataclass(frozen=True)
class Section:
    id: str
    title_key: str
    icon: str
    order: int
    view: str
    module_id: str | None = None


CORE = (
    Section("chat", "section.chat", "chat", 10, "chat"),
    Section("models", "section.models", "models", 15, "models"),  # 4.11: экран «Модели»
)


def section_of(manifest: Manifest) -> Section | None:
    """Описание раздела из манифеста; без `ui.section` — модуль без раздела."""
    ui = manifest.ui or {}
    raw = ui.get("section")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("ui.section must be an object")
    view, title, icon, order = (
        raw.get("view"),
        raw.get("title_key"),
        raw.get("icon", "module"),
        raw.get("order", 100),
    )
    if view not in VIEWS or view in CORE_VIEWS:
        allowed = tuple(v for v in VIEWS if v not in CORE_VIEWS)
        raise ValueError(f"ui.section.view must be one of {allowed}")
    if not isinstance(title, str) or not _TITLE.fullmatch(title):
        raise ValueError("ui.section.title_key must be a translation key section.*")
    if not isinstance(icon, str) or not _ICON.fullmatch(icon):
        raise ValueError("ui.section.icon must be a short icon name")
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 10_000:
        raise ValueError("ui.section.order must be an integer 0–10000")
    return Section(manifest.module_id, title, icon, order, view, manifest.module_id)


def load_texts(folder: Path, module_id: str) -> dict[str, dict[str, str]]:
    """Переводы модуля из `<folder>/i18n`; нарушение правил 3.14 — ValueError."""
    i18n = folder / "i18n"
    if not i18n.is_dir():
        return {}
    catalogs = {path.stem: load_catalog(path.stem, i18n) for path in sorted(i18n.glob("*.json"))}
    check_texts(catalogs, module_id)
    return {lang: catalogs[lang] for lang in LANGUAGES}


def check_texts(catalogs: dict[str, dict[str, str]], module_id: str) -> None:
    allowed = re.compile(rf"section\.{module_id}|module\.{module_id}\.[a-z0-9_.]+")
    errors = check_catalogs(catalogs, where=f"module {module_id}", allowed=allowed)
    core = load_catalog(LANGUAGES[0])
    errors += [
        f"module {module_id}: key {key} already exists in the core catalog"
        for key in sorted(set().union(*catalogs.values()) & core.keys())
    ]
    if errors:
        raise ValueError("; ".join(errors))


def load_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("the manifest must be a JSON object")
    for name in ("required", "optional", "capabilities", "permissions"):
        if name in raw:
            raw[name] = tuple(raw[name])
    return Manifest(**raw)


class SectionRegistry:
    def __init__(self, host: ModuleHost) -> None:
        self.host = host
        self._sections: dict[str, Section] = {s.id: s for s in CORE}
        self.problems: list[dict[str, str]] = []  # модули, которые не удалось установить
        self._texts: dict[str, dict[str, dict[str, str]]] = {}  # модуль -> язык -> ключ -> текст

    def install(
        self,
        manifest: Manifest,
        sql: Sequence[Migration] = (),
        texts: dict[str, dict[str, str]] | None = None,
    ) -> Section | None:
        """Зарегистрировать модуль в Host (с его SQL-миграциями, 3.12, и переводами, 3.14);
        процесс не запускается."""
        section = section_of(manifest)
        if section is not None and section.id in self._sections:
            raise ValueError(f"section {section.id} already exists")
        if texts:
            check_texts(texts, manifest.module_id)
        self.host.register(manifest, approved_permissions=manifest.permissions, sql_migrations=sql)
        if section is not None:
            self._sections[section.id] = section
        if texts:
            self._texts[manifest.module_id] = texts
        return section

    def texts(self) -> dict[str, dict[str, str]]:
        """Переводы установленных модулей: язык -> ключ -> текст."""
        merged: dict[str, dict[str, str]] = {lang: {} for lang in LANGUAGES}
        for catalogs in self._texts.values():
            for lang in LANGUAGES:
                merged[lang].update(catalogs.get(lang, {}))
        return merged

    def discover(self, modules_dir: Path) -> None:
        """Установленные модули из `<папка>/<id>/manifest.json`; испорченный — в problems."""
        if not modules_dir.is_dir():
            return
        for path in sorted(modules_dir.glob("*/manifest.json")):
            try:
                manifest = load_manifest(path)
                self.install(
                    manifest,
                    load_migrations(path.parent / "migrations"),
                    load_texts(path.parent, manifest.module_id),
                )
            except Exception as error:  # один сломанный модуль не мешает остальным
                self.problems.append(
                    {"path": str(path), "error": f"{type(error).__name__}: {error}"}
                )

    def _state(self, section: Section) -> dict[str, Any]:
        if section.module_id is None:
            return {"state": "ready", "error": None}
        status = self.host.status(section.module_id)[section.module_id]
        state = status["state"]
        # registered — установлен, но не загружен (загрузится при первом открытии).
        return {"state": "available" if state == "registered" else state, "error": status["error"]}

    def remove(self, module_id: str) -> None:
        """Убрать раздел удалённого модуля."""
        self._sections = {k: v for k, v in self._sections.items() if v.module_id != module_id}
        self._texts.pop(module_id, None)

    def list(self) -> list[dict[str, Any]]:
        sections = sorted(self._sections.values(), key=lambda s: (s.order, s.id))
        return [{**asdict(s), **self._state(s)} for s in sections]

    def open(self, section_id: str) -> dict[str, Any]:
        """Открыть раздел; модуль активируется при первом открытии (ленивая загрузка)."""
        section = self._sections.get(section_id)
        if section is None:
            raise ChatError("section.not_found", section=section_id)
        if section.module_id is not None:
            state = self.host.status(section.module_id)[section.module_id]["state"]
            try:
                if state == "suspended":
                    self.host.resume(section.module_id)
                else:  # для уже активного модуля activate ничего не делает
                    self.host.activate(section.module_id)
            except Exception:  # noqa: S110 — состояние и текст ошибки уже в статусе Host
                pass
        return {**asdict(section), **self._state(section)}


class ModuleDataApi:
    """3.12: данные модулей для экрана приложения — список, экспорт, сброс, удаление."""

    def __init__(self, registry: SectionRegistry, exports: Path) -> None:
        self.registry, self.exports = registry, exports

    def list(self) -> list[dict[str, Any]]:
        host = self.registry.host
        return [
            {
                "module_id": module_id,
                "state": status["state"],
                "version": status["version"],
                "data_bytes": host.data_bytes(module_id),
            }
            for module_id, status in sorted(host.status().items())
        ]

    def _require(self, module_id: str) -> None:
        if module_id not in self.registry.host.status():
            raise ChatError("module.not_found", module_id=module_id)

    def export(self, module_id: str) -> dict[str, Any]:
        self._require(module_id)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return self.registry.host.export_data(module_id, self.exports / f"{module_id}-{stamp}.zip")

    def reset(self, module_id: str) -> dict[str, Any]:
        """Остановить модуль и стереть только его данные; Core и другие модули не трогаются."""
        self._require(module_id)
        host = self.registry.host
        if host.status(module_id)[module_id]["worker"] is not None:
            host.dispose(module_id)
        return dict(host.reset_data(module_id)[module_id])

    def uninstall(self, module_id: str, purge: bool) -> dict[str, Any]:
        self._require(module_id)
        self.registry.host.uninstall(module_id, purge_data=purge)
        self.registry.remove(module_id)
        return {"module_id": module_id, "purged": purge}
