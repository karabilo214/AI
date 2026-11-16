"""3.11 (CORE-UI-01): реестр UI-разделов и ленивое подключение модулей.

- «Чат» — встроенный раздел, есть всегда; Agent и Character — разделы модулей
  (этапы 6 и 9), появляются, только если модуль установлен.
- Раздел модуля описывается в манифесте: `ui = {"section": {...}}`. Оболочка не
  загружает код модулей — у неё встроенные виды (`VIEWS`), модуль даёт описание.
- Установленные модули ищутся в `<папка приложения>/modules/<id>/manifest.json`.
- Ленивость: процесс модуля стартует при первом открытии раздела, а не при запуске.
  Повторное открытие ничего не перезапускает, поэтому переключение разделов не меняет
  состояние и разрешения модуля. Ошибка модуля — статус его раздела, остальные работают.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Manifest
from ramvi_service.modules.host import ModuleHost

VIEWS = ("chat", "agent", "character")  # виды, которые умеет показывать оболочка
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


CORE = (Section("chat", "section.chat", "chat", 10, "chat"),)


def section_of(manifest: Manifest) -> Section | None:
    """Описание раздела из манифеста; без `ui.section` — модуль без раздела."""
    ui = manifest.ui or {}
    raw = ui.get("section")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("ui.section должен быть объектом")
    view, title, icon, order = (
        raw.get("view"),
        raw.get("title_key"),
        raw.get("icon", "module"),
        raw.get("order", 100),
    )
    if view not in VIEWS or view == "chat":
        raise ValueError(f"ui.section.view должен быть одним из {VIEWS[1:]}")
    if not isinstance(title, str) or not _TITLE.fullmatch(title):
        raise ValueError("ui.section.title_key — ключ перевода section.*")
    if not isinstance(icon, str) or not _ICON.fullmatch(icon):
        raise ValueError("ui.section.icon — короткое имя иконки")
    if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 10_000:
        raise ValueError("ui.section.order — целое 0–10000")
    return Section(manifest.module_id, title, icon, order, view, manifest.module_id)


def load_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("манифест должен быть объектом JSON")
    for name in ("required", "optional", "capabilities", "permissions"):
        if name in raw:
            raw[name] = tuple(raw[name])
    return Manifest(**raw)


class SectionRegistry:
    def __init__(self, host: ModuleHost) -> None:
        self.host = host
        self._sections: dict[str, Section] = {s.id: s for s in CORE}
        self.problems: list[dict[str, str]] = []  # модули, которые не удалось установить

    def install(self, manifest: Manifest) -> Section | None:
        """Зарегистрировать модуль в Host; процесс модуля не запускается."""
        section = section_of(manifest)
        if section is not None and section.id in self._sections:
            raise ValueError(f"раздел {section.id} уже есть")
        self.host.register(manifest)
        if section is not None:
            self._sections[section.id] = section
        return section

    def discover(self, modules_dir: Path) -> None:
        """Установленные модули из `<папка>/<id>/manifest.json`; испорченный — в problems."""
        if not modules_dir.is_dir():
            return
        for path in sorted(modules_dir.glob("*/manifest.json")):
            try:
                self.install(load_manifest(path))
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
