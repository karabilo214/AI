"""3.7: экран настроек — чтение и проверенная запись `settings.json` папки приложения.

Файл лежит в папке приложения (рядом с exe), а не в папке данных: иначе папку
данных нельзя было бы сменить. Оболочка читает из него `data_dir`, `language` и `theme`
до запуска сервиса; пишет — только через этот API (одна проверка, `settings.py`).

Что когда применяется:
- `data_dir`, `language` — после перезапуска приложения (`restart_required`);
- `theme` — сразу: оболочка меняет тему окна после сохранения (3.16);
- `models_dir` — сразу: «Обновить список» подключает найденные в ней модели на месте
  (`POST /models/scan`, в т.ч. Ollama `blobs`), и туда пойдут загрузки (4.3); ничего не копируется;
- `api_port`, `network_access` — вместе с внешним API (раздел 7: 7.1, 7.4);
  до этого порт не открывается, сеть выключена.
"""

import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ramvi_service import settings as st
from ramvi_service.chat import ChatError

UI_LANGUAGES = ("ru", "en")  # есть каталоги ramvi/i18n/<язык>.json
FOLDERS = ("data_dir", "models_dir")
RESTART = ("data_dir", "language")


def check_folder(path: Path) -> None:
    """Папка существует (создаётся) и доступна на запись; иначе OSError."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".write-test-{os.getpid()}"
    probe.write_bytes(b"")
    probe.unlink()


class AppConfig:
    def __init__(self, app_dir: Path, data_dir: Path) -> None:
        self.app_dir, self.data_dir = Path(app_dir), Path(data_dir)

    def get(self) -> dict[str, Any]:
        current = st.load(self.app_dir)
        return {
            "settings": asdict(current),
            "resolved": {
                name: str(current.resolve(self.app_dir, name).resolve()) for name in FOLDERS
            },
            "current_data_dir": str(self.data_dir.resolve()),
            "languages": list(UI_LANGUAGES),
        }

    def update(self, body: dict[str, Any]) -> dict[str, Any]:
        """Изменить переданные поля; всё проверяется до записи, ошибка — ничего не меняет."""
        current = st.load(self.app_dir)
        updated = current
        for name, value in body.items():
            if name in FOLDERS:
                if not isinstance(value, str) or not value.strip():
                    raise ChatError("settings.invalid", field=name, detail="empty")
                path = Path(value).expanduser()
                folder = path if path.is_absolute() else self.app_dir / path
                try:
                    check_folder(folder)
                except OSError as error:
                    raise ChatError(
                        "settings.folder_no_access", field=name, path=str(folder), detail=str(error)
                    ) from None
                updated = updated.with_path(self.app_dir, name, folder)
            elif name == "language" and value not in UI_LANGUAGES:
                raise ChatError("settings.invalid", field=name, detail=str(value))
            else:
                try:
                    updated = replace(updated, **{name: value}).validate()
                except (TypeError, ValueError) as error:
                    raise ChatError("settings.invalid", field=name, detail=str(error)) from None
        st.save(self.app_dir, updated)
        changed = [n for n in RESTART if getattr(updated, n) != getattr(current, n)]
        return {**self.get(), "restart_required": changed}
