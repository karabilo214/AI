"""3.7: настройки приложения в JSON-файле папки данных.

Пути внутри папки приложения хранятся относительными: смена буквы диска
или перенос распакованной папки их не ломает (PORT-01).
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

LANGUAGES = ("ru", "en", "uk")
THEMES = ("system", "light", "dark")  # 3.16: тема окна оболочки
FILE_NAME = "settings.json"


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data"
    models_dir: str = "models"
    language: str = "ru"
    api_port: int = 11435
    network_access: bool = False  # слушать не только 127.0.0.1 — только по явному включению
    theme: str = "system"

    def validate(self) -> "Settings":
        if self.language not in LANGUAGES:
            raise ValueError(f"язык должен быть одним из {LANGUAGES}")
        if (
            isinstance(self.api_port, bool)
            or not isinstance(self.api_port, int)
            or not 1024 <= self.api_port <= 65535
        ):
            raise ValueError("порт API должен быть целым числом 1024–65535")
        if self.theme not in THEMES:
            raise ValueError(f"тема должна быть одной из {THEMES}")
        if not isinstance(self.network_access, bool):
            raise ValueError("сетевой доступ должен быть true/false")
        for name in ("data_dir", "models_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} не может быть пустым")
        return self

    def resolve(self, app_dir: str | os.PathLike[str], name: str) -> Path:
        """Абсолютный путь папки data_dir/models_dir относительно папки приложения."""
        path = Path(getattr(self, name))
        return path if path.is_absolute() else Path(app_dir) / path

    def with_path(
        self, app_dir: str | os.PathLike[str], name: str, path: str | os.PathLike[str]
    ) -> "Settings":
        """Задать папку; внутри папки приложения сохраняется относительный путь."""
        absolute = Path(path).resolve()
        try:
            value = absolute.relative_to(Path(app_dir).resolve()).as_posix() or "."
        except ValueError:
            value = str(absolute)
        changes: dict[str, Any] = {name: value}
        return replace(self, **changes).validate()


def load(data_dir: str | os.PathLike[str]) -> Settings:
    """Нет файла — настройки по умолчанию. Испорченный файл — ошибка, а не тихий сброс."""
    path = Path(data_dir) / FILE_NAME
    if not path.exists():
        return Settings()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидался объект JSON")
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in raw.items() if k in known}).validate()


def save(data_dir: str | os.PathLike[str], settings: Settings) -> Path:
    """Атомарная запись: сбой посреди записи не портит прежний файл."""
    settings.validate()
    return write_json(Path(data_dir), FILE_NAME, asdict(settings))


def write_json(folder: Path, name: str, value: Any) -> Path:
    """Записать JSON атомарно (временный файл рядом + os.replace)."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{Path(name).stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
