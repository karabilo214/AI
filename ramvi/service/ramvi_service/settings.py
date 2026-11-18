"""3.7: настройки приложения в JSON-файле папки данных.

Пути внутри папки приложения хранятся относительными: смена буквы диска
или перенос распакованной папки их не ломает (PORT-01).

3.14: неверное значение — `SettingsError` с ключом `settings.*` (текст — в каталоге).
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from ramvi_service import web
from ramvi_service.i18n import UserError
from ramvi_service.task_profile import TaskProfile
from ramvi_service.voice.models import LANGUAGES as VOICE_LANGUAGES
from ramvi_service.voice.models import MODELS

VOICE_MODELS = tuple(MODELS)
VOICE_BACKENDS = ("local", "api")

LANGUAGES = ("ru", "en", "uk")
THEMES = ("system", "light", "dark")  # 3.16: тема окна оболочки
FILE_NAME = "settings.json"


class SettingsError(UserError, ValueError):
    """Неверная настройка: ключ сообщения и параметры."""


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data"
    models_dir: str = "models"
    language: str = "ru"
    api_port: int = 11435
    network_access: bool = False  # слушать не только 127.0.0.1 — только по явному включению
    theme: str = "system"
    # 4.5: профиль задачи (режим, приоритет, контекст; 0 — по режиму)
    task_mode: str = "chat"
    task_priority: str = "balanced"
    task_context: int = 0
    # 5.2: голос — микрофон («» — системный), язык распознавания, модель whisper.cpp
    voice_device: str = ""
    voice_language: str = "auto"
    voice_model: str = "large-v3-turbo-q5_0"
    # 5.7: распознавание через сторонний API — по умолчанию выключено; ключ не здесь,
    # а в хранилище секретов ОС (secrets_store.py)
    voice_backend: str = "local"
    voice_api_url: str = ""
    voice_api_model: str = "whisper-1"

    @property
    def task(self) -> TaskProfile:
        return TaskProfile(self.task_mode, self.task_priority, self.task_context)

    def validate(self) -> "Settings":
        if self.language not in LANGUAGES:
            raise SettingsError("settings.bad_language", allowed=", ".join(LANGUAGES))
        if (
            isinstance(self.api_port, bool)
            or not isinstance(self.api_port, int)
            or not 1024 <= self.api_port <= 65535
        ):
            raise SettingsError("settings.bad_port")
        if self.theme not in THEMES:
            raise SettingsError("settings.bad_theme", allowed=", ".join(THEMES))
        if not isinstance(self.network_access, bool):
            raise SettingsError("settings.bad_network")
        if not isinstance(self.voice_device, str):
            raise SettingsError("settings.bad_voice_device")
        if self.voice_language not in VOICE_LANGUAGES:
            raise SettingsError("settings.bad_voice_language", allowed=", ".join(VOICE_LANGUAGES))
        if self.voice_model not in VOICE_MODELS:
            raise SettingsError("settings.bad_voice_model", allowed=", ".join(VOICE_MODELS))
        if self.voice_backend not in VOICE_BACKENDS:
            raise SettingsError("settings.bad_voice_backend", allowed=", ".join(VOICE_BACKENDS))
        if not isinstance(self.voice_api_url, str) or (
            self.voice_api_url and not web.allowed(self.voice_api_url)
        ):
            raise SettingsError("settings.bad_voice_api_url")
        if (
            not isinstance(self.voice_api_model, str)
            or not self.voice_api_model.strip()
            or len(self.voice_api_model) > 100
        ):
            raise SettingsError("settings.bad_voice_api_model")
        problem = self.task.problem()
        if problem is not None:
            raise SettingsError(problem[0], **problem[1])
        for name in ("data_dir", "models_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SettingsError("settings.empty_folder", field=name)
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
    raw = json.loads(path.read_text(encoding="utf-8-sig"))  # Блокнот Windows пишет BOM
    if not isinstance(raw, dict):
        raise SettingsError("settings.bad_file", path=str(path))
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
