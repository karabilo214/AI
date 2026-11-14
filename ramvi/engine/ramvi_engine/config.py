"""Настройки RAMVI, совместимость переменных и каталоги изменяемых данных."""

import os
import sys
from pathlib import Path
from typing import overload


@overload
def env(name: str, default: str) -> str: ...


@overload
def env(name: str, default: None = None) -> str | None: ...


def env(name: str, default: str | None = None) -> str | None:
    """Прочитать RAMVI_<имя>, затем прежнее LLM_<имя>; пустое новое значение значимо."""
    suffix = name.removeprefix("RAMVI_").removeprefix("LLM_")
    current, legacy = f"RAMVI_{suffix}", f"LLM_{suffix}"
    if current in os.environ:
        return os.environ[current]
    return os.environ.get(legacy, default)


def data_dir() -> Path:
    """Вернуть каталог данных без создания; относительная настройка привязана к cwd."""
    configured = env("DATA_DIR")
    if configured is not None:
        if not configured.strip():
            raise ValueError("RAMVI_DATA_DIR не должен быть пустым")
        return Path(configured).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "RAMVI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "RAMVI"
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "ramvi"


def cache_dir() -> Path:
    """Вернуть каталог кэша RAMVI без создания."""
    return data_dir() / "cache"


def work_dir() -> Path:
    """Вернуть каталог временных рабочих файлов RAMVI без создания."""
    return data_dir() / "work"


def results_dir() -> Path:
    """Вернуть каталог результатов проверок RAMVI без создания."""
    return data_dir() / "results"


def ensure_dir(path: Path) -> Path:
    """Создать каталог непосредственно перед записью и вернуть тот же путь."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def tuning_cache_for_read(destination: str | Path | None = None) -> Path:
    """Использовать прежний tuning-кэш, пока нового ещё нет; старый файл не изменять."""
    current = cache_dir() / "tune.json"
    requested = Path(destination) if destination is not None else current
    if requested != current or requested.exists():
        return requested
    legacy = Path.home() / ".cache" / "llm_app" / "tune.json"
    return legacy if legacy.is_file() else requested
