"""Различение HF-папки, локального GGUF и имени Ollama без тяжёлых импортов."""

from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath


def _name(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("нужно непустое имя или путь модели")
    return value


def is_ollama_name(name: str) -> bool:
    """Drive/UNC и существующие локальные пути не являются model:tag."""
    name = _name(name)
    path = Path(name)
    if path.exists() or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive:
        return False
    if name.lower().endswith(".gguf"):
        return False
    model, separator, tag = name.partition(":")
    return bool(separator and model and tag and ":" not in tag)


def is_gguf_model(name: str) -> bool:
    """Каталог HF приоритетен; GGUF распознаётся также по magic без suffix."""
    name = _name(name)
    path = Path(name)
    if path.is_dir():
        return False
    if name.lower().endswith(".gguf"):
        return True
    if path.is_file():
        with path.open("rb") as model:
            return model.read(4) == b"GGUF"
    return is_ollama_name(name)


def resolve_gguf_name(name: str, resolver: Callable[[str], str]) -> str:
    """Вызвать Ollama resolver только для настоящего model:tag, не C:\\path."""
    return resolver(name) if is_ollama_name(name) else name
