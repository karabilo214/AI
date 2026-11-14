"""Переход переменных и путей RAMVI без моделей и вычислительных зависимостей."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

config = importlib.import_module("ramvi_engine.config")


@pytest.mark.parametrize("name", ["CPP_THREADS", "RAMVI_CPP_THREADS", "LLM_CPP_THREADS"])
def test_new_environment_name_has_priority_even_when_empty(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Новое имя выбирается по наличию ключа, а не по непустому значению."""
    monkeypatch.setenv("LLM_CPP_THREADS", "8")
    monkeypatch.setenv("RAMVI_CPP_THREADS", "")
    assert config.env(name, "16") == ""
    monkeypatch.setenv("RAMVI_CPP_THREADS", "4")
    assert config.env(name, "16") == "4"


@pytest.mark.parametrize(
    ("suffix", "default"),
    [
        ("NO_CPP", None),
        ("CPP_THREADS", "16"),
        ("VRAM_RESERVE_GIB", "0.9"),
        ("CPU_FRAC", "0.58"),
        ("LAYER_PROFILE", None),
        ("DRAFT_GATE", "0"),
        ("PREFILL", "legacy"),
        ("PREFILL_MIN_TOKENS", "256"),
    ],
)
def test_legacy_environment_values_and_defaults_are_preserved(
    suffix: str, default: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Все существующие переключатели сохраняют старые значения по умолчанию."""
    monkeypatch.delenv(f"RAMVI_{suffix}", raising=False)
    monkeypatch.delenv(f"LLM_{suffix}", raising=False)
    assert config.env(suffix, default) == default
    monkeypatch.setenv(f"LLM_{suffix}", "legacy-value")
    assert config.env(suffix, default) == "legacy-value"
    monkeypatch.setenv(f"RAMVI_{suffix}", "current-value")
    assert config.env(suffix, default) == "current-value"


def test_configured_data_directory_is_lazy_and_relative_to_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Portable-каталог может задаваться относительно папки запуска без записи при импорте."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAMVI_DATA_DIR", "portable-data")
    assert config.data_dir() == tmp_path / "portable-data"
    assert config.cache_dir() == tmp_path / "portable-data" / "cache"
    assert config.work_dir() == tmp_path / "portable-data" / "work"
    assert config.results_dir() == tmp_path / "portable-data" / "results"
    assert not (tmp_path / "portable-data").exists()
    result = config.ensure_dir(config.work_dir())
    assert result == tmp_path / "portable-data" / "work"
    assert result.is_dir()


def test_legacy_data_directory_and_empty_new_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM_DATA_DIR совместим, но не перекрывает явно пустой RAMVI_DATA_DIR."""
    monkeypatch.delenv("RAMVI_DATA_DIR", raising=False)
    monkeypatch.setenv("LLM_DATA_DIR", str(tmp_path / "legacy-data"))
    assert config.data_dir() == tmp_path / "legacy-data"
    monkeypatch.setenv("RAMVI_DATA_DIR", "")
    with pytest.raises(ValueError, match="RAMVI_DATA_DIR"):
        config.data_dir()


@pytest.mark.parametrize(
    ("platform", "parts"),
    [
        ("win32", ("AppData", "Local", "RAMVI")),
        ("darwin", ("Library", "Application Support", "RAMVI")),
        ("linux", (".local", "share", "ramvi")),
    ],
)
def test_platform_defaults(
    platform: str, parts: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Стандартный каталог зависит от платформы, а не от расположения исходников."""
    monkeypatch.delenv("RAMVI_DATA_DIR", raising=False)
    monkeypatch.delenv("LLM_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(config, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert config.data_dir() == tmp_path.joinpath(*parts)
    assert not tmp_path.joinpath(*parts).exists()


@pytest.mark.parametrize(
    ("platform", "variable", "name"),
    [("win32", "LOCALAPPDATA", "RAMVI"), ("linux", "XDG_DATA_HOME", "ramvi")],
)
def test_platform_data_root_is_respected(
    platform: str,
    variable: str,
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Настройки корня ОС сохраняются, если каталог RAMVI явно не задан."""
    monkeypatch.delenv("RAMVI_DATA_DIR", raising=False)
    monkeypatch.delenv("LLM_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setenv(variable, str(tmp_path))
    assert config.data_dir() == tmp_path / name


def test_existing_tuning_cache_is_read_without_moving_or_overwriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Первый запуск читает прежнюю настройку и не создаёт новый холодный кэш."""
    monkeypatch.setenv("RAMVI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    legacy = tmp_path / "home" / ".cache" / "llm_app" / "tune.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"entries": {"known": 1}}', encoding="utf-8")
    before = legacy.read_bytes()
    assert config.tuning_cache_for_read() == legacy
    assert legacy.read_bytes() == before
    assert not (tmp_path / "data").exists()
    current = config.cache_dir() / "tune.json"
    current.parent.mkdir(parents=True)
    current.write_text("new cache", encoding="utf-8")
    assert config.tuning_cache_for_read() == current
    assert legacy.read_bytes() == before


def test_custom_tuning_path_does_not_fall_back_to_unrelated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Явная замена CACHE в инструментах или тестах остаётся изолированной."""
    monkeypatch.setenv("RAMVI_DATA_DIR", str(tmp_path / "data"))
    target = tmp_path / "custom.json"
    assert config.tuning_cache_for_read(target) == target
    assert not target.exists()
