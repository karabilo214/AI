"""3.7: экран настроек — API `/settings` поверх `settings.json` папки приложения."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import settings as st
from ramvi_service.app import build
from ramvi_service.app_config import AppConfig
from ramvi_service.chat import ChatError
from test_chat import ENGINE, TOKEN, Client


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    app = tmp_path / "app"
    app.mkdir()
    return AppConfig(app, app / "data")


def code_of(call: Any) -> tuple[str, dict[str, Any]]:
    with pytest.raises(ChatError) as error:
        call()
    return error.value.code, error.value.params


def test_defaults_are_inside_app_folder(config: AppConfig) -> None:
    got = config.get()

    assert got["settings"] == {
        "data_dir": "data",
        "models_dir": "models",
        "language": "ru",
        "api_port": 11435,
        "network_access": False,
        "network_allow": "192.168.0.0/16, 10.0.0.0/8, 172.16.0.0/12",
        "theme": "system",
        "task_mode": "chat",
        "task_priority": "balanced",
        "task_context": 0,
        "voice_device": "",
        "voice_language": "auto",
        "voice_model": "large-v3-turbo-q5_0",
        "voice_backend": "local",
        "voice_api_url": "",
        "voice_api_model": "whisper-1",
        "voice_auto_send": True,
        "attach_max_files": 5,
        "attach_max_file_kb": 128,
        "attach_max_total_kb": 256,
        "kv_cache_ram_mb": 2048,
        "kv_cache_disk_mb": 8192,
    }
    assert got["resolved"]["data_dir"] == str((config.app_dir / "data").resolve())
    assert got["current_data_dir"] == str((config.app_dir / "data").resolve())
    assert got["languages"] == ["ru", "en"]
    assert not (config.app_dir / "settings.json").exists()  # чтение ничего не пишет


def test_update_saves_and_reports_restart(config: AppConfig) -> None:
    got = config.update({"language": "en", "api_port": 12000, "network_access": True})

    assert got["restart_required"] == []  # язык — без перезапуска (3.14)
    saved = json.loads((config.app_dir / "settings.json").read_text(encoding="utf-8"))
    assert (
        saved["language"] == "en" and saved["api_port"] == 12000 and saved["network_access"] is True
    )
    assert config.update({"api_port": 12001})["restart_required"] == []  # язык уже сохранён


def test_theme_applies_without_restart_and_is_checked(config: AppConfig) -> None:
    got = config.update({"theme": "dark"})

    assert got["restart_required"] == [] and got["settings"]["theme"] == "dark"
    code, params = code_of(lambda: config.update({"theme": "blue"}))
    assert code == "settings.invalid" and params["field"] == "theme"
    assert config.get()["settings"]["theme"] == "dark"  # ошибка ничего не записала


def test_folder_inside_app_stored_relative_outside_absolute(
    config: AppConfig, tmp_path: Path
) -> None:
    inside = config.app_dir / "store" / "data2"
    outside = tmp_path / "elsewhere" / "models"

    got = config.update({"data_dir": str(inside), "models_dir": str(outside)})

    assert (
        got["settings"]["data_dir"] == "store/data2"
    )  # переживает перенос папки/смену буквы диска
    assert got["settings"]["models_dir"] == str(outside.resolve())
    assert got["restart_required"] == ["data_dir"]
    assert inside.is_dir() and outside.is_dir()  # созданы и проверены на запись
    assert got["resolved"]["data_dir"] == str(inside.resolve())
    assert (
        config.update({"data_dir": "store/data2"})["restart_required"] == []
    )  # относительный ввод


def test_invalid_values_change_nothing(config: AppConfig) -> None:
    config.update({"api_port": 12000})
    before = (config.app_dir / "settings.json").read_text(encoding="utf-8")

    cases = [
        ({"api_port": 80}, "api_port"),
        ({"api_port": "12000"}, "api_port"),
        ({"api_port": True}, "api_port"),
        ({"network_access": "yes"}, "network_access"),
        ({"language": "uk"}, "language"),  # нет каталога интерфейса
        ({"data_dir": "  "}, "data_dir"),
        ({"color_scheme": "dark"}, "color_scheme"),  # неизвестное поле
        ({"language": "en", "api_port": 1}, "api_port"),  # одна ошибка — не пишется ничего
    ]
    for body, field in cases:
        code, params = code_of(lambda body=body: config.update(body))
        assert (code, params["field"]) == ("settings.invalid", field), body
    assert (config.app_dir / "settings.json").read_text(encoding="utf-8") == before


def test_folder_without_access_is_reported(config: AppConfig, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("не папка", encoding="utf-8")

    code, params = code_of(lambda: config.update({"models_dir": str(blocker / "models")}))

    assert code == "settings.folder_no_access"
    assert params["field"] == "models_dir" and params["path"] == str(blocker / "models")
    assert not (config.app_dir / "settings.json").exists()


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="права POSIX, не root")
def test_read_only_folder_is_reported(config: AppConfig, tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert (
            code_of(lambda: config.update({"data_dir": str(locked)}))[0]
            == "settings.folder_no_access"
        )
    finally:
        locked.chmod(0o700)


def test_settings_survive_service_restart_via_http(tmp_path: Path) -> None:
    app, data = tmp_path / "app", tmp_path / "app" / "data"
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        client = Client(server.port)
        assert client.ok("GET", "/settings")["settings"]["language"] == "ru"
        saved = client.ok("POST", "/settings", {"language": "en", "models_dir": str(app / "m")})
        assert saved["restart_required"] == []  # язык без перезапуска (3.14)
        assert saved["settings"]["models_dir"] == "m"
        status, error = client.request("POST", "/settings", {"api_port": 5})
        assert status == 400 and error["error"]["code"] == "settings.invalid"
        assert error["error"]["params"]["field"] == "api_port"
    finally:
        server.close()
        engines.stop()
    assert st.load(app).language == "en"
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        assert Client(server.port).ok("GET", "/settings")["settings"]["language"] == "en"
    finally:
        server.close()
        engines.stop()


def test_without_app_dir_settings_are_not_exposed(tmp_path: Path) -> None:
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        status, error = Client(server.port).request("GET", "/settings")
        assert status == 404 and error["error"]["code"] == "api.not_found"
    finally:
        server.close()
        engines.stop()


def test_every_settings_code_has_texts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for language in ("ru", "en"):
        catalog = json.loads((root / "i18n" / f"{language}.json").read_text(encoding="utf-8"))
        for key in ("settings.invalid", "settings.folder_no_access"):
            assert key in catalog, (language, key)


def test_kv_cache_sizes_are_checked_and_reach_the_engine(config: AppConfig) -> None:
    """B.7: размеры кэша неактивных чатов — в настройках (0 — не хранить), передаются движку."""
    from ramvi_service.app import engine_environment
    from ramvi_service.chat import ChatError

    assert engine_environment(config.app_dir) == {
        "RAMVI_KV_RAM_MB": "2048",
        "RAMVI_KV_DISK_MB": "8192",
    }
    config.update({"kv_cache_ram_mb": 512, "kv_cache_disk_mb": 0})
    assert engine_environment(config.app_dir) == {"RAMVI_KV_RAM_MB": "512", "RAMVI_KV_DISK_MB": "0"}
    for bad in (-1, True, "1"):
        with pytest.raises(ChatError):
            config.update({"kv_cache_ram_mb": bad})
