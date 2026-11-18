"""8.2 (PORT-03): модели отдельно от приложения — вид диска, предупреждение, перенос папки."""

import shutil
from pathlib import Path

import pytest
from ramvi_service import app_config as config_module
from ramvi_service import settings as st
from ramvi_service.app_config import AppConfig
from ramvi_service.hardware import drive_kind
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core
from test_onboarding import write_gguf
from test_portable import start

MOUNTS = """/dev/nvme0n1p2 / ext4 rw 0 0
//nas/share /mnt/nas cifs rw 0 0
server:/export /mnt/nfs nfs4 rw 0 0
/dev/sdb1 /mnt/nas/local ext4 rw 0 0
/dev/sdc1 /media/user/My\\040Disk vfat rw 0 0
"""


def test_drive_kind_on_each_system() -> None:
    assert drive_kind("\\\\nas\\models", system="Windows") == "network"
    assert drive_kind("//nas/models", system="Windows") == "network"
    assert drive_kind("/Volumes/USB/models", system="Darwin") == "removable"
    assert drive_kind("/Users/me/models", system="Darwin") == "local"
    assert drive_kind("/mnt/nas/qwen", system="Linux", mounts=MOUNTS) == "network"
    assert drive_kind("/mnt/nfs/a/b", system="Linux", mounts=MOUNTS) == "network"
    assert (
        drive_kind("/mnt/nas/local/m", system="Linux", mounts=MOUNTS) == "local"
    )  # ближе — свой диск
    assert drive_kind("/home/me/models", system="Linux", mounts=MOUNTS) == "local"
    assert drive_kind("/mnt/nasty", system="Linux", mounts=MOUNTS) == "local"  # не префикс-путь
    assert drive_kind("x", system="Plan9") == "unknown"


def test_settings_warn_about_slow_models_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = AppConfig(tmp_path, tmp_path / "data")
    assert app.get()["models_dir_warning"] is None
    monkeypatch.setattr(config_module, "drive_kind", lambda path: "network")
    info = app.get()
    assert info["models_dir_kind"] == "network"
    assert info["models_dir_warning"] == "settings.models_dir_network"
    monkeypatch.setattr(config_module, "drive_kind", lambda path: "removable")
    assert app.update({"language": "en"})["models_dir_warning"] == "settings.models_dir_removable"


def test_separate_models_folder_survives_moving_the_app(tmp_path: Path) -> None:
    """Приёмка PORT-03: папка моделей отдельно; переносим приложение — модель не скачивается
    заново и находится по прежнему пути."""
    models = tmp_path / "D" / "models"  # «диск D:» с моделями
    models.mkdir(parents=True)
    write_gguf(models / "qwen3-8b-q4.gguf")
    first = tmp_path / "E" / "RAMVI"
    first.mkdir(parents=True)
    st.save(first, st.Settings(models_dir=str(models)))
    server, engines, client = start(first)
    try:
        assert client.ok("POST", "/models/scan")["added"]
        model_id = client.ok("GET", "/models")["models"][0]["id"]
    finally:
        server.close()
        engines.stop()
    second = tmp_path / "F" / "RAMVI"
    second.parent.mkdir()
    shutil.move(first, second)
    server, engines, client = start(second)
    try:
        assert client.ok("GET", "/settings")["resolved"]["models_dir"] == str(models.resolve())
        assert [m["id"] for m in client.ok("GET", "/models")["models"]] == [model_id]
        assert client.ok("POST", "/models/scan")["added"] == []  # уже подключена, не дубль
        chat = client.ok("POST", "/chats", {"model_id": model_id})["chat"]["id"]
        assert client.events(f"/chats/{chat}/send", {"text": "раз"})[-1]["finish"] == "stop"
    finally:
        server.close()
        engines.stop()
    with open_core(second / "data") as db:
        (stored,) = Models(db).all()
    assert stored.path == str((models / "qwen3-8b-q4.gguf").resolve())  # файл не копировался
    assert not (second / "models").exists() or not any((second / "models").iterdir())
