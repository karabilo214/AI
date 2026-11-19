"""8.3 (PORT-04): подписанные обновления — подготовка, применение, откат, сохранность данных."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import __version__, ed25519, web
from ramvi_service import updates as up
from ramvi_storage.repositories import Chats
from ramvi_storage.schema import open_core

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_update", ROOT / "packaging" / "make_update.py")
assert _spec is not None and _spec.loader is not None
mu = importlib.util.module_from_spec(_spec)
sys.modules["make_update"] = mu
_spec.loader.exec_module(mu)

SECRET = bytes(range(32))
PUBLIC = ed25519.public_key(SECRET).hex()
OTHER = bytes(range(1, 33))


def test_ed25519_matches_rfc8032_vectors() -> None:
    vectors = [
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            b"",
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        ),
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            bytes([0x72]),
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        ),
    ]
    for secret, public, message, signature in vectors:
        assert ed25519.public_key(bytes.fromhex(secret)).hex() == public
        assert ed25519.sign(bytes.fromhex(secret), message).hex() == signature
        assert ed25519.verify(bytes.fromhex(public), message, bytes.fromhex(signature))
        broken = bytearray(bytes.fromhex(signature))
        broken[10] ^= 1
        assert not ed25519.verify(bytes.fromhex(public), message, bytes(broken))
        assert not ed25519.verify(bytes.fromhex(public), message + b"x", bytes.fromhex(signature))


def write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode() if isinstance(text, str) else text)
    return path


def bundle(folder: Path, version: str, **files: str) -> Path:
    write(folder / "BUILD.json", json.dumps({"version": version}))
    for name, text in files.items():
        write(folder / name.replace("__", "/"), text)
    return folder


def digests(folder: Path, names: list[str]) -> dict[str, str | None]:
    return {
        n: hashlib.sha256((folder / n).read_bytes()).hexdigest() if (folder / n).exists() else None
        for n in names
    }


@pytest.fixture
def app(tmp_path: Path) -> Path:
    """Установленный комплект 0.8.0 с данными пользователя: чат, модель, настройки."""
    folder = bundle(
        tmp_path / "RAMVI",
        "0.8.0",
        **{
            "ramvi__service__core.py": "old core",
            "ramvi__service__gone.py": "removed in 0.9",
            "runtime__python__python312.dll": "old dll",
        },
    )
    write(folder / "RAMVI.exe", "old shell")
    write(folder / "settings.json", '{"language": "ru"}')
    write(folder / "models" / "qwen.gguf", "MODEL BYTES")
    with open_core(folder / "data") as db:
        chat = Chats(db).create("моя история")
        Chats(db).add_message(chat.id, "user", "привет")
    return folder


def package(tmp_path: Path, app: Path, version: str = "0.9.0", secret: bytes = SECRET) -> Path:
    new = bundle(
        tmp_path / f"new-{version}",
        version,
        **{
            "ramvi__service__core.py": "new core",
            "ramvi__service__added.py": "new file",
            "runtime__python__python312.dll": "old dll",
        },
    )
    write(new / "RAMVI.exe", "new shell")
    out = tmp_path / f"update-{version}.zip"
    mu.build_package(new, version, secret, out, previous=app, notes="test")
    return out


FILES = [
    "RAMVI.exe",
    "BUILD.json",
    "ramvi/service/core.py",
    "ramvi/service/gone.py",
    "ramvi/service/added.py",
    "runtime/python/python312.dll",
]


def user_data(app: Path) -> tuple[list[str], bytes, bytes]:
    with open_core(app / "data") as db:
        chats = [c.title for c in Chats(db).recent()]
    return chats, (app / "models" / "qwen.gguf").read_bytes(), (app / "settings.json").read_bytes()


def test_package_contains_only_changes(tmp_path: Path, app: Path) -> None:
    with zipfile.ZipFile(package(tmp_path, app)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        names = sorted(archive.namelist())
    assert sorted(f["path"] for f in manifest["files"]) == [
        "BUILD.json",
        "RAMVI.exe",
        "ramvi/service/added.py",
        "ramvi/service/core.py",
    ]  # неизменившаяся DLL не попала в пакет
    assert manifest["remove"] == ["ramvi/service/gone.py"]
    assert "manifest.sig" in names and "payload/ramvi/service/core.py" in names


def test_stage_apply_and_rollback_keep_user_data(tmp_path: Path, app: Path) -> None:
    before_files, before_data = digests(app, FILES), user_data(app)
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    assert updater.stage(package(tmp_path, app)) == {
        "staged": "0.9.0",
        "files": 4,
        "current": "0.8.0",
    }
    assert digests(app, FILES) == before_files  # подготовка не трогает рабочие файлы
    result = updater.apply()
    assert result == {"applied": "0.9.0", "previous": "0.8.0", "files": 4}
    assert (app / "ramvi/service/core.py").read_text() == "new core"
    assert (app / "ramvi/service/added.py").exists() and not (
        app / "ramvi/service/gone.py"
    ).exists()
    assert (app / "RAMVI.exe").read_text() == "new shell"
    assert up.current_version(app) == "0.9.0"
    assert user_data(app) == before_data
    assert updater.status()["health"] == "pending"
    rolled = updater.rollback()
    assert rolled == {"rolled_back": "0.9.0", "database_restored": False}
    assert digests(app, FILES) == before_files  # всё как было, добавленный файл убран
    assert user_data(app) == before_data  # приёмка PORT-04: история и модели на месте
    assert up.current_version(app) == "0.8.0"


def test_successful_update_is_marked_good(tmp_path: Path, app: Path) -> None:
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    updater.apply()
    updater.mark_good()
    assert updater.status()["health"] == "good" and updater.status()["applied"] == "0.9.0"
    assert not (app / "updates" / "staging" / "0.9.0").exists()


def test_failure_mid_apply_rolls_back(
    tmp_path: Path, app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_files, before_data = digests(app, FILES), user_data(app)
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    real, calls = os.replace, {"n": 0}

    def flaky(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 4 and "updates" not in str(dst):  # посреди замены файлов
            raise PermissionError("file is locked")
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(up.UpdateError) as error:
        updater.apply()
    assert error.value.code == "update.apply_failed"
    monkeypatch.setattr(os, "replace", real)
    assert digests(app, FILES) == before_files and user_data(app) == before_data
    assert updater.status()["health"] == "rolled_back"


def test_rollback_restores_database_changed_by_new_version(tmp_path: Path, app: Path) -> None:
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    updater.apply()
    with open_core(app / "data") as db:  # «новая версия» применила миграцию, прежняя её не знает
        db.execute(
            "INSERT INTO schema_migrations (owner, version, name, checksum, applied_at)"
            " VALUES ('core', 99, 'future', ?, 0)",
            ("0" * 64,),
        )
    assert updater.rollback()["database_restored"] is True
    assert up._schema_version(app / "data" / "ramvi.sqlite3") < 99
    kept = [
        p
        for p in (app / "data" / "backups").glob("ramvi.sqlite3.after-update-0.9.0-*")
        if not p.name.endswith(("-wal", "-shm"))
    ]
    assert kept and up._schema_version(kept[0]) == 99  # изменённая база не удалена
    assert user_data(app)[0] == ["моя история"]


def _signed_zip(tmp_path: Path, manifest: dict[str, Any], files: dict[str, bytes]) -> Path:
    raw = json.dumps(manifest).encode()
    out = tmp_path / "custom.zip"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("manifest.json", raw)
        archive.writestr("manifest.sig", base64.b64encode(ed25519.sign(SECRET, raw)).decode())
        for name, data in files.items():
            archive.writestr("payload/" + name, data)
    return out


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("data/ramvi.sqlite3", "update.forbidden_path"),
        ("models/qwen.gguf", "update.forbidden_path"),
        ("settings.json", "update.forbidden_path"),
        ("updates/state.json", "update.forbidden_path"),
        ("ramvi", "update.forbidden_path"),
        ("../evil.exe", "update.bad_path"),
        ("/etc/passwd", "update.bad_path"),
        ("C:/Windows/x.dll", "update.bad_path"),
        ("ramvi\\..\\x", "update.bad_path"),
    ],
)
def test_package_cannot_touch_data_models_or_escape(
    tmp_path: Path, app: Path, path: str, code: str
) -> None:
    data = b"evil"
    manifest = {
        "schema": up.SCHEMA,
        "product": up.PRODUCT,
        "version": "0.9.0",
        "files": [{"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}],
    }
    with pytest.raises(up.UpdateError) as error:
        up.Updater(app, app / "data", keys=(PUBLIC,)).stage(
            _signed_zip(tmp_path, manifest, {path: data})
        )
    assert error.value.code == code


def test_untrusted_tampered_or_old_packages_are_refused(tmp_path: Path, app: Path) -> None:
    before = digests(app, FILES)
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    foreign = package(tmp_path, app, secret=OTHER)  # подписан чужим ключом
    with pytest.raises(up.UpdateError, match="bad_signature"):
        updater.stage(foreign)
    with pytest.raises(up.UpdateError, match="no_trusted_key"):
        up.Updater(app, app / "data", keys=()).stage(package(tmp_path, app))
    good = package(tmp_path, app)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(tampered, "w") as out:
        for item in source.infolist():
            data = source.read(item)
            if item.filename == "payload/ramvi/service/core.py":
                data = b"malicious"
            out.writestr(item, data)
    with pytest.raises(up.UpdateError) as error:
        updater.stage(tampered)
    assert error.value.code == "update.bad_file"
    with pytest.raises(up.UpdateError, match="not_newer"):
        updater.stage(package(tmp_path, app, version="0.8.0"))
    not_zip = write(tmp_path / "x.zip", "not a zip")
    with pytest.raises(up.UpdateError, match="bad_archive"):
        updater.stage(not_zip)
    assert digests(app, FILES) == before
    assert "staged" not in updater.status()


def test_online_check_needs_consent_and_https(tmp_path: Path, app: Path) -> None:
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    with pytest.raises(up.UpdateError, match="no_consent"):
        updater.check_online(False, "https://updates.example/ramvi")
    with pytest.raises(web.Refused):
        updater.check_online(True, "http://updates.example/ramvi")

    def fetch_for(version: str) -> Any:
        raw = json.dumps(
            {"version": version, "archive": "https://updates.example/ramvi/u.zip"}
        ).encode()
        sig = base64.b64encode(ed25519.sign(SECRET, raw))
        return lambda url: sig if url.endswith(".sig") else raw

    found = updater.check_online(True, "https://updates.example/ramvi", fetch_for("0.9.1"))
    assert found == {
        "version": "0.9.1",
        "archive": "https://updates.example/ramvi/u.zip",
        "notes": "",
    }
    assert updater.check_online(True, "https://updates.example/ramvi", fetch_for("0.8.0")) is None


def test_versions_and_cli(tmp_path: Path, app: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert up.parse_version("0.10.0") > up.parse_version("0.9.9")
    with pytest.raises(up.UpdateError):
        up.parse_version("1.x")
    assert up.current_version(tmp_path / "nowhere") == __version__
    up.main(["status", "--app-dir", str(app), "--data-dir", str(app / "data")])
    assert json.loads(capsys.readouterr().out)["current"] == "0.8.0"
    with pytest.raises(SystemExit):
        up.main(["apply", "--app-dir", str(app), "--data-dir", str(app / "data")])
    assert "update.nothing_staged" in capsys.readouterr().out


def test_ui_routes_report_and_stage(tmp_path: Path, app: Path) -> None:
    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, Client

    server, engines = build(app / "data", TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        ui = Client(server.port)
        assert ui.ok("GET", "/updates")["current"] == "0.8.0"
        pkg = package(tmp_path, app)
        status, error = ui.request("POST", "/updates/stage", {"path": str(pkg)})
        # Приложение доверяет только ключу выпуска RAMVI: тестовый ключ — чужой.
        assert status == 400 and error["error"]["code"] == "update.bad_signature"
        server.updater.keys = (PUBLIC,)
        staged = ui.ok("POST", "/updates/stage", {"path": str(pkg)})
        assert staged["staged"] == "0.9.0" and staged["current"] == "0.8.0"
        assert (app / "ramvi/service/core.py").read_text() == "old core"  # применит оболочка
    finally:
        server.close()
        engines.stop()


def test_cli_in_a_real_process(tmp_path: Path, app: Path) -> None:
    """Так вызывает обновления оболочка: отдельный процесс, ответ — одна строка JSON."""
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(ROOT / p) for p in ("service", "storage")),
    }
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ramvi_service.updates",
            "apply",
            "--app-dir",
            str(app),
            "--data-dir",
            str(app / "data"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert done.returncode == 2
    assert (
        json.loads(done.stdout.strip().splitlines()[-1])["error"]["code"] == "update.nothing_staged"
    )


def test_release_key_is_trusted_and_well_formed() -> None:
    assert len(up.TRUSTED_KEYS) >= 1
    for key in up.TRUSTED_KEYS:
        assert len(bytes.fromhex(key)) == 32
    assert PUBLIC not in up.TRUSTED_KEYS  # тестовый ключ не должен попасть в доверенные


def test_rollback_moves_new_files_aside_instead_of_deleting(
    tmp_path: Path, app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows: откат запускает новый python.exe, и новый python312.dll занят — удалить его
    нельзя, а перенести можно (найдено проверкой на Windows)."""
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    updater.apply()
    assert up.current_version(app) == "0.9.0" and updater.status()["current"] == "0.9.0"

    def locked(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("file is in use")

    monkeypatch.setattr(Path, "unlink", locked)  # любой занятый файл
    assert updater.rollback()["rolled_back"] == "0.9.0"
    monkeypatch.undo()
    assert (app / "ramvi/service/core.py").read_text() == "old core"
    assert not (app / "ramvi/service/added.py").exists()
    assert (app / "updates/backup/0.9.0/rejected/ramvi/service/added.py").exists()
    assert updater.status()["current"] == "0.8.0"


def test_rollback_reports_files_it_could_not_restore(
    tmp_path: Path, app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    updater.apply()
    real = os.replace

    def stuck(src: Any, dst: Any) -> None:
        if str(dst).endswith("RAMVI.exe") and "rejected" not in str(dst):
            raise PermissionError("locked")
        real(src, dst)

    monkeypatch.setattr(os, "replace", stuck)
    with pytest.raises(up.UpdateError) as error:
        updater.rollback()
    assert error.value.code == "update.rollback_incomplete"
    assert error.value.params == {"files": "RAMVI.exe"}
    assert (app / "ramvi/service/core.py").read_text() == "old core"  # остальное вернулось
    monkeypatch.setattr(os, "replace", real)
    assert updater.status()["rollback_failed"] == ["RAMVI.exe"]


def test_new_cycle_forgets_the_previous_rollback(tmp_path: Path, app: Path) -> None:
    """Найдено на Windows: после отката и нового обновления в состоянии висело «откачено»."""
    updater = up.Updater(app, app / "data", keys=(PUBLIC,))
    updater.stage(package(tmp_path, app))
    updater.apply()
    updater.rollback()
    assert updater.status()["rolled_back"] == "0.9.0"
    updater.stage(package(tmp_path, app))
    assert "rolled_back" not in updater.status() and "journal" not in updater.status()
    updater.apply()
    updater.mark_good()
    state = updater.status()
    assert (state["applied"], state["health"], "rolled_back" in state) == ("0.9.0", "good", False)
