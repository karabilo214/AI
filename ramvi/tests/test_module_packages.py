"""8.8 (CORE-MOD-08): пакеты модулей — установка, обновление с откатом, отключение, удаление.

Второй, тестовый модуль `notes` живёт только в пакете (свой код в code/, своя SQL-миграция)
и работает в настоящем процессе модуля. Рядом активен модуль `other`: его процесс и данные
не должны меняться, пока `notes` ставят, обновляют, откатывают и удаляют.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Manifest
from ramvi_service.modules.host import ModuleHost
from ramvi_service.modules.packages import (
    ModulePackages,
    api_compatible,
    build_package,
    read_package,
)
from ramvi_service.sections import SectionRegistry

CODE = """
class Notes:
    def initialize(self, context):
        self.services = context["services"]
    def activate(self): pass
    def deactivate(self): pass
    def suspend(self): pass
    def resume(self): pass
    def dispose(self): pass
    def checkpoint(self): return {}
    def add(self, text):
        return dict(self.services.call("data.execute",
            {"sql": "INSERT INTO notes(text) VALUES (?)", "params": [text]}))
    def rows(self):
        return dict(self.services.call("data.query", {"sql": "SELECT * FROM notes ORDER BY id"}))
    def version(self): return {"version": VERSION}

def create():
    return Notes()
"""


def make_module(root: Path, version: str, migrations: dict[str, str], api: str = "1.0") -> Path:
    folder = root / f"notes-{version}"
    shutil.rmtree(folder, ignore_errors=True)
    code = folder / "code" / "notes_module"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text(f"VERSION = {version!r}\n" + CODE, encoding="utf-8")
    manifest = {
        "module_id": "notes",
        "version": version,
        "entrypoint": "notes_module:create",
        "api_version": api,
        "permissions": ["data"],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "migrations").mkdir()
    for name, sql in migrations.items():
        (folder / "migrations" / name).write_text(sql, encoding="utf-8")
    (folder / "resources").mkdir()
    (folder / "resources" / "readme.txt").write_text("notes " + version, encoding="utf-8")
    return folder


V1 = {"0001_init.sql": "CREATE TABLE notes(id INTEGER PRIMARY KEY, text TEXT NOT NULL);"}
V2 = V1 | {"0002_tags.sql": "ALTER TABLE notes ADD COLUMN tag TEXT DEFAULT 'none';"}
BROKEN = V2 | {"0003_broken.sql": "ALTER TABLE no_such_table ADD COLUMN x TEXT;"}


@pytest.fixture
def setup(tmp_path: Path) -> Any:
    app, data = tmp_path / "app" / "modules", tmp_path / "data"
    host = ModuleHost(data, timeout=20)
    registry = SectionRegistry(host)
    packages = ModulePackages(app, registry)
    host.register(Manifest("other", "1.0.0", "ramvi.tests.module_fixture:create"))
    host.activate("other")
    yield packages, host, registry, tmp_path
    host.close()


def package(tmp_path: Path, version: str, migrations: dict[str, str], api: str = "1.0") -> Path:
    target = tmp_path / "pkg" / f"notes-{version}.zip"
    build_package(make_module(tmp_path / "src", version, migrations, api), target)
    return target


def rows(host: ModuleHost) -> list[Any]:
    return list(host.invoke("notes", "rows")["rows"])


def test_package_checks(tmp_path: Path) -> None:
    good = package(tmp_path, "1.0.0", V1)
    assert read_package(good)["module_id"] == "notes"
    assert api_compatible("1.0") and not api_compatible("2.0") and not api_compatible("1.9")
    with pytest.raises(ChatError, match="module.api_incompatible"):
        read_package(package(tmp_path, "1.0.1", V1, api="2.0"))
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            dst.writestr(item, data + b"#" if item.filename.endswith("__init__.py") else data)
    with pytest.raises(ChatError, match="module.integrity"):
        read_package(tampered)
    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(extra, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item))
        dst.writestr("code/evil.py", "x")
    with pytest.raises(ChatError, match="module.integrity"):
        read_package(extra)
    escape = tmp_path / "escape.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(escape, "w") as dst:
        meta = json.loads(src.read("package.json"))
        meta["files"]["../outside.txt"] = {"sha256": "0" * 64, "size": 1}
        for item in src.infolist():
            if item.filename != "package.json":
                dst.writestr(item, src.read(item))
        dst.writestr("package.json", json.dumps(meta))
        dst.writestr("../outside.txt", "x")
    with pytest.raises(ChatError, match="module.package_invalid"):
        read_package(escape)


def test_install_update_rollback_disable_uninstall(setup: Any) -> None:
    packages, host, registry, tmp_path = setup
    other_before = host.status("other")["other"]
    # установка: свой код из пакета, своя база
    assert packages.install(package(tmp_path, "1.0.0", V1))["previous"] is None
    host.activate("notes")
    assert host.invoke("notes", "version") == {"version": "1.0.0"}
    host.invoke("notes", "add", {"text": "первая"})
    # обновление: миграция 2, данные на месте, `other` не перезапускался
    assert packages.install(package(tmp_path, "1.1.0", V2))["previous"] == "1.0.0"
    assert host.invoke("notes", "version") == {"version": "1.1.0"}
    assert [r[1:] for r in rows(host)] == [["первая", "none"]]
    assert host.status("other")["other"]["worker"] == other_before["worker"]
    host.invoke("notes", "add", {"text": "вторая"})
    # неудачное обновление: миграция падает — код и данные 1.1.0 на месте
    with pytest.raises(ChatError, match="module.update_failed"):
        packages.install(package(tmp_path, "1.2.0", BROKEN))
    assert (
        json.loads((packages.dir / "notes" / "manifest.json").read_text(encoding="utf-8"))[
            "version"
        ]
        == "1.1.0"
    )
    host.activate("notes")
    assert host.invoke("notes", "version") == {"version": "1.1.0"}
    assert [r[1] for r in rows(host)] == ["первая", "вторая"]
    # откат к 1.0.0: код и данные той версии (схема без tag)
    assert packages.rollback("notes")["version"] == "1.0.0"
    host.activate("notes")
    assert host.invoke("notes", "version") == {"version": "1.0.0"}
    assert rows(host) == [[1, "первая"]]
    # отключение: процесса и раздела нет, код и данные на месте; включение возвращает
    packages.disable("notes")
    assert "notes" not in host.status() and (packages.dir / "notes" / ".disabled").exists()
    packages.enable("notes")
    host.activate("notes")
    assert rows(host) == [[1, "первая"]]
    # удаление без purge: кода нет, данные остались; повторная установка их видит
    data_db = host.root / "notes" / "data.sqlite3"
    packages.uninstall("notes")
    assert not (packages.dir / "notes").exists() and data_db.exists()
    packages.install(package(tmp_path, "1.0.0", V1))
    host.activate("notes")
    assert rows(host) == [[1, "первая"]]
    # удаление с явным purge — данные стёрты; `other` всё это время работал
    packages.uninstall("notes", purge_data=True)
    assert not data_db.exists()
    assert host.status("other")["other"]["state"] == "active"
    assert host.status("other")["other"]["worker"] == other_before["worker"]


def test_package_routes(tmp_path: Path) -> None:
    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, Client

    app, data = tmp_path / "app", tmp_path / "app" / "data"
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        ui = Client(server.port)
        installed = ui.ok("POST", "/modules/install", {"path": str(package(tmp_path, "1.0.0", V1))})
        assert installed["package"]["version"] == "1.0.0"
        listed = {p["module_id"]: p for p in ui.ok("GET", "/modules/packages")["packages"]}
        assert listed["notes"]["packaged"] and not listed["notes"]["disabled"]
        assert ui.ok("POST", "/modules/notes/disable")["package"]["disabled"] is True
        status, body = ui.request(
            "POST", "/modules/install", {"path": str(tmp_path / "missing.zip")}
        )
        assert status == 400 and body["error"]["code"] == "module.package_invalid"
        status, body = ui.request("POST", "/modules/notes/rollback")
        assert body["error"]["code"] == "module.no_backup"
        assert (
            ui.ok("POST", "/modules/notes/uninstall", {"purge": False})["uninstalled"]["purged"]
            is False
        )
        assert not (app / "modules" / "notes").exists()
    finally:
        server.close()
        engines.stop()
