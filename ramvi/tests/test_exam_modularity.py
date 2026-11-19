"""8.9 (CORE-MOD-09): экзамен модульности на настоящем сервисе.

Состояния второго модуля `probe` (свой пакет, свой код, своя база, настоящий процесс):
нет модуля (и нет Character) → установлен → активен (держит микрофон и подписку на события)
→ suspend/resume → падение процесса → снова активен → deactivate → обновление и откат →
выключен/включён → удалён (данные сохранены). После каждого шага: аренды устройств и подписки
модуля освобождены, процесс модуля завершён, соседний модуль `other` работает тем же
процессом, чат отвечает. Память процесса сервиса и задержка чата — относительно исходного
состояния (это замер одного прогона на тестовом движке, не обещание «нулевой стоимости»).
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.contracts.modules import Manifest
from ramvi_service.modules.packages import build_package
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402

CODE = """
class Probe:
    def initialize(self, context):
        self.services = context["services"]
    def activate(self): pass
    def deactivate(self): pass
    def suspend(self): pass
    def resume(self): pass
    def dispose(self): pass
    def checkpoint(self): return {}
    def grab(self):
        lease = self.services.call("resources.acquire", {"amounts": {"audio_input": 1}})
        sub = self.services.call("events.subscribe", {})
        return {"lease": lease["lease_id"], "subscription": sub["subscription_id"]}
    def add(self, text):
        return dict(self.services.call("data.execute",
            {"sql": "INSERT INTO notes(text) VALUES (?)", "params": [text]}))
    def rows(self):
        return dict(self.services.call("data.query", {"sql": "SELECT text FROM notes ORDER BY id"}))
    def version(self): return {"version": VERSION}
    def crash(self):
        import os
        os._exit(13)

def create():
    return Probe()
"""
V1 = {"0001_init.sql": "CREATE TABLE notes(id INTEGER PRIMARY KEY, text TEXT NOT NULL);"}
V2 = V1 | {"0002_tag.sql": "ALTER TABLE notes ADD COLUMN tag TEXT DEFAULT 'x';"}


def probe_package(root: Path, version: str, migrations: dict[str, str]) -> Path:
    folder = root / f"probe-{version}"
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "code" / "probe_module").mkdir(parents=True)
    (folder / "code" / "probe_module" / "__init__.py").write_text(f"VERSION = {version!r}\n" + CODE)
    manifest = {
        "module_id": "probe",
        "version": version,
        "entrypoint": "probe_module:create",
        "permissions": ["data", "resources", "events"],
        "resource_limits": {"audio_input": 1},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "migrations").mkdir()
    for name, sql in migrations.items():
        (folder / "migrations" / name).write_text(sql)
    target = root / f"probe-{version}.zip"
    build_package(folder, target)
    return target


def chat_ms(ui: Client, chat: str, n: int = 5) -> float:
    times = []
    for i in range(n):
        started = time.perf_counter()
        events = ui.events(f"/chats/{chat}/send", {"text": f"проверка {i}"})
        assert events[-1]["finish"] == "stop"
        times.append((time.perf_counter() - started) * 1000)
    return float(sorted(times)[len(times) // 2])


def rss_mb() -> float:
    """Текущая RSS процесса сервиса (ps есть и на Linux, и на macOS)."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.split()[0]) / 1024


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def worker_pid(host: Any, module_id: str) -> int:
    worker = host.status(module_id)[module_id]["worker"]
    assert worker is not None and worker.get("pid"), worker
    return int(worker["pid"])


@pytest.fixture
def service(tmp_path: Path) -> Any:
    app, data = tmp_path / "app", tmp_path / "app" / "data"
    add_models(data, "model-a")
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    yield server, Client(server.port), tmp_path
    server.close()
    engines.stop()


def test_modularity_exam(service: Any) -> None:
    server, ui, tmp_path = service
    host = server.packages.host
    report: dict[str, Any] = {}

    def module_free() -> None:
        """Модуль probe ничего не держит: ни аренд, ни подписок, ни процесса."""
        assert host.resources.status()["leases"] == 0, host.resources.status()
        assert host.resources.status()["used"].get("audio_input", 0) == 0
        assert not any(s.owner.module == "probe" for s in host.events._subscriptions.values())

    # 0. Без модулей (и без Character): чат работает — исходные замеры
    assert "character" not in host.status() and "probe" not in host.status()
    chat = ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
    chat_ms(ui, chat, 2)  # прогрев движка
    base_ms, base_rss = chat_ms(ui, chat), rss_mb()
    module_pids: set[int] = set()
    host.register(Manifest("other", "1.0.0", "ramvi.tests.module_fixture:create"))
    host.activate("other")
    other_worker = host.status("other")["other"]["worker"]

    # 1. Установлен (процесс не запущен), 2. активен — держит микрофон и подписку
    ui.ok("POST", "/modules/install", {"path": str(probe_package(tmp_path, "1.0.0", V1))})
    assert host.status("probe")["probe"]["state"] == "registered"
    host.activate("probe")
    module_pids.add(worker_pid(host, "probe"))
    held = host.invoke("probe", "grab")
    assert held["lease"] and host.resources.status()["used"]["audio_input"] == 1
    host.invoke("probe", "add", {"text": "до сбоя"})
    report["active_rss_mb"] = round(rss_mb() - base_rss, 1)

    # 3. suspend/resume: данные на месте, модуль снова отвечает
    host.suspend("probe")
    host.resume("probe")
    assert host.invoke("probe", "rows")["rows"] == [["до сбоя"]]

    # 4. Падение процесса модуля: его аренды и подписки освобождены, сосед и чат работают
    with pytest.raises(RuntimeError):
        host.invoke("probe", "crash")
    deadline = time.monotonic() + 10
    while host.resources.status()["leases"] and time.monotonic() < deadline:
        time.sleep(0.05)
    module_free()
    assert host.status("other")["other"]["worker"] == other_worker
    assert chat_ms(ui, chat, 1) > 0

    # 5. Снова активен, затем deactivate: всё освобождено, процесс завершён
    host.activate("probe")
    pid = worker_pid(host, "probe")
    module_pids.add(pid)
    host.invoke("probe", "grab")
    host.deactivate("probe")
    module_free()
    assert host.status("probe")["probe"]["worker"] is None

    # 6. Обновление 1.1.0 (миграция) и откат к 1.0.0: данные своей версии, сосед не тронут
    ui.ok("POST", "/modules/install", {"path": str(probe_package(tmp_path, "1.1.0", V2))})
    host.activate("probe")
    assert host.invoke("probe", "version") == {"version": "1.1.0"}
    assert host.invoke("probe", "rows")["rows"] == [["до сбоя"]]
    ui.ok("POST", "/modules/probe/rollback")
    host.activate("probe")
    assert host.invoke("probe", "version") == {"version": "1.0.0"}
    assert host.invoke("probe", "rows")["rows"] == [["до сбоя"]]
    assert host.status("other")["other"]["worker"] == other_worker

    # 7. Выключен: процесса и регистрации нет; включён — данные на месте
    host.invoke("probe", "grab")
    ui.ok("POST", "/modules/probe/disable")
    module_free()
    assert "probe" not in host.status()
    ui.ok("POST", "/modules/probe/enable")
    host.activate("probe")
    assert host.invoke("probe", "rows")["rows"] == [["до сбоя"]]

    # 8. Удалён без purge: код ушёл, данные сохранены до явного выбора
    host.invoke("probe", "grab")
    ui.ok("POST", "/modules/probe/uninstall", {"purge": False})
    module_free()
    assert (host.root / "probe" / "data.sqlite3").exists()
    assert not (server.packages.dir / "probe").exists()

    # Итог: сосед работает тем же процессом, лишних процессов нет, чат не стал медленнее
    assert host.status("other")["other"]["state"] == "active"
    assert host.status("other")["other"]["worker"] == other_worker
    module_pids.add(worker_pid(host, "other"))
    host.deactivate("other")
    deadline = time.monotonic() + 10
    while any(alive(p) for p in module_pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not [p for p in module_pids if alive(p)], "остались процессы модулей"
    report["module_processes_checked"] = len(module_pids)
    after_ms = chat_ms(ui, chat)
    report |= {
        "chat_ms_before": round(base_ms, 1),
        "chat_ms_after": round(after_ms, 1),
        "rss_mb_after": round(rss_mb() - base_rss, 1),
    }
    print("modularity", json.dumps(report, ensure_ascii=False))
    assert after_ms < base_ms * 1.5 + 20  # медленнее только в пределах шума замера


def test_character_installed_disabled_enabled_chat_keeps_working(tmp_path: Path) -> None:
    """Настоящий модуль Character: установлен → выключен → включён; чат работает всё время,
    данные Character при выключении не трогаются."""
    root = Path(__file__).resolve().parents[1]
    app, data = tmp_path / "app", tmp_path / "app" / "data"
    shutil.copytree(root / "modules" / "character", app / "modules" / "character")
    add_models(data, "model-a")
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    try:
        ui = Client(server.port)
        host = server.packages.host
        assert host.status("character")["character"]["state"] == "registered"
        chat = ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
        assert ui.events(f"/chats/{chat}/send", {"text": "раз"})[-1]["finish"] == "stop"
        marker = host.root / "character" / "keep.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("данные персонажа")
        assert any(s["id"] == "character" for s in ui.ok("GET", "/sections")["sections"])
        ui.ok("POST", "/modules/character/disable")
        assert "character" not in host.status()
        assert not any(s["id"] == "character" for s in ui.ok("GET", "/sections")["sections"])
        assert ui.events(f"/chats/{chat}/send", {"text": "два"})[-1]["finish"] == "stop"
        assert marker.read_text() == "данные персонажа"
        ui.ok("POST", "/modules/character/enable")
        assert host.status("character")["character"]["state"] == "registered"
        assert any(s["id"] == "character" for s in ui.ok("GET", "/sections")["sections"])
    finally:
        server.close()
        engines.stop()
