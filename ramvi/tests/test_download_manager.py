"""4.3 MOD-02/MOD-04: менеджер загрузок — очередь, пауза, обрыв сети, отмена, перезапуск."""

import hashlib
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.catalog import Catalog, CatalogEntry
from test_chat import ENGINE, TOKEN, Client
from test_onboarding import write_gguf

PIECE = 64 * 1024


class FileServer:
    """Отдаёт файлы с поддержкой Range; умеет медленно и с обрывами."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.ranges: list[str | None] = []
        self.delay = 0.0  # пауза на каждый кусок PIECE
        self.drops = 0  # сколько ближайших ответов оборвать на середине
        self.redirect: dict[str, str] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path in owner.redirect:
                    self.send_response(302)
                    self.send_header("Location", owner.redirect[self.path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = owner.files.get(self.path)
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                header = self.headers.get("Range")
                owner.ranges.append(header)
                start = int(header.split("=")[1].rstrip("-")) if header else 0
                self.send_response(206 if header else 200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                drop = owner.drops > 0
                if drop:
                    owner.drops -= 1
                try:
                    for offset in range(0, len(body), PIECE):
                        if drop and offset >= len(body) // 2:
                            self.connection.close()  # обрыв сети посреди файла
                            return
                        self.wfile.write(body[offset : offset + PIECE])
                        time.sleep(owner.delay)
                except OSError:
                    pass

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def gguf_bytes(tmp: Path, name: str = "Qwen3 8B") -> bytes:
    data = write_gguf(tmp / "src.gguf", name=name).read_bytes()
    return data + bytes(range(256)) * 16384  # ~4 МБ: загрузчик читает по 1 МБ, успеть нажать паузу


class Provider:
    name = "test"

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self.items = entries

    def entries(self) -> list[CatalogEntry]:
        return self.items


def entry(base: str, path: str, data: bytes, quant: str = "Q4_K_M", **changes: Any) -> CatalogEntry:
    fields: dict[str, Any] = dict(
        id=f"test:{path}",
        provider="test",
        name=f"model {quant}",
        source="org/model",
        version="r1",
        url=base + path,
        page=base,
        file_name=path.strip("/"),
        bytes=len(data),
        checksum="sha256:" + hashlib.sha256(data).hexdigest(),
        format="gguf",
        architecture="qwen3",
        quantization=quant,
        license="apache-2.0",
        capabilities=("chat",),
    )
    fields.update(changes)
    return CatalogEntry(**fields)


class App:
    """Сервис с каталогом из тестового провайдера."""

    def __init__(self, root: Path, entries: list[CatalogEntry], retries: tuple[float, ...]) -> None:
        self.root, self.entries, self.retries = root, entries, retries
        self.start()

    def start(self) -> None:
        catalog = Catalog(self.root / "data", [Provider(self.entries)])
        catalog.refresh()
        self.server, self.engines = build(
            self.root / "data",
            TOKEN,
            engine_command=ENGINE,
            app_dir=self.root / "app",
            catalog=catalog,
            download_options={"retries": self.retries},
        )
        self.server.start()
        self.client = Client(self.server.port)

    def stop(self) -> None:
        self.server.close()
        self.engines.stop()

    def download(self, download_id: str) -> dict[str, Any]:
        found = self.client.ok("GET", "/downloads")["downloads"]
        return next(d for d in found if d["id"] == download_id)

    def wait(self, download_id: str, test: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = self.download(download_id)
            if test(current):
                return current
            time.sleep(0.02)
        raise AssertionError(f"timeout: {self.download(download_id)}")

    def until(self, download_id: str, *states: str) -> dict[str, Any]:
        return self.wait(download_id, lambda d: d["state"] in states)

    @property
    def models_dir(self) -> Path:
        return self.root / "app" / "models"


@pytest.fixture
def files() -> Iterator[FileServer]:
    server = FileServer()
    yield server
    server.close()


def make(tmp_path: Path, files: FileServer, *extra: CatalogEntry, retries=(0.05,)) -> App:  # type: ignore[no-untyped-def]
    data = gguf_bytes(tmp_path)
    files.files["/Qwen3-8B-Q4_K_M.gguf"] = data
    return App(tmp_path, [entry(files.base, "/Qwen3-8B-Q4_K_M.gguf", data), *extra], retries)


def test_download_from_catalog_installs_model_with_origin(
    tmp_path: Path, files: FileServer
) -> None:
    app = make(tmp_path, files)
    try:
        started = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})
        done = app.until(started["download"]["id"], "done", "failed")

        assert done["state"] == "done", done
        assert done["bytes_done"] == done["total_bytes"] and done["model_id"]
        assert (app.models_dir / "Qwen3-8B-Q4_K_M.gguf").is_file()
        assert not list(app.models_dir.glob("*.part"))
        models = {m["id"]: m for m in app.client.ok("GET", "/models")["models"]}
        assert done["model_id"] in models
        stored = app.server.chat.models_repo.get(done["model_id"])
        assert stored.provider == "test" and stored.license == "apache-2.0"
        assert stored.sha256 == hashlib.sha256(files.files["/Qwen3-8B-Q4_K_M.gguf"]).hexdigest()
        assert stored.capabilities["catalog_id"] == "test:/Qwen3-8B-Q4_K_M.gguf"
        assert (
            stored.path == "models/Qwen3-8B-Q4_K_M.gguf"
        )  # внутри папки приложения — относительно
    finally:
        app.stop()


def test_pause_and_resume_continue_by_range(tmp_path: Path, files: FileServer) -> None:
    files.delay = 0.02
    app = make(tmp_path, files)
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
            "download"
        ]["id"]
        app.wait(did, lambda d: d["state"] == "running" and d["bytes_done"] > 0)
        app.client.ok("POST", f"/downloads/{did}/pause")
        paused = app.until(did, "paused")
        part = app.models_dir / "Qwen3-8B-Q4_K_M.gguf.part"
        assert part.is_file() and 0 < part.stat().st_size < paused["total_bytes"]
        assert not (app.models_dir / "Qwen3-8B-Q4_K_M.gguf").exists()  # частичный — не модель
        assert app.client.ok("GET", "/models")["models"] == []

        files.delay = 0
        app.client.ok("POST", f"/downloads/{did}/resume")
        assert app.until(did, "done", "failed")["state"] == "done"
        assert files.ranges[0] is None and files.ranges[-1] is not None  # продолжение по Range
    finally:
        app.stop()


def test_network_drop_is_retried_from_where_it_stopped(tmp_path: Path, files: FileServer) -> None:
    files.drops = 1
    app = make(tmp_path, files)
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
            "download"
        ]["id"]
        assert app.until(did, "done", "failed", "paused")["state"] == "done"
        assert len(files.ranges) == 2 and files.ranges[1] is not None
    finally:
        app.stop()


def test_network_gone_ends_in_pause_with_reason_then_resumes(
    tmp_path: Path, files: FileServer
) -> None:
    files.drops = 99
    app = make(tmp_path, files, retries=(0.05, 0.05))
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
            "download"
        ]["id"]
        paused = app.until(did, "paused", "done", "failed")
        assert paused["state"] == "paused"
        assert paused["error"] == {"code": "download.interrupted", "params": {}}
        assert len(files.ranges) == 3  # первая попытка + 2 повтора
        assert app.client.ok("GET", "/models")["models"] == []

        files.drops = 0  # сеть вернулась
        app.client.ok("POST", f"/downloads/{did}/resume")
        done = app.until(did, "done", "failed")
        assert done["state"] == "done" and done["error"] is None
    finally:
        app.stop()


def test_cancel_removes_partial_file(tmp_path: Path, files: FileServer) -> None:
    files.delay = 0.02
    app = make(tmp_path, files)
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
            "download"
        ]["id"]
        app.wait(did, lambda d: d["bytes_done"] > 0)
        app.client.ok("POST", f"/downloads/{did}/cancel")
        assert app.until(did, "cancelled")["state"] == "cancelled"
        assert list(app.models_dir.iterdir()) == []
        assert app.client.ok("GET", "/models")["models"] == []
    finally:
        app.stop()


def test_wrong_hash_is_not_activated(tmp_path: Path, files: FileServer) -> None:
    data = gguf_bytes(tmp_path)
    files.files["/bad.gguf"] = data
    bad = entry(files.base, "/bad.gguf", data, checksum="sha256:" + "0" * 64)
    app = make(tmp_path, files, bad)
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/bad.gguf"})["download"]["id"]
        failed = app.until(did, "failed", "done")
        assert failed["error"]["code"] == "download.hash_mismatch"
        assert not (app.models_dir / "bad.gguf").exists()
        assert not (app.models_dir / "bad.gguf.part").exists()
        assert app.client.ok("GET", "/models")["models"] == []
    finally:
        app.stop()


def test_file_that_is_not_a_model_does_not_become_one(tmp_path: Path, files: FileServer) -> None:
    junk = b"not a gguf" * 1000
    files.files["/junk.gguf"] = junk
    app = make(tmp_path, files, entry(files.base, "/junk.gguf", junk))
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/junk.gguf"})["download"]["id"]
        failed = app.until(did, "failed", "done")
        assert failed["state"] == "failed" and failed["error"]["code"] == "import.not_gguf"
        assert app.client.ok("GET", "/models")["models"] == []
    finally:
        app.stop()


def test_only_supported_catalog_entries_can_be_downloaded(
    tmp_path: Path, files: FileServer
) -> None:
    data = gguf_bytes(tmp_path)
    files.files["/q8.gguf"] = data
    app = make(tmp_path, files, entry(files.base, "/q8.gguf", data, quant="Q8_0"))
    try:
        for entry_id in ("test:/q8.gguf", "nope", "test:/../../x"):
            status, body = app.client.request("POST", "/downloads", {"entry_id": entry_id})
            assert status == 404 and body["error"]["code"] == "download.unknown_entry"
        assert files.ranges == []  # ни одного запроса к файлу
        status, body = app.client.request("POST", "/downloads/0123456789abcdef/pause")
        assert status == 404 and body["error"]["code"] == "download.not_found"
    finally:
        app.stop()


def test_queue_runs_one_at_a_time_and_start_is_idempotent(
    tmp_path: Path, files: FileServer
) -> None:
    files.delay = 0.005
    second = gguf_bytes(tmp_path, name="Qwen3 4B")
    files.files["/Qwen3-4B-Q4_K_M.gguf"] = second
    app = make(tmp_path, files, entry(files.base, "/Qwen3-4B-Q4_K_M.gguf", second))
    try:
        first = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})
        again = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})
        assert again["download"]["id"] == first["download"]["id"]
        other = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-4B-Q4_K_M.gguf"})
        a, b = first["download"]["id"], other["download"]["id"]
        app.wait(a, lambda d: d["state"] == "running")
        assert app.download(b)["state"] == "queued"
        assert app.until(a, "done")["state"] == "done"
        assert app.until(b, "done")["state"] == "done"
        assert len(app.client.ok("GET", "/models")["models"]) == 2
    finally:
        app.stop()


def test_restart_keeps_partial_download_paused(tmp_path: Path, files: FileServer) -> None:
    files.delay = 0.02
    app = make(tmp_path, files)
    did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
        "download"
    ]["id"]
    app.wait(did, lambda d: d["bytes_done"] > 0)
    app.stop()  # выход посреди загрузки
    part = app.models_dir / "Qwen3-8B-Q4_K_M.gguf.part"
    assert part.is_file()

    files.delay = 0
    app.start()
    try:
        after = app.download(did)
        assert after["state"] == "paused"  # сам не качает после запуска
        assert after["bytes_done"] == part.stat().st_size
        app.client.ok("POST", f"/downloads/{did}/resume")
        assert app.until(did, "done", "failed")["state"] == "done"
        assert files.ranges[-1] is not None
        assert app.download(did)["model_id"]  # модель находится и после перезапуска
    finally:
        app.stop()


def test_redirect_keeps_range_and_must_stay_https(tmp_path: Path, files: FileServer) -> None:
    data = gguf_bytes(tmp_path)
    files.files["/cdn/file"] = data
    files.redirect["/resolve/x.gguf"] = files.base + "/cdn/file"
    files.redirect["/evil/y.gguf"] = "http://example.org/y.gguf"  # переадресация не на HTTPS
    app = make(
        tmp_path,
        files,
        entry(files.base, "/resolve/x.gguf", data, file_name="x.gguf"),
        entry(files.base, "/evil/y.gguf", data, file_name="y.gguf"),
    )
    try:
        ok = app.client.ok("POST", "/downloads", {"entry_id": "test:/resolve/x.gguf"})
        assert app.until(ok["download"]["id"], "done", "failed")["state"] == "done"
        evil = app.client.ok("POST", "/downloads", {"entry_id": "test:/evil/y.gguf"})
        failed = app.until(evil["download"]["id"], "failed", "done", "paused")
        assert failed["state"] == "failed"
        assert failed["error"]["code"] == "download.only_https"
    finally:
        app.stop()


def test_crash_mid_download_comes_back_paused(tmp_path: Path, files: FileServer) -> None:
    """Процесс убит: в базе осталось «running» — после запуска это пауза, не автозагрузка."""
    from ramvi_storage.repositories import Downloads
    from ramvi_storage.schema import open_core

    app = make(tmp_path, files)
    app.stop()
    with open_core(tmp_path / "data") as db:
        record = Downloads(db).create(
            "test:/Qwen3-8B-Q4_K_M.gguf",
            files.base + "/Qwen3-8B-Q4_K_M.gguf",
            str(app.models_dir / "Qwen3-8B-Q4_K_M.gguf.part"),
            hashlib.sha256(files.files["/Qwen3-8B-Q4_K_M.gguf"]).hexdigest(),
        )
        Downloads(db).update(record.id, state="running")
    app.start()
    try:
        time.sleep(0.3)
        assert app.download(record.id)["state"] == "paused"
        assert files.ranges == []
    finally:
        app.stop()


@pytest.mark.parametrize("action, state", [("pause", "paused"), ("cancel", "cancelled")])
def test_pause_or_cancel_while_waiting_to_retry(
    tmp_path: Path, files: FileServer, action: str, state: str
) -> None:
    files.drops = 99
    app = make(tmp_path, files, retries=(30.0,))  # долгое ожидание перед повтором
    try:
        did = app.client.ok("POST", "/downloads", {"entry_id": "test:/Qwen3-8B-Q4_K_M.gguf"})[
            "download"
        ]["id"]
        app.wait(did, lambda d: d["error"] is not None and d["state"] == "running")
        started = time.monotonic()
        app.client.ok("POST", f"/downloads/{did}/{action}")
        final = app.until(did, state)
        assert time.monotonic() - started < 5  # не ждали 30 секунд
        assert len(files.ranges) == 1  # повтора не было
        part = app.models_dir / "Qwen3-8B-Q4_K_M.gguf.part"
        assert part.exists() == (state == "paused")
        assert final["state"] == state
    finally:
        app.stop()
