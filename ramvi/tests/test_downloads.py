import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ramvi_service import downloads as dl

DATA = bytes(range(256)) * 12000  # ~3 МБ
SHA = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    ranges = True

    def do_GET(self):
        start = 0
        header = self.headers.get("Range")
        if header and self.ranges:
            start = int(header.split("=")[1].rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        body = DATA[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.server.requests.append(header)
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/m.gguf"
    srv.shutdown()
    Handler.ranges = True


def test_download_activates_only_after_hash(server, tmp_path):
    _, url = server
    path = dl.Download(url, tmp_path, "m.gguf", SHA).run()
    assert path.read_bytes() == DATA and not (tmp_path / "m.gguf.part").exists()


def test_pause_then_resume_by_range(server, tmp_path):
    srv, url = server
    job = dl.Download(url, tmp_path, "m.gguf", SHA)
    job.on_progress = lambda p: p.done >= 1024 * 1024 and job.pause()
    with pytest.raises(dl.Paused):
        job.run()
    part = tmp_path / "m.gguf.part"
    kept = part.stat().st_size
    assert 0 < kept < len(DATA) and not (tmp_path / "m.gguf").exists()
    job.on_progress = None
    assert job.run().read_bytes() == DATA
    assert srv.requests == [None, f"bytes={kept}-"]


def test_server_without_range_restarts(server, tmp_path):
    _, url = server
    Handler.ranges = False
    (tmp_path / "m.gguf.part").write_bytes(DATA[:1000])
    assert dl.Download(url, tmp_path, "m.gguf", SHA).run().read_bytes() == DATA


def test_bad_hash_not_activated(server, tmp_path):
    _, url = server
    with pytest.raises(dl.DownloadError, match="хеш"):
        dl.Download(url, tmp_path, "m.gguf", "0" * 64).run()
    assert list(tmp_path.iterdir()) == []


def test_no_space(server, tmp_path, monkeypatch):
    _, url = server
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    with pytest.raises(dl.DownloadError, match="мало места"):
        dl.Download(url, tmp_path, "m.gguf", SHA, size=len(DATA)).run()


def test_speed_limit(server, tmp_path):
    _, url = server
    started = time.monotonic()
    dl.Download(url, tmp_path, "m.gguf", SHA, limit_bps=len(DATA) // 2).run()
    assert time.monotonic() - started >= 1.8


@pytest.mark.parametrize("name", ["../x.gguf", "/etc/x", "C:\\x.gguf", "a/../../x", "..\\x", ""])
def test_path_protection(tmp_path, name):
    with pytest.raises(dl.DownloadError):
        dl.safe_target(tmp_path, name)


def test_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (models / "link").symlink_to(outside)
    with pytest.raises(dl.DownloadError):
        dl.safe_target(models, "link/x.gguf")
    assert dl.safe_target(models, "sub/x.gguf") == (models / "sub/x.gguf").resolve()
