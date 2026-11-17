"""4.2 MOD-01: каталог моделей — адаптеры, обязательные поля, фильтр, офлайн-копия."""

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.catalog import Catalog
from ramvi_service.catalog.entry import CatalogEntry, problems
from ramvi_service.catalog.http import FetchError, fetch
from ramvi_service.catalog.providers import HuggingFace, OllamaRegistry, license_id, quantization
from ramvi_service.catalog.support import capabilities, unsupported_reason

SHA = {name: hashlib.sha256(name.encode()).hexdigest() for name in ("q4", "q8", "q5", "moe", "f16")}
APACHE = "                                 Apache License\n   Version 2.0, January 2004\n"


def hf_repo(arch: str = "qwen3", **extra: Any) -> dict[str, Any]:
    return {
        "id": "Qwen/Qwen3-8B-GGUF",
        "sha": "a" * 40,
        "cardData": {"license": "apache-2.0"},
        "gguf": {"architecture": arch, "context_length": 40960},
        "siblings": [
            {"rfilename": "README.md", "size": 10},
            {"rfilename": "Qwen3-8B-Q4_K_M.gguf", "size": 5000, "lfs": {"sha256": SHA["q4"]}},
            {"rfilename": "Qwen3-8B-Q8_0.gguf", "size": 8000, "lfs": {"sha256": SHA["q8"]}},
            {"rfilename": "Qwen3-8B-Q5_K_M.gguf", "size": 6000, "lfs": {"sha256": SHA["q5"]}},
            {
                "rfilename": "big-Q4_K_M-00001-of-00002.gguf",
                "size": 1,
                "lfs": {"sha256": SHA["q4"]},
            },
            {"rfilename": "no-hash-Q4_K_M.gguf", "size": 5000},  # нет SHA-256 — отбрасывается
        ],
        **extra,
    }


def ollama_files(tag: str, family: str, file_type: str, digest: str) -> dict[str, bytes]:
    config = json.dumps(
        {"model_format": "gguf", "model_family": family, "file_type": file_type, "model_type": "8B"}
    ).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": len(config)},
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 777},
                {
                    "mediaType": "application/vnd.ollama.image.license",
                    "digest": "sha256:" + "1" * 64,
                },
            ],
        }
    ).encode()
    model, label = tag.split(":")
    return {
        f"/v2/library/{model}/manifests/{label}": manifest,
        f"/v2/library/{model}/blobs/{config_digest}": config,
        f"/v2/library/{model}/blobs/sha256:{'1' * 64}": APACHE.encode(),
    }


class Fixture:
    """Локальный HTTP вместо Hugging Face и реестра Ollama; записывает запросы."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.down = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                fixture.requests.append(("GET", self.path, dict(self.headers)))
                body = None if fixture.down else fixture.files.get(self.path)
                self.send_response(200 if body is not None else 503)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def web() -> Iterator[Fixture]:
    fixture = Fixture()
    fixture.files["/api/models/Qwen/Qwen3-8B-GGUF?blobs=true"] = json.dumps(hf_repo()).encode()
    fixture.files.update(ollama_files("qwen3:8b", "qwen3", "Q4_K_M", "sha256:" + SHA["q4"]))
    fixture.files.update(ollama_files("qwen3:30b", "qwen3moe", "Q4_K_M", "sha256:" + SHA["moe"]))
    fixture.files.update(ollama_files("qwen3:30b-fp16", "qwen3moe", "F16", "sha256:" + SHA["f16"]))
    yield fixture
    fixture.close()


def providers(web: Fixture) -> list[Any]:
    return [
        HuggingFace(["Qwen/Qwen3-8B-GGUF"], base=web.base),
        OllamaRegistry(["qwen3:8b", "qwen3:30b", "qwen3:30b-fp16"], base=web.base, page=web.base),
    ]


def test_huggingface_adapter_fills_required_fields_pinned_to_revision(web: Fixture) -> None:
    entries = {e.file_name: e for e in HuggingFace(["Qwen/Qwen3-8B-GGUF"], base=web.base).entries()}

    assert set(entries) == {
        "Qwen3-8B-Q4_K_M.gguf",
        "Qwen3-8B-Q8_0.gguf",
        "Qwen3-8B-Q5_K_M.gguf",
        "no-hash-Q4_K_M.gguf",
    }  # части разбитого файла не берутся
    q4 = entries["Qwen3-8B-Q4_K_M.gguf"]
    assert problems(q4) == []  # http допустим только у 127.0.0.1 (тестовый сервер)
    assert q4.version == "a" * 40
    assert q4.url == f"{web.base}/Qwen/Qwen3-8B-GGUF/resolve/{'a' * 40}/Qwen3-8B-Q4_K_M.gguf"
    assert (q4.bytes, q4.checksum, q4.architecture, q4.quantization, q4.license) == (
        5000,
        "sha256:" + SHA["q4"],
        "qwen3",
        "Q4_K_M",
        "apache-2.0",
    )
    assert q4.context_length == 40960 and "chat" in q4.capabilities
    assert "checksum" in problems(entries["no-hash-Q4_K_M.gguf"])


def test_ollama_adapter_uses_manifest_digest_and_model_layer(web: Fixture) -> None:
    registry = OllamaRegistry(["qwen3:8b"], base=web.base, page=web.base)
    (entry,) = registry.entries()
    manifest = web.files["/v2/library/qwen3/manifests/8b"]

    assert entry.version == "sha256:" + hashlib.sha256(manifest).hexdigest()
    assert entry.checksum == "sha256:" + SHA["q4"]  # digest слоя = SHA-256 файла
    assert entry.url == f"{web.base}/v2/library/qwen3/blobs/sha256:{SHA['q4']}"
    assert (entry.bytes, entry.architecture, entry.quantization, entry.license) == (
        777,
        "qwen3",
        "Q4_K_M",
        "apache-2.0",
    )
    accept = [h for m, p, h in web.requests if p.endswith("/manifests/8b")][0]
    assert accept["Accept"] == "application/vnd.docker.distribution.manifest.v2+json"


def test_unsupported_reasons() -> None:
    def entry(**changes: Any) -> CatalogEntry:
        base = dict(
            id="x",
            provider="p",
            name="n",
            source="s",
            version="v",
            url="https://example.org/f",
            page="https://example.org",
            file_name="f",
            bytes=1,
            checksum="sha256:" + "0" * 64,
            format="gguf",
            architecture="qwen3",
            quantization="Q4_K_M",
            license="apache-2.0",
            capabilities=("chat",),
        )
        return CatalogEntry(**{**base, **changes})

    assert unsupported_reason(entry()) is None
    assert unsupported_reason(entry(quantization="F16")) is None
    assert unsupported_reason(entry(quantization="Q8_0")) == "quantization"
    assert unsupported_reason(entry(quantization="Q6_K")) == "quantization"  # не проверен на железе
    assert unsupported_reason(entry(architecture="qwen3moe", quantization="F16")) == "quantization"
    assert unsupported_reason(entry(architecture="llama")) == "architecture"
    assert unsupported_reason(entry(format="safetensors")) == "format"
    assert problems(entry()) == []
    assert problems(entry(license="", checksum="md5:1", url="http://x", bytes=0)) == [
        "bytes",
        "checksum",
        "license",
        "url",
    ]


class HttpsWeb:
    """Те же ответы, но адреса как у настоящих провайдеров (https) — для полной цепочки."""

    def __init__(self, web: Fixture) -> None:
        self.web = web

    def __call__(self, url: str, headers: Any = None, *, limit: int = 0) -> bytes:
        return fetch(
            url.replace("https://hf.test", self.web.base).replace(
                "https://ollama.test", self.web.base
            ),
            headers,
        )


def real_like(web: Fixture) -> list[Any]:
    get = HttpsWeb(web)
    return [
        HuggingFace(["Qwen/Qwen3-8B-GGUF"], base="https://hf.test", fetch=get),
        OllamaRegistry(
            ["qwen3:8b", "qwen3:30b", "qwen3:30b-fp16"],
            base="https://ollama.test",
            page="https://ollama.test",
            fetch=get,
        ),
    ]


def test_catalog_shows_supported_entries_and_hides_the_rest(tmp_path: Path, web: Fixture) -> None:
    view = Catalog(tmp_path, real_like(web)).refresh()
    shown = sorted(e["name"] for e in view["entries"])

    assert shown == ["Qwen3-8B Q4_K_M", "qwen3:30b Q4_K_M", "qwen3:8b Q4_K_M"]
    assert view["providers"]["huggingface"]["hidden"] == 2  # Q8_0, Q5_K_M (без хеша не сохранён)
    assert view["providers"]["ollama"]["hidden"] == 1  # qwen3moe F16
    assert view["stale"] is False
    for entry in view["entries"]:
        assert entry["capabilities"] and entry["license"] and entry["checksum"]


def test_offline_shows_last_saved_copy(tmp_path: Path, web: Fixture) -> None:
    first = Catalog(tmp_path, real_like(web)).refresh()
    web.down = True

    again = Catalog(tmp_path, real_like(web)).refresh()  # сеть пропала
    fresh_process = Catalog(tmp_path, real_like(web)).view()  # новый процесс, только диск

    assert again["entries"] == first["entries"] == fresh_process["entries"]
    assert again["stale"] is True
    error = again["providers"]["huggingface"]["error"]
    assert error == {"code": "catalog.unavailable", "params": {"provider": "huggingface"}}
    assert again["providers"]["ollama"]["fetched_at"] == first["providers"]["ollama"]["fetched_at"]


def test_one_provider_failing_keeps_the_other(tmp_path: Path, web: Fixture) -> None:
    del web.files["/api/models/Qwen/Qwen3-8B-GGUF?blobs=true"]
    view = Catalog(tmp_path, real_like(web)).refresh()

    assert view["providers"]["huggingface"]["error"] is not None
    assert view["providers"]["ollama"]["error"] is None
    assert {e["provider"] for e in view["entries"]} == {"ollama"}


def test_broken_or_missing_cache_gives_empty_catalog(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path, [])
    assert catalog.view() == {"providers": {}, "entries": [], "stale": False}
    catalog.path.parent.mkdir(parents=True)
    catalog.path.write_text("{oops", encoding="utf-8")
    assert catalog.view()["entries"] == []
    catalog.path.write_text(json.dumps({"schema": 99, "providers": {}}), encoding="utf-8")
    assert catalog.view()["entries"] == []


def test_requests_carry_nothing_about_user(tmp_path: Path, web: Fixture) -> None:
    Catalog(tmp_path, real_like(web)).refresh()

    assert web.requests and {m for m, _, _ in web.requests} == {"GET"}
    for _, path, headers in web.requests:
        assert set(headers) <= {"Host", "User-Agent", "Accept", "Accept-Encoding", "Connection"}
        assert headers["User-Agent"] == "RAMVI"
        assert "?" not in path or path.endswith("?blobs=true")


def test_fetch_refuses_plain_http_outside_loopback() -> None:
    with pytest.raises(FetchError):
        fetch("http://huggingface.co/api/models/x/y")
    with pytest.raises(FetchError):
        fetch("file:///etc/passwd")


def test_fetch_limits_response_size(web: Fixture) -> None:
    web.files["/big"] = b"x" * 100
    assert fetch(f"{web.base}/big", limit=100) == b"x" * 100
    with pytest.raises(FetchError):
        fetch(f"{web.base}/big", limit=99)


def test_bad_source_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        HuggingFace(["../etc"])
    with pytest.raises(ValueError):
        OllamaRegistry(["qwen3:8b/../../x"])


def test_parsing_helpers() -> None:
    assert quantization("Qwen3-8B-Q4_K_M.gguf") == "Q4_K_M"
    assert quantization("qwen3-0.6b-q8_0.gguf") == "Q8_0"
    assert quantization("model-f16.gguf") == "F16"
    assert quantization("model-IQ4_XS.gguf") == "IQ4_XS"
    assert quantization("model.gguf") is None
    assert license_id(APACHE) == "apache-2.0"
    assert license_id("MIT License\n\nCopyright") == "mit"
    assert license_id("Tongyi Qianwen LICENSE AGREEMENT") == "other"
    assert license_id("") == ""


def test_capabilities_come_from_engine_map() -> None:
    assert capabilities("gguf", "qwen3") == ("chat", "tools", "json", "draft")
    assert capabilities("safetensors", "qwen3") == ("chat", "tools", "draft")  # JSON — только GGUF
    assert capabilities("gguf", "llama") == ()


def test_ui_api_catalog_routes(tmp_path: Path, web: Fixture) -> None:
    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, Client

    server, engines = build(
        tmp_path / "data",
        TOKEN,
        engine_command=ENGINE,
        catalog=Catalog(tmp_path / "data", real_like(web)),
    )
    server.start()
    try:
        client = Client(server.port)
        assert client.ok("GET", "/catalog")["entries"] == []  # до обновления — сеть не трогается
        assert web.requests == []
        refreshed = client.ok("POST", "/catalog/refresh")
        assert len(refreshed["entries"]) == 3
        assert client.ok("GET", "/catalog")["entries"] == refreshed["entries"]
    finally:
        server.close()
        engines.stop()
