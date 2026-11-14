"""Публичная карта возможностей: неизвестные модели и отсутствие GPU."""

import json
import os
import subprocess
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from ramvi_engine.capabilities import capability_catalog, engine_capabilities
from ramvi_engine.server.app import App, make_handler


def test_catalog_is_conservative_and_independent() -> None:
    catalog = capability_catalog()
    assert catalog["runtime_verified"] is False
    assert catalog["unknown_combinations"] == "unsupported"
    assert {(r["format"], r["architecture"]) for r in catalog["rows"]} == {
        ("safetensors", "qwen3"),
        ("gguf", "qwen3"),
        ("gguf", "qwen3moe"),
    }
    for row in catalog["rows"]:
        assert not row["cpu_only"]
        assert not row["tools"]["executes_tools"]
        assert row["json"]["supported"] == (row["format"] == "gguf")
        if row["model_type"] == "moe":
            assert "stream" not in row["execution_modes"]
            assert "native_qmm" in row["requirements"]
    catalog["rows"].clear()
    assert len(capability_catalog()["rows"]) == 3


def test_loaded_models_and_unknown_architecture() -> None:
    for architecture in ["qwen3", "qwen3moe", "unknown"]:
        eng = SimpleNamespace(
            g=SimpleNamespace(meta={"general.architecture": architecture}), maxt=3072
        )
        loaded = engine_capabilities(eng, draft_loaded=True)["loaded_model"]
        assert loaded["recognized"] == (architecture != "unknown")
        assert loaded["context_tokens"] == 3072
        assert loaded["draft_loaded"] is True
        assert loaded["readiness"] == "not_probed"
    assert not engine_capabilities(object())["loaded_model"]["recognized"]
    hf = SimpleNamespace(ck=SimpleNamespace(cfg={"model_type": "qwen3"}))
    assert engine_capabilities(hf)["loaded_model"]["format"] == "safetensors"


def test_cli_without_compute_dependencies() -> None:
    source = Path(__file__).resolve().parents[1]
    script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'triton', 'transformers'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from ramvi_engine.cli import main
sys.argv = ['ramvi-engine', 'capabilities']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout) == capability_catalog()


def test_http_capabilities_requires_auth_and_does_not_generate() -> None:
    app = App(object(), object(), "test", api_key="secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        for headers, expected in [({}, 401), ({"Authorization": "Bearer secret"}, 200)]:
            conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                conn.request("GET", "/capabilities", headers=headers)
                response = conn.getresponse()
                body = json.loads(response.read())
                assert response.status == expected
                if expected == 200:
                    assert body["scope"] == "implementation_catalog"
                    assert not body["loaded_model"]["recognized"]
            finally:
                conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_draft_metadata_tracks_actual_bundle_after_unload_and_restart() -> None:
    from ramvi_engine.lifecycle import EngineSession, ModelBundle

    draft_enabled = False

    def factory() -> ModelBundle:
        engine = SimpleNamespace(maxt=32, ck=SimpleNamespace(cfg={"model_type": "qwen3"}))
        return ModelBundle(engine, object(), object() if draft_enabled else None)

    session = EngineSession(factory)
    # The same stable handle survives every bundle replacement.
    app = App(session, object(), "test", draft=session.draft_handle, api_key="secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:

        def check(expected: bool) -> None:
            assert session.status()["draft_loaded"] is expected
            assert app.metadata()["draft"] is expected
            conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                conn.request("GET", "/capabilities", headers={"Authorization": "Bearer secret"})
                response = conn.getresponse()
                assert response.status == 200
                assert json.loads(response.read())["loaded_model"]["draft_loaded"] is expected
            finally:
                conn.close()

        check(False)
        session.load()
        check(False)  # auto memory plan omitted requested draft
        draft_enabled = True
        session.restart()
        check(True)
        session.unload()
        check(False)
        session.load()
        check(True)
        draft_enabled = False
        session.restart()
        check(False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        session.close()
