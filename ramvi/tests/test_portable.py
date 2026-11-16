"""3.9 (PORT-01): переносимая папка — относительные пути переживают перенос и смену буквы диска."""

import shutil
from pathlib import Path

import pytest
from ramvi_service.app import build
from ramvi_service.engines import TOKENIZER_ENV, default_engine, resolve_model
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client
from test_onboarding import write_gguf


def start(app: Path):  # type: ignore[no-untyped-def]
    server, engines = build(app / "data", TOKEN, engine_command=ENGINE, app_dir=app)
    server.start()
    return server, engines, Client(server.port)


def test_model_inside_app_folder_survives_moving_whole_folder(tmp_path: Path) -> None:
    first = tmp_path / "E" / "RAMVI"  # «диск E:»
    (first / "models").mkdir(parents=True)
    write_gguf(first / "models" / "qwen3-8b-q4.gguf")
    (tmp_path / "shared").mkdir()
    outside = write_gguf(tmp_path / "shared" / "other.gguf")

    server, engines, client = start(first)
    try:
        inside_id = client.ok(
            "POST", "/models/import", {"path": str(first / "models" / "qwen3-8b-q4.gguf")}
        )["model"]["id"]
        outside_id = client.ok("POST", "/models/import", {"path": str(outside)})["model"]["id"]
        chat = client.ok("POST", "/chats", {"model_id": inside_id})["chat"]["id"]
        assert client.events(f"/chats/{chat}/send", {"text": "раз"})[-1]["finish"] == "stop"
    finally:
        server.close()
        engines.stop()
    with open_core(first / "data") as db:
        stored = {m.id: m.path for m in Models(db).all()}
    assert stored[inside_id] == "models/qwen3-8b-q4.gguf"  # относительный
    assert stored[outside_id] == str(outside.resolve())  # вне папки — как был

    second = tmp_path / "F" / "RAMVI"  # та же папка на «диске F:»
    second.parent.mkdir()
    shutil.move(first, second)
    server, engines, client = start(second)
    try:
        messages = client.ok("GET", f"/chats/{chat}/messages")["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        events = client.events(f"/chats/{chat}/send", {"text": "два"})
        assert events[-1]["finish"] == "stop", events[-1]
        assert "".join(e.get("text", "") for e in events if e["type"] == "text").startswith(
            "[qwen3-8b-q4.gguf]"
        )
        status = client.ok("GET", "/status")["engine"]
        assert status["state"] == "ready" and status["model_id"] == inside_id
        # Повторный импорт того же файла на новом месте — та же запись, не дубль.
        again = client.ok(
            "POST", "/models/import", {"path": str(second / "models" / "qwen3-8b-q4.gguf")}
        )
        assert again["model"]["id"] == inside_id
    finally:
        server.close()
        engines.stop()


def test_resolve_model_paths(tmp_path: Path) -> None:
    model = Model("m", "local", "qwen3", "models/m.gguf", 0.0)

    assert resolve_model(model, tmp_path).path == str((tmp_path / "models" / "m.gguf").resolve())
    absolute = Model("a", "local", "qwen3", str(tmp_path / "x.gguf"), 0.0)
    assert resolve_model(absolute, Path("/elsewhere")) is absolute
    assert Path(resolve_model(model, None).path).is_absolute()  # от текущей папки, как движок


def test_bundled_tokenizer_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model("m", "local", "qwen3", str(tmp_path / "m.gguf"), 0.0)
    monkeypatch.delenv(TOKENIZER_ENV, raising=False)
    assert default_engine(model).config.tokenizer == "Qwen3-8B"  # type: ignore[attr-defined]

    monkeypatch.setenv(TOKENIZER_ENV, str(tmp_path / "runtime" / "tokenizers" / "Qwen3-8B"))
    engine = default_engine(model)
    assert engine.config.tokenizer == str(tmp_path / "runtime" / "tokenizers" / "Qwen3-8B")  # type: ignore[attr-defined]
    assert "--tokenizer" in engine._arguments()  # type: ignore[attr-defined]
