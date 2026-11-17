"""4.11: модель по умолчанию и удаление модели; файл пользователя не удаляется никогда."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.sections import section_of
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client
from test_model_fit import tiny_qwen3


@pytest.fixture
def app(tmp_path: Path) -> Iterator[tuple[Any, Any, Client, dict[str, Path]]]:
    root, data = tmp_path / "app", tmp_path / "app" / "data"
    (root / "models").mkdir(parents=True)
    own = tiny_qwen3(root / "models" / "model-own.gguf")  # скачан RAMVI в папку моделей
    (tmp_path / "elsewhere").mkdir()
    user = tiny_qwen3(tmp_path / "elsewhere" / "model-user.gguf")  # подключён на месте
    with open_core(data) as db:
        models = Models(db)
        models.upsert(
            Model(
                "model-own",
                "test",
                "qwen3",
                "models/model-own.gguf",
                0.0,
                capabilities={"imported": False, "catalog_id": "test:own"},
            )
        )
        models.upsert(
            Model(
                "model-user",
                "local",
                "qwen3",
                str(user),
                1.0,
                capabilities={"imported": True, "copied": False},
            )
        )
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=root)
    server.start()
    try:
        yield server, engines, Client(server.port), {"own": own, "user": user}
    finally:
        server.close()
        engines.stop()


def ids(client: Client) -> list[str]:
    return sorted(m["id"] for m in client.ok("GET", "/models")["models"])


def test_default_model_is_used_for_new_chats(app: Any) -> None:
    _, _, client, _ = app
    assert client.ok("POST", "/models/model-user/default") == {"default_model": "model-user"}
    assert client.ok("GET", "/onboarding")["default_model"] == "model-user"
    assert client.ok("POST", "/chats", {})["chat"]["model_id"] == "model-user"
    status, body = client.request("POST", "/models/nope/default")
    assert status == 404 and body["error"]["code"] == "chat.model_not_found"


def test_user_file_is_never_deleted(app: Any) -> None:
    _, _, client, files = app
    status, body = client.request("POST", "/models/model-user/delete", {"delete_file": True})
    assert status == 400 and body["error"]["code"] == "models.file_not_ours"
    assert (
        ids(client) == ["model-own", "model-user"] and files["user"].exists()
    )  # ничего не тронуто

    assert client.ok("POST", "/models/model-user/delete", {}) == {
        "deleted": True,
        "file_deleted": False,
    }
    assert ids(client) == ["model-own"] and files["user"].exists()  # забыта, файл на месте


def test_downloaded_file_is_deleted_only_on_request(app: Any) -> None:
    _, _, client, files = app
    answer = client.ok("POST", "/models/model-own/delete", {"delete_file": True})
    assert answer == {"deleted": True, "file_deleted": True}
    assert not files["own"].exists() and ids(client) == ["model-user"]
    status, body = client.request("POST", "/models/model-user/delete", {"delete_file": "yes"})
    assert status == 400 and body["error"]["code"] == "api.bad_request"


def test_loaded_model_is_unloaded_before_delete(app: Any) -> None:
    _, engines, client, _ = app
    client.ok("POST", "/engine/load", {"model_id": "model-user"})
    deadline = time.monotonic() + 20
    while engines.status()["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engines.model_id == "model-user"
    client.ok("POST", "/models/model-user/delete", {})
    assert engines.model_id is None and ids(client) == ["model-own"]


def test_models_section_is_built_in_and_reserved(app: Any) -> None:
    _, _, client, _ = app
    sections = client.ok("GET", "/sections")["sections"]
    assert [s["id"] for s in sections][:2] == ["chat", "models"]
    from test_sections import manifest

    with pytest.raises(ValueError):
        section_of(manifest("thief", view="models"))


def test_locked_file_keeps_the_model_and_busy_model_is_refused(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows: открытый движком файл не удалить — запись остаётся, ошибка понятная."""
    import os

    server, _, client, files = app

    def locked(path: Any) -> None:
        raise PermissionError(32, "The process cannot access the file", str(path))

    monkeypatch.setattr(os, "remove", locked)
    status, body = client.request("POST", "/models/model-own/delete", {"delete_file": True})
    assert status == 400 and body["error"]["code"] == "models.file_busy"
    assert "model-own" in ids(client) and files["own"].exists()
    monkeypatch.undo()

    assert server.library is not None
    server.library.busy = lambda model_id: model_id == "model-own"  # идёт контрольный прогон
    status, body = client.request("POST", "/models/model-own/delete", {"delete_file": True})
    assert status == 400 and body["error"]["code"] == "models.busy"
    assert "model-own" in ids(client)
