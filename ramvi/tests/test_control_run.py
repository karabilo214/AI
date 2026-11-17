"""4.7 AUTO-05: контрольный прогон — замер, ключ результата, нестабильность, откат при OOM."""

import hashlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.control_run import hardware_key, long_prompt
from ramvi_service.engines import EngineManager
from ramvi_service.hardware import GPUInfo
from ramvi_service.ui_api import UIServer
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client
from test_compat import LAPTOP
from test_model_fit import tiny_qwen3


def add_gguf(data_dir: Path, *names: str) -> dict[str, Path]:
    """Настоящие маленькие GGUF; имя файла задаёт поведение поддельного движка."""
    folder = data_dir / "files"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    with open_core(data_dir) as db:
        for i, name in enumerate(names):
            path = tiny_qwen3(folder / f"{name}.gguf")
            Models(db).upsert(Model(name, "local", "qwen3", str(path), float(i)))
            paths[name] = path
    return paths


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager, Client, dict[str, Path]]]:
    paths = add_gguf(tmp_path, "model-plain", "model-runtime-oom", "model-jitter", "model-other")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        yield server, engines, Client(server.port), paths
    finally:
        server.close()
        engines.stop()


def run(client: Client, model_id: str) -> dict[str, Any]:
    client.ok("POST", f"/models/{model_id}/control")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state: dict[str, Any] = client.ok("GET", f"/models/{model_id}/control")["control"]
        if state["state"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError("control run timeout")


def test_control_run_measures_and_saves_result(service: Any, tmp_path: Path) -> None:
    server, engines, client, paths = service
    state = run(client, "model-plain")
    assert state["state"] == "done", state
    result = state["result"]
    assert [r["kind"] for r in result["runs"]] == ["short", "long", "short"]  # прогрев не в счёт
    assert result["tok_s"] > 0 and result["ttft_s"] > 0 and result["stable"] is True
    assert (result["peak_ram"], result["peak_vram"]) == (2 * 2**30, 6 * 2**30)  # из ramvi_metrics
    assert (result["resident"], result["layers"], result["context"]) == (30, 40, 2048)
    # Ключ: модель + отпечаток железа + контекст; хеш файла посчитан и сохранён в модели.
    sha = hashlib.sha256(paths["model-plain"].read_bytes()).hexdigest()
    assert result["sha256"] == sha
    with open_core(tmp_path) as db:
        models = Models(db)
        assert models.get("model-plain").sha256 == sha
        saved = models.latest_profile("model-plain", result["hardware_key"])
    assert saved is not None and saved.context == 2048 and saved.tokps == result["tok_s"]
    assert saved.peak_vram == 6 * 2**30


def test_measurement_replaces_estimate_for_installed_model(service: Any) -> None:
    server, engines, client, _ = service
    run(client, "model-plain")
    models = {m["id"]: m for m in client.ok("GET", "/models")["models"]}
    compat = models["model-plain"]["compat"]
    assert compat["speed"]["code"] == "compat.speed_here"
    assert compat["tok_s"]["measured"] is True
    assert compat["control"]["stable"] is True and compat["control"]["resident"] == 30
    other = models["model-other"]["compat"]
    assert "control" not in other  # без прогона — оценка, как раньше


def test_unstable_run_is_marked_and_lowered(service: Any) -> None:
    server, engines, client, _ = service
    result = run(client, "model-jitter")["result"]
    assert result["stable"] is False and result["spread"] > 0.25
    compat = {m["id"]: m for m in client.ok("GET", "/models")["models"]}["model-jitter"]["compat"]
    assert {"code": "compat.reason.unstable", "params": {}} in compat["reasons"]
    assert compat["control"]["stable"] is False


def test_failed_rollback_is_not_saved(tmp_path: Path) -> None:
    add_gguf(tmp_path, "model-runtime-oom")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        client = Client(server.port)
        # Поддельный движок падает при > 20 слоях: 30 → 22 (ещё OOM) — отката мало.
        first = run(client, "model-runtime-oom")["result"]
        assert first["rolled_back_from"] == 30 and first["oom"] is True
        assert server.control is not None
        model = server.chat.models_repo.get("model-runtime-oom")
        assert server.control.resident_for(model, 2048) is None  # неудачный откат не сохраняется
    finally:
        server.close()
        engines.stop()


def test_rollback_that_fits_is_saved_and_reused(tmp_path: Path, monkeypatch: Any) -> None:
    from ramvi_service import control_run

    monkeypatch.setattr(
        control_run, "SAFE_FACTOR", 0.6
    )  # 30 → 18 слоёв: поддельный движок не падает
    add_gguf(tmp_path, "model-runtime-oom")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    seen: list[Any] = []
    make = engines._make_engine

    def capture(model: Any, draft: Any = None, **options: Any) -> Any:
        seen.append(options.get("resident"))
        return make(model, draft, **options)

    engines._make_engine = capture
    server.start()
    try:
        client = Client(server.port)
        result = run(client, "model-runtime-oom")["result"]
        assert (result["rolled_back_from"], result["resident"], result["oom"]) == (30, 18, False)
        assert result["stable"] is True
        compat = client.ok("GET", "/models")["models"][0]["compat"]
        assert {
            "code": "compat.reason.rolled_back",
            "params": {"before": 30, "after": 18},
        } in compat["reasons"]
        seen.clear()
        engines.unload()
        client.ok("POST", "/engine/load", {"model_id": "model-runtime-oom"})
        deadline = time.monotonic() + 20
        while not seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert seen == [18]  # следующая загрузка — сразу с безопасным планом
    finally:
        server.close()
        engines.stop()


def test_previous_model_comes_back_after_the_run(service: Any) -> None:
    server, engines, client, _ = service
    client.ok("POST", "/engine/load", {"model_id": "model-other"})
    deadline = time.monotonic() + 20
    while engines.status()["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    run(client, "model-plain")
    assert engines.model_id == "model-other"


def test_auto_run_after_download(tmp_path: Path) -> None:
    from ramvi_service.catalog import Catalog
    from ramvi_service.catalog.headers import HeaderStore
    from test_download_manager import FileServer, Provider, entry

    files = FileServer()
    data = tiny_qwen3(tmp_path / "src.gguf").read_bytes()
    files.files["/model-plain.gguf"] = data
    item = entry(files.base, "/model-plain.gguf", data)
    catalog = Catalog(
        tmp_path / "data",
        [Provider([item])],
        headers=HeaderStore(tmp_path / "h", lambda url, size: data[:size]),
    )
    catalog.refresh()
    server, engines = build(
        tmp_path / "data",
        TOKEN,
        engine_command=ENGINE,
        app_dir=tmp_path / "app",
        catalog=catalog,
        auto_control=True,
    )
    server.start()
    try:
        client = Client(server.port)
        did = client.ok("POST", "/downloads", {"entry_id": item.id})["download"]["id"]
        deadline = time.monotonic() + 60
        state: dict[str, Any] = {}
        while time.monotonic() < deadline:
            downloads = client.ok("GET", "/downloads")["downloads"]
            model_id = next(d for d in downloads if d["id"] == did)["model_id"]
            if model_id:
                state = client.ok("GET", f"/models/{model_id}/control")["control"]
                if state["state"] == "done":
                    break
            time.sleep(0.05)
        assert state.get("state") == "done", state
    finally:
        server.close()
        engines.stop()
        files.close()


def test_hardware_key_and_prompts() -> None:
    other_driver = LAPTOP.__class__(
        **{**LAPTOP.__dict__, "gpus": (GPUInfo("RTX 5060", "NVIDIA", 8151, 100, "999.0"),)}
    )
    busy = LAPTOP.__class__(**{**LAPTOP.__dict__, "ram_available_mb": 10})
    assert hardware_key(LAPTOP) == hardware_key(busy)  # занятость памяти не меняет ключ
    assert hardware_key(LAPTOP) != hardware_key(other_driver)  # драйвер — меняет
    assert len(long_prompt(2048).split()) > len(long_prompt(512).split())


def test_unknown_model_is_404(service: Any) -> None:
    _, _, client, _ = service
    status, body = client.request("POST", "/models/nope/control")
    assert status == 404 and body["error"]["code"] == "chat.model_not_found"


def test_normal_run_does_not_pin_the_plan_and_replaced_file_drops_result(
    service: Any, tmp_path: Path
) -> None:
    server, engines, client, _ = service
    run(client, "model-plain")
    model = server.chat.models_repo.get("model-plain")
    assert server.control.resident_for(model, 2048) is None  # без отката план не закрепляется
    with open_core(tmp_path) as db:  # файл заменили: другой SHA-256
        models = Models(db)
        models.upsert(Model(**{**models.get("model-plain").__dict__, "sha256": "f" * 64}))
    compat = {m["id"]: m for m in client.ok("GET", "/models")["models"]}["model-plain"]["compat"]
    assert "control" not in compat and compat["speed"]["code"] != "compat.speed_here"
