"""3.6 (UI-01): мастер первого запуска — проверки, импорт GGUF, выбор после перезапуска."""

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.app import build
from ramvi_service.chat import ChatError
from ramvi_service.hardware import GPUInfo, HardwareProfile
from ramvi_service.onboarding import NotGGUF, Onboarding, checks, read_gguf_header
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core
from test_chat import ENGINE, TOKEN, Client, add_models

CUDA = {
    "torch": "2.9.0",
    "cuda": True,
    "cuda_version": "12.8",
    "devices": ["RTX 5060"],
    "error": None,
}


def profile(**changes: Any) -> HardwareProfile:
    base: dict[str, Any] = dict(
        os="Windows 11",
        arch="AMD64",
        cpu_name="Ryzen 7 260",
        isa=("avx2",),
        cores_physical=8,
        cores_logical=16,
        ram_total_mb=16 * 1024,
        ram_available_mb=9 * 1024,
        disk_path="D:/ramvi",
        disk_total_mb=500_000,
        disk_free_mb=120_000,
        gpus=(GPUInfo("NVIDIA GeForce RTX 5060", "NVIDIA", 8151, 7000, "580.1"),),
    )
    base.update(changes)
    return HardwareProfile(**base)


def by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in items}


def test_checks_all_ok_on_windows_with_nvidia_and_cuda() -> None:
    result = checks(profile(), CUDA)

    assert [c["id"] for c in result] == ["os", "cpu", "ram", "gpu", "driver", "disk", "backend"]
    assert {c["status"] for c in result} == {"ok"}
    found = by_id(result)
    assert found["gpu"]["params"] == {
        "name": "NVIDIA GeForce RTX 5060",
        "vram_gb": 8.0,
        "free_gb": 6.8,
    }
    assert found["ram"]["params"] == {"total_gb": 16.0, "available_gb": 9.0}
    assert found["backend"]["params"] == {"cuda": "12.8", "torch": "2.9.0"}


def test_driver_check_names_the_needed_version() -> None:
    """8.1 (PORT-02): драйвер NVIDIA в диагностике — версия и что нужно для CUDA комплекта."""
    cuda13 = {**CUDA, "cuda_version": "13.2"}
    ok = by_id(checks(profile(gpus=(GPUInfo("RTX 5060", "NVIDIA", 8151, 7000, "616.92"),)), cuda13))
    assert ok["driver"] == {
        "id": "driver",
        "status": "ok",
        "code": "system.driver",
        "params": {"version": "616.92", "need": "580", "cuda": "13.2"},
    }
    old = by_id(
        checks(profile(gpus=(GPUInfo("RTX 5060", "NVIDIA", 8151, 7000, "572.83"),)), cuda13)
    )
    assert old["driver"]["status"] == "fail" and old["driver"]["code"] == "system.driver_old"
    assert old["driver"]["params"] == {"version": "572.83", "need": "580", "cuda": "13.2"}
    cuda12 = by_id(checks(profile(gpus=(GPUInfo("RTX", "NVIDIA", 8151, 7000, "528.33"),)), CUDA))
    assert cuda12["driver"]["status"] == "ok" and cuda12["driver"]["params"]["need"] == "528.33"
    linux = by_id(
        checks(
            profile(os="Linux 6.8", gpus=(GPUInfo("RTX", "NVIDIA", 8151, 7000, "525.60.12"),)), CUDA
        )
    )
    assert linux["driver"]["code"] == "system.driver_old"  # 525.60.12 < 525.60.13
    no_cuda = {**cuda13, "cuda": False}
    missing = by_id(checks(profile(gpus=(GPUInfo("RTX", "NVIDIA", None, None, None),)), no_cuda))
    assert missing["driver"] == {
        "id": "driver",
        "status": "fail",
        "code": "system.driver_missing",
        "params": {"need": "580"},
    }
    # Windows Sandbox: nvidia-smi нет, но CUDA работает через драйвер хоста — не ошибка.
    sandbox = by_id(checks(profile(gpus=(GPUInfo("RTX", "NVIDIA", None, None, None),)), cuda13))
    assert sandbox["driver"] == {
        "id": "driver",
        "status": "ok",
        "code": "system.driver_working",
        "params": {"cuda": "13.2"},
    }
    unknown = by_id(checks(profile(), {"torch": "2.9.0", "cuda": False, "cuda_version": None}))
    assert unknown["driver"]["code"] == "system.driver_found"
    assert unknown["driver"]["params"] == {"version": "580.1"}


def test_checks_on_mac_are_the_cloud_mode_not_errors() -> None:
    """B.12: на Mac локальных моделей нет — мастер говорит об облачном режиме, без ошибок про NVIDIA."""
    hw = profile(
        os="Darwin 23.1.0", arch="arm64", gpus=(GPUInfo("Apple M1", "Apple"),), disk_free_mb=1500
    )
    found = by_id(checks(hw, {"skipped": "cloud_only"}))

    assert (found["os"]["status"], found["os"]["code"]) == ("ok", "system.os_cloud_only")
    assert "gpu" not in found and "backend" not in found and "driver" not in found
    assert (found["cloud"]["status"], found["cloud"]["code"]) == ("ok", "system.cloud_ready")
    assert (found["disk"]["status"], found["disk"]["code"]) == ("warn", "system.disk_low")
    assert all(c["status"] != "fail" for c in found.values())
    # Карта NVIDIA есть, а torch её не видит — тогда совет про драйвер.
    with_card = by_id(checks(profile(), {"torch": "2.9.0", "cuda": False}))
    assert with_card["backend"]["code"] == "system.backend_no_cuda"
    no_card = by_id(
        checks(profile(gpus=(GPUInfo("Intel UHD", "Intel"),)), {"torch": "2.9.0", "cuda": False})
    )
    assert (no_card["gpu"]["status"], no_card["backend"]["code"]) == (
        "fail",
        "system.backend_no_gpu",
    )


def test_mac_does_not_probe_torch(tmp_path: Path) -> None:
    calls = []

    def probe() -> dict[str, Any]:
        calls.append(1)
        return CUDA

    mac = profile(os="Darwin 23.1.0", arch="arm64", gpus=(GPUInfo("Apple M1", "Apple"),))
    with open_core(tmp_path) as db:
        report = Onboarding(tmp_path, Models(db), hardware=lambda _: mac, backends=probe).system()
        windows = Onboarding(
            tmp_path, Models(db), hardware=lambda _: profile(), backends=probe
        ).system()
    assert report["cloud_only"] is True and report["backends"] == {"skipped": "cloud_only"}
    assert windows["cloud_only"] is False and len(calls) == 1  # torch проверен только для Windows


def test_checks_unknown_values_and_missing_torch() -> None:
    hw = profile(
        gpus=(),
        ram_total_mb=None,
        ram_available_mb=None,
        cpu_name=None,
        cores_logical=None,
        disk_free_mb=None,
    )
    found = by_id(checks(hw, {"torch": None, "error": "ModuleNotFoundError: torch"}))

    assert found["cpu"]["status"] == found["ram"]["status"] == found["disk"]["status"] == "unknown"
    assert [found[i]["code"] for i in ("cpu", "ram", "disk")] == [
        "system.cpu_unknown",
        "system.ram_unknown",
        "system.disk_unknown",
    ]
    assert found["gpu"]["code"] == "system.gpu_none"
    assert found["backend"]["code"] == "system.backend_missing"
    assert found["backend"]["params"]["detail"] == "ModuleNotFoundError: torch"


# --- GGUF ---------------------------------------------------------------------------------


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(
    path: Path,
    architecture: str = "qwen3",
    file_type: int | None = 15,
    name: str | None = "Qwen3 14B",
) -> Path:
    """Минимальный GGUF v3: массив строк перед нужными ключами проверяет пропуск массивов."""
    kv = [
        _s("tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 8, 3)
        + _s("a")
        + _s("bb")
        + _s("ccc"),
        _s("tokenizer.ggml.scores")
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 6, 2)
        + struct.pack("<ff", 0.5, 1.5),
        _s("general.architecture") + struct.pack("<I", 8) + _s(architecture),
    ]
    if file_type is not None:
        kv.append(_s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type))
    if name is not None:
        kv.append(_s("general.name") + struct.pack("<I", 8) + _s(name))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv) + b"\0" * 64)
    return path


def test_read_gguf_header(tmp_path: Path) -> None:
    header = read_gguf_header(write_gguf(tmp_path / "m.gguf"))

    assert header == {"architecture": "qwen3", "quant": "Q4_K_M", "name": "Qwen3 14B"}
    assert (
        read_gguf_header(write_gguf(tmp_path / "x.gguf", file_type=99, name=None))["quant"]
        == "type 99"
    )


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"GGML" + b"\0" * 40,
        b"GGUF" + struct.pack("<IQQ", 1, 0, 0),
        b"GGUF" + struct.pack("<IQQ", 3, 0, 5) + _s("general.architecture"),
    ],
)
def test_read_gguf_header_rejects_bad_files(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(content)
    with pytest.raises(NotGGUF):
        read_gguf_header(path)


# --- импорт и состояние мастера --------------------------------------------------------


@pytest.fixture
def onboarding(tmp_path: Path):  # type: ignore[no-untyped-def]
    data = tmp_path / "data"
    with open_core(data) as db:
        yield Onboarding(data, Models(db), hardware=lambda _: profile(), backends=lambda: CUDA)


def code_of(call: Any) -> tuple[str, dict[str, Any]]:
    with pytest.raises(ChatError) as error:
        call()
    return error.value.code, error.value.params


def test_import_registers_file_in_place(onboarding: Onboarding, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    source = write_gguf(tmp_path / "models" / "Qwen3-14B-Q4_K_M.gguf")
    before = sorted(p.name for p in onboarding.data_dir.rglob("*"))

    model = onboarding.import_model(str(source))

    assert model["id"] == "qwen3-14b-q4_k_m"
    assert (model["architecture"], model["quant"], model["bytes"]) == (
        "qwen3",
        "Q4_K_M",
        source.stat().st_size,
    )
    assert model["capabilities"] == {"format": "gguf", "imported": True, "copied": False}
    assert onboarding.models.get(model["id"]).path == str(source.resolve())
    after = sorted(p.name for p in onboarding.data_dir.rglob("*"))
    assert after == before  # файл не копировался в папку данных
    assert onboarding.import_model(str(source))["id"] == model["id"]  # повтор — та же запись


def test_same_name_from_other_folder_gets_new_id(onboarding: Onboarding, tmp_path: Path) -> None:
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        write_gguf(tmp_path / folder / "model.gguf")

    ids = [onboarding.import_model(str(tmp_path / f / "model.gguf"))["id"] for f in ("a", "b")]

    assert ids == ["model", "model-2"]


def test_import_errors(onboarding: Onboarding, tmp_path: Path) -> None:
    llama = write_gguf(tmp_path / "llama.gguf", architecture="llama")
    text = tmp_path / "notes.gguf"
    text.write_text("hello", encoding="utf-8")

    assert code_of(lambda: onboarding.import_model("relative/m.gguf"))[0] == "import.not_absolute"
    assert (
        code_of(lambda: onboarding.import_model(str(tmp_path / "none.gguf")))[0]
        == "import.not_found"
    )
    assert code_of(lambda: onboarding.import_model(str(tmp_path)))[0] == "import.not_a_file"
    assert code_of(lambda: onboarding.import_model(str(text)))[0] == "import.not_gguf"
    code, params = code_of(lambda: onboarding.import_model(str(llama)))
    assert code == "import.unsupported_architecture"
    assert params["architecture"] == "llama" and params["supported"] == "qwen3, qwen3moe"
    assert onboarding.models.all() == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="права POSIX, не root")
def test_import_unreadable_file(onboarding: Onboarding, tmp_path: Path) -> None:
    locked = write_gguf(tmp_path / "locked.gguf")
    locked.chmod(0)
    try:
        assert code_of(lambda: onboarding.import_model(str(locked)))[0] == "import.no_access"
    finally:
        locked.chmod(0o600)


def test_complete_with_model_or_skip(onboarding: Onboarding, tmp_path: Path) -> None:
    assert onboarding.state() == {"done": False, "default_model": None}
    assert code_of(lambda: onboarding.complete("missing"))[0] == "chat.model_not_found"
    assert onboarding.complete(None) == {"done": True, "default_model": None}

    model = onboarding.import_model(str(write_gguf(tmp_path / "m.gguf")))
    assert onboarding.complete(model["id"]) == {"done": True, "default_model": model["id"]}
    saved = json.loads((onboarding.data_dir / "state.json").read_text(encoding="utf-8"))
    assert saved == {"default_model": model["id"], "onboarding_done": True}
    assert not (onboarding.data_dir / "settings.json").exists()  # настройки приложения не трогаются


def test_removed_default_model_is_not_offered(onboarding: Onboarding, tmp_path: Path) -> None:
    model = onboarding.import_model(str(write_gguf(tmp_path / "m.gguf")))
    onboarding.complete(model["id"])
    onboarding.models.delete(model["id"])

    assert onboarding.state() == {"done": True, "default_model": None}


def test_system_caches_backend_probe(tmp_path: Path) -> None:
    calls = []

    def probe() -> dict[str, Any]:
        calls.append(1)
        return CUDA

    with open_core(tmp_path) as db:
        onboarding = Onboarding(tmp_path, Models(db), hardware=lambda _: profile(), backends=probe)
        first, second = onboarding.system(), onboarding.system()

    assert len(calls) == 1
    assert first["checks"] == second["checks"] and first["hardware"]["cpu_name"] == "Ryzen 7 260"


# --- через HTTP и перезапуск сервиса -----------------------------------------------------


def start(data: Path):  # type: ignore[no-untyped-def]
    server, engines = build(data, TOKEN, engine_command=ENGINE)
    assert server.onboarding is not None
    server.onboarding._probe = lambda: CUDA  # без настоящего torch в проверках
    server.start()
    return server, engines, Client(server.port)


def test_wizard_choice_and_history_survive_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    model_file = write_gguf(tmp_path / "Qwen3-8B.gguf")
    add_models(data, "model-a")
    server, engines, client = start(data)
    try:
        assert client.ok("GET", "/onboarding") == {"done": False, "default_model": None}
        checks_ = client.ok("GET", "/system")["checks"]
        expected = (  # B.12: на Mac — облачный режим, без проверок видеокарты и torch
            ["os", "cpu", "ram", "disk", "cloud"]
            if sys.platform == "darwin"
            else ["os", "cpu", "ram", "gpu", "disk", "backend"]
        )
        assert [c["id"] for c in checks_] == expected
        imported = client.ok("POST", "/models/import", {"path": str(model_file)})["model"]
        status, error = client.request("POST", "/models/import", {"path": "relative.gguf"})
        assert status == 400 and error["error"]["code"] == "import.not_absolute"
        client.ok("POST", "/onboarding", {"model_id": imported["id"]})
        chat = client.ok("POST", "/chats")["chat"]
        assert chat["model_id"] == imported["id"]
        client.ok(
            "POST", f"/chats/{chat['id']}/model", {"model_id": "model-a"}
        )  # последний явный выбор
        client.ok("POST", f"/chats/{chat['id']}/title", {"title": "мой диалог"})
    finally:
        server.close()
        engines.stop()

    server, engines, client = start(data)
    try:
        assert client.ok("GET", "/onboarding") == {"done": True, "default_model": "model-a"}
        (saved,) = client.ok("GET", "/chats")["chats"]
        assert saved["title"] == "мой диалог" and saved["model_id"] == "model-a"
        assert client.ok("POST", "/chats")["chat"]["model_id"] == "model-a"
        assert {m["id"] for m in client.ok("GET", "/models")["models"]} == {
            "model-a",
            imported["id"],
        }
    finally:
        server.close()
        engines.stop()


def test_every_onboarding_code_has_texts_in_both_languages() -> None:
    import json
    import re

    root = Path(__file__).resolve().parents[1]
    source = (root / "service" / "ramvi_service" / "onboarding.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'"((?:system|import)\.[a-z_]+)"', source))
    assert len(codes) >= 18, codes
    for language in ("ru", "en"):
        catalog = json.loads((root / "i18n" / f"{language}.json").read_text(encoding="utf-8"))
        assert sorted(codes - catalog.keys()) == [], language


def test_import_accepts_moe_like_engine(onboarding: Onboarding, tmp_path: Path) -> None:
    """Движок грузит qwen3moe (model_loader → core.moe) — импорт тоже должен принять."""
    model = onboarding.import_model(
        str(write_gguf(tmp_path / "qwen3-30b.gguf", architecture="qwen3moe"))
    )
    assert model["architecture"] == "qwen3moe"


def test_supported_architectures_match_engine_loader() -> None:
    import re

    from ramvi_service.onboarding import SUPPORTED_ARCHITECTURES

    root = Path(__file__).resolve().parents[1]
    loader = (root / "engine" / "ramvi_engine" / "model_loader.py").read_text(encoding="utf-8")
    match = re.search(r"if architecture not in \(([^)]*)\)", loader)
    assert match is not None, "проверка архитектур в model_loader изменилась"
    engine = tuple(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
    assert SUPPORTED_ARCHITECTURES == engine


def test_ollama_blob_gets_readable_id(onboarding: Onboarding, tmp_path: Path) -> None:
    """Файлы Ollama называются хешем — id из заголовка (как на ASUS: sha256-7f40…)."""
    blob = write_gguf(tmp_path / ("sha256-" + "7f4030143c1c4772" * 4), name="Qwen3 0.6B")
    assert onboarding.import_model(str(blob))["id"] == "qwen3-0.6b-q4_k_m"
    plain = write_gguf(tmp_path / "sha256-short.gguf")  # не хеш — имя файла как есть
    assert onboarding.import_model(str(plain))["id"] == "sha256-short"


def test_scan_models_dir_imports_ollama_blobs_in_place(tmp_path: Path) -> None:
    """«Обновить список»: файлы без .gguf (Ollama blobs) подключаются по заголовку, на месте."""
    from ramvi_service import settings as st

    app, data = tmp_path / "app", tmp_path / "app" / "data"
    app.mkdir()
    ollama = tmp_path / "ollama" / "models"
    blobs = ollama / "blobs"
    blobs.mkdir(parents=True)
    model_file = write_gguf(blobs / ("sha256-" + "a" * 64), name="Qwen3 14B")
    write_gguf(blobs / ("sha256-" + "b" * 64), architecture="llama", name="Llama")
    (blobs / ("sha256-" + "c" * 64)).write_text(
        '{"license": "text"}', encoding="utf-8"
    )  # не модель
    st.save(app, st.Settings(models_dir=str(ollama)))  # указана сама папка Ollama
    with open_core(data) as db:
        onboarding = Onboarding(
            data, Models(db), hardware=lambda _: profile(), backends=lambda: CUDA, app_dir=app
        )
        found = onboarding.scan_models_dir()

        assert found["added"] == ["qwen3-14b-q4_k_m"]
        assert found["folder"] == str(ollama)
        assert [s["code"] for s in found["skipped"]] == ["import.unsupported_architecture"]
        assert onboarding.models.get("qwen3-14b-q4_k_m").path == str(model_file.resolve())
        assert onboarding.scan_models_dir()["added"] == []  # повтор не дублирует
        assert [m.id for m in onboarding.models.all()] == ["qwen3-14b-q4_k_m"]
        assert sorted(p.name for p in blobs.iterdir())[0].startswith("sha256-a")  # не скопирован


def test_scan_route_any_extension_in_own_folder(tmp_path: Path) -> None:
    """POST /models/scan: своя папка, файл с любым расширением; ответ — новые и весь список."""
    from ramvi_service import settings as st

    data, folder = tmp_path / "data", tmp_path / "mine"
    folder.mkdir()
    write_gguf(folder / "qwen-file.bin", name="Qwen3 8B")
    add_models(data, "model-a")
    st.save(data, st.Settings(models_dir=str(folder)))  # без папки приложения настройки в data
    server, engines, client = start(data)
    try:
        found = client.ok("POST", "/models/scan", {})
        assert found["added"] == ["qwen-file"] and found["skipped"] == []
        assert sorted(m["id"] for m in found["models"]) == ["model-a", "qwen-file"]
        assert client.ok("POST", "/models/scan", {})["added"] == []
    finally:
        server.close()
        engines.stop()


def test_wmi_driver_versions_and_missing_vc_runtime() -> None:
    """8.1: в WMI у NVIDIA своя запись версии; виртуальный драйвер Microsoft — не NVIDIA."""
    from ramvi_service.hardware import nvidia_driver_from_wmi, parse_wmi_json

    assert (
        nvidia_driver_from_wmi("32.0.16.1692") == "616.92"
    )  # ноутбук с RTX 5060: nvidia-smi 616.92
    assert nvidia_driver_from_wmi("32.0.15.6692") == "566.92"
    assert nvidia_driver_from_wmi("10.0.26100.1150") is None  # Windows Sandbox
    assert nvidia_driver_from_wmi("junk") is None and nvidia_driver_from_wmi(None) is None
    _, gpus = parse_wmi_json(
        '{"gpu": [{"Name": "NVIDIA GeForce RTX 5060", "DriverVersion": "10.0.26100.1150"},'
        ' {"Name": "AMD Radeon 780M", "DriverVersion": "32.0.32015.7029"}]}'
    )
    assert [(g.vendor, g.driver) for g in gpus] == [("NVIDIA", None), ("AMD", "32.0.32015.7029")]
    vcredist = by_id(
        checks(
            profile(),
            {
                "torch": None,
                "cuda": False,
                "error": 'OSError: [WinError 126] Error loading "c10.dll"',
            },
        )
    )
    assert vcredist["backend"]["code"] == "system.backend_vcredist"
