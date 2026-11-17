"""4.5 AUTO-02/AUTO-03: профиль задачи и проверка выполнимости по свободной памяти."""

from pathlib import Path
from typing import Any

import pytest
from ramvi_service import compat
from ramvi_service.catalog.headers import HeaderStore
from ramvi_service.gguf_meta import GGUFMeta
from ramvi_service.hardware import GPUInfo
from ramvi_service.model_fit import ModelFit
from ramvi_service.settings import Settings, SettingsError
from ramvi_service.task_profile import MODES, TaskProfile
from test_compat import LAPTOP
from test_model_fit import header, tiny_qwen3

Q4_K, Q6_K, F32 = 12, 14, 0


def qwen3_14b() -> GGUFMeta:
    """Заголовок с размерами настоящего Qwen3-14B Q4_K_M (без файла и весов)."""
    d, heads, kv, hd, ff, layers, vocab = 5120, 40, 8, 128, 17408, 40, 151936
    meta = {
        "general.architecture": "qwen3",
        "general.size_label": "14B",
        "general.file_type": 15,
        "qwen3.block_count": layers,
        "qwen3.embedding_length": d,
        "qwen3.attention.head_count": heads,
        "qwen3.attention.head_count_kv": kv,
        "qwen3.attention.key_length": hd,
        "qwen3.attention.value_length": hd,
        "qwen3.feed_forward_length": ff,
        "qwen3.context_length": 40960,
    }
    tensors: dict[str, tuple[tuple[int, ...], int]] = {
        "token_embd.weight": ((d, vocab), Q4_K),
        "output_norm.weight": ((d,), F32),
        "output.weight": ((d, vocab), Q6_K),
    }
    for i in range(layers):
        tensors.update(
            {
                f"blk.{i}.attn_norm.weight": ((d,), F32),
                f"blk.{i}.ffn_norm.weight": ((d,), F32),
                f"blk.{i}.attn_q_norm.weight": ((hd,), F32),
                f"blk.{i}.attn_k_norm.weight": ((hd,), F32),
                f"blk.{i}.attn_q.weight": ((d, heads * hd), Q4_K),
                f"blk.{i}.attn_k.weight": ((d, kv * hd), Q4_K),
                f"blk.{i}.attn_v.weight": ((d, kv * hd), Q6_K if i % 2 else Q4_K),
                f"blk.{i}.attn_output.weight": ((heads * hd, d), Q4_K),
                f"blk.{i}.ffn_gate.weight": ((d, ff), Q4_K),
                f"blk.{i}.ffn_up.weight": ((d, ff), Q4_K),
                f"blk.{i}.ffn_down.weight": ((ff, d), Q6_K if i % 2 else Q4_K),
            }
        )
    return GGUFMeta(meta, tensors)


DEFAULT = TaskProfile()


def fit(profile: TaskProfile = DEFAULT, hw: Any = LAPTOP) -> ModelFit:
    store = HeaderStore(Path("unused"))
    return ModelFit(store, lambda: hw, native_f16=lambda: True, profile=lambda: profile)


CAPS = ("chat", "tools", "json", "draft")


def test_14b_is_feasible_on_8gb_gpu_with_enough_ram() -> None:
    """Приёмка AUTO-03: гибридный план 14B на 8 ГБ VRAM при достаточной RAM — выполним."""
    verdict = fit().verdict(qwen3_14b(), "Qwen3-14B Q4_K_M", 9_000_000_000, False, CAPS)
    assert verdict["status"] == compat.SLOW, verdict  # не «неподходящая»
    plan = verdict["plan"]
    assert 0 < plan["gpu"] < plan["layers"] == 40
    assert verdict["now"]["fits"] is True  # и в свободную сейчас память (7,9 ГБ VRAM, 7,5 ГБ RAM)
    assert verdict["task"]["suits"] is True
    assert verdict["speed"]["code"] == "compat.speed_range"  # есть замер 14B


def test_profile_context_changes_the_plan_not_the_files(tmp_path: Path) -> None:
    meta = qwen3_14b()
    short = fit(TaskProfile("chat")).verdict(meta, "14B", 9 * 10**9, False, CAPS)
    long = fit(TaskProfile("long_context")).verdict(meta, "14B", 9 * 10**9, False, CAPS)
    assert (short["context"], long["context"]) == (2048, 32768)
    # KV-кэш 32k вытесняет слои из VRAM: меньше слоёв на карте, больше RAM.
    assert long["plan"]["gpu"] < short["plan"]["gpu"]
    assert long["memory"]["ram"] > short["memory"]["ram"]
    # Замер 14B снят при контексте 2048 — при 32k показывается оценка, а не чужой замер.
    assert short["speed"]["code"] == "compat.speed_range"
    assert long["speed"]["code"] == "compat.speed_estimate"


def test_task_needs_capabilities_and_context_length(tmp_path: Path) -> None:
    meta = header(tmp_path)  # контекст модели 4096
    agent = fit(TaskProfile("agent", context=4096)).verdict(meta, "m", 10**6, False, ("chat",))
    assert agent["task"] == {
        "suits": False,
        "reasons": [{"code": "task.reason.capability", "params": {"capability": "tools"}}],
    }
    assert fit(TaskProfile("agent", context=4096)).verdict(meta, "m", 10**6, False, CAPS)["task"][
        "suits"
    ]
    long = fit(TaskProfile("long_context")).verdict(meta, "m", 10**6, False, CAPS)["task"]
    assert long == {
        "suits": False,
        "reasons": [{"code": "task.reason.context", "params": {"model": 4096, "need": 32768}}],
    }
    json_mode = fit(TaskProfile("json")).verdict(meta, "m", 10**6, False, ("chat", "tools"))
    assert json_mode["task"]["reasons"][0]["params"] == {"capability": "json"}
    # Подходит ли задаче — отдельно от статуса железа.
    assert agent["status"] == compat.RECOMMENDED


def test_busy_now_is_a_note_not_a_new_status() -> None:
    busy_gpu = GPUInfo("RTX 5060", "NVIDIA", 8151, 200, "616.92")
    busy = LAPTOP.__class__(**{**LAPTOP.__dict__, "gpus": (busy_gpu,), "ram_available_mb": 300})
    verdict = fit(hw=busy).verdict(qwen3_14b(), "14B", 9 * 10**9, False, CAPS)
    assert verdict["status"] == compat.SLOW  # компьютер способен
    assert verdict["now"] == {"fits": False}  # но сейчас память занята
    assert {"code": "compat.reason.busy_now", "params": {}} in verdict["reasons"]


def test_profile_validation_and_defaults() -> None:
    assert TaskProfile().effective_context == 2048
    assert TaskProfile("code", context=16384).effective_context == 16384
    assert TaskProfile("agent").needs == ("tools",)
    assert set(MODES) == {"chat", "code", "agent", "long_context", "json", "economy"}
    for bad in (
        dict(task_mode="poetry"),
        dict(task_priority="cheap"),
        dict(task_context=100),
        dict(task_context=True),
    ):
        with pytest.raises(SettingsError):
            Settings(**bad).validate()


def test_api_profile_from_settings_rebuilds_recommendations(tmp_path: Path) -> None:
    from test_download_manager import App, FileServer, entry

    files = FileServer()
    data = tiny_qwen3(tmp_path / "src.gguf").read_bytes()
    files.files["/tiny-Q4_K_M.gguf"] = data
    app = App(tmp_path, [entry(files.base, "/tiny-Q4_K_M.gguf", data)], (0.05,), headers=True)
    try:
        before = app.client.ok("GET", "/catalog")
        assert before["profile"] == {
            "mode": "chat",
            "priority": "speed",
            "context": 2048,
            "needs": [],
        }
        assert before["entries"][0]["compat"]["context"] == 2048
        files_before = sorted(p.name for p in (tmp_path / "data").rglob("*") if p.is_file())

        app.client.ok("POST", "/settings", {"task_mode": "long_context", "task_priority": "ram"})
        after = app.client.ok("GET", "/catalog")
        assert after["profile"]["context"] == 32768 and after["profile"]["priority"] == "ram"
        assert after["entries"][0]["compat"]["context"] == 32768
        assert after["entries"][0]["compat"]["task"]["suits"] is False  # у модели 4096
        assert app.engines.context() == 32768  # следующая загрузка — с контекстом профиля

        status, body = app.client.request("POST", "/settings", {"task_mode": "poetry"})
        assert status == 400 and body["error"]["code"] == "settings.invalid"
        files_after = sorted(p.name for p in (tmp_path / "data").rglob("*") if p.is_file())
        assert files_after == files_before  # профиль не трогает файлы данных и моделей
    finally:
        app.stop()
        files.close()


def test_engine_is_started_with_profile_context(tmp_path: Path) -> None:
    from ramvi_service.engines import default_engine
    from ramvi_storage.repositories import Model

    target = default_engine(Model("m", "local", "qwen3", str(tmp_path / "m.gguf"), 0.0), maxt=8192)
    assert target.config.maxt == 8192  # type: ignore[attr-defined]


def test_model_load_uses_profile_context(tmp_path: Path) -> None:
    import json

    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, Client, add_models
    from test_model_load import wait_engine

    add_models(tmp_path, "model-a")
    (tmp_path / "settings.json").write_text(json.dumps({"task_mode": "code"}), encoding="utf-8")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    seen: list[int] = []
    make = engines._make_engine

    def capture(model: Any, draft: Any = None, **options: Any) -> Any:
        seen.append(options["maxt"])
        target = make(model, draft, **options)
        command = target._command  # type: ignore[attr-defined]  # контекст — аргументом
        assert command[command.index("--maxt") + 1] == "8192"
        return target

    engines._make_engine = capture
    server.start()
    try:
        client = Client(server.port)
        client.ok("POST", "/engine/load", {"model_id": "model-a"})
        assert wait_engine(client)["state"] == "ready"
        assert seen == [8192]
    finally:
        server.close()
        engines.stop()


def test_engine_creation_error_does_not_hang_loading(tmp_path: Path) -> None:
    from ramvi_service.app import build
    from test_chat import ENGINE, TOKEN, Client, add_models
    from test_model_load import wait_engine

    add_models(tmp_path, "model-a")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)

    def broken(model: Any, draft: Any = None, **options: Any) -> Any:
        raise RuntimeError("bad engine config")

    engines._make_engine = broken
    server.start()
    try:
        client = Client(server.port)
        client.ok("POST", "/engine/load", {"model_id": "model-a"})
        engine = wait_engine(client)  # ждёт, пока не кончится «загружается»
        assert engine["loading"] is None and engine["state"] == "stopped"
    finally:
        server.close()
        engines.stop()
