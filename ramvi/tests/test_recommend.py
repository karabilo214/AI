"""4.6 AUTO-04: выбор модели с причиной и альтернативами, без рейтинга «умнее»."""

from pathlib import Path
from typing import Any

import pytest
from ramvi_service import compat
from ramvi_service.i18n import Msg, load_catalog, render
from ramvi_service.recommend import rank
from ramvi_service.task_profile import TaskProfile

MIB = 1024 * 1024
RU = load_catalog("ru")


def text(value: Any) -> str:
    def build(item: Any) -> Any:
        if isinstance(item, dict) and "code" in item:
            return Msg(item["code"], {k: build(v) for k, v in item["params"].items()})
        if isinstance(item, list):
            return [build(x) for x in item]
        return item

    return render(build(value), RU)


def item(
    name: str,
    size: float,
    speed: tuple[float, float] | None,
    *,
    status: str = compat.RECOMMENDED,
    quant: str = "Q4_K_M",
    ram_mb: int = 0,
    vram_mb: int = 4000,
    off_gpu_mb: int = 0,
    suits: bool = True,
    installed: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    tok = None if speed is None else {"low": speed[0], "high": speed[1], "measured": False}
    return {
        "id": f"cat:{name}",
        "name": name,
        "installed": installed,
        "compat": {
            "status": status,
            "reasons": [],
            "speed": {"code": "compat.speed_estimate", "params": {"low": "1", "high": "2"}},
            "plan": {"off_gpu_bytes": off_gpu_mb * MIB},
            "memory": {"vram": vram_mb * MIB, "ram": ram_mb * MIB, "mapped": 0},
            "task": {"suits": suits, "reasons": []},
            "tok_s": tok,
            "model": {"key": key or name, "size_b": size, "quant": quant},
        },
    }


LAPTOP_LIKE = [
    item("0.6B", 0.6, (230, 270), vram_mb=1700),
    item("4B", 4, (81, 95), vram_mb=3700),
    item("8B", 8, (50, 59), vram_mb=5900),
    item("14B", 14, (7.8, 9.1), status=compat.SLOW, ram_mb=1500, off_gpu_mb=1900, vram_mb=8000),
    item("32B", 32, (0.7, 1.3), status=compat.SLOW, ram_mb=8800, off_gpu_mb=12000, vram_mb=8000),
]


def test_balanced_picks_largest_comfortable_model() -> None:
    result = rank(LAPTOP_LIKE, "balanced")
    assert result["best"]["name"] == "8B"
    assert "Рейтинга «умнее» нет" in text(result["reason"])
    assert 2 <= len(result["alternatives"]) <= 3


def test_each_priority_orders_by_its_measurable_value() -> None:
    assert rank(LAPTOP_LIKE, "speed")["best"]["name"] == "0.6B"
    assert rank(LAPTOP_LIKE, "ram")["best"]["name"] == "0.6B"
    assert rank(LAPTOP_LIKE, "ttft")["best"]["name"] == "0.6B"
    quality = rank(LAPTOP_LIKE, "quality")
    assert quality["best"]["name"] == "14B"  # 32B медленнее 3 ток/с — не интерактивна
    assert "Рейтинга «умнее» нет" in text(quality["reason"])


def test_only_slow_models_fall_back_with_honest_reason() -> None:
    slow = [item("32B", 32, (0.7, 1.3), status=compat.SLOW), item("70B", 70, (0.2, 0.4))]
    for priority in ("balanced", "quality"):
        result = rank(slow, priority)
        assert result["best"]["name"] == "70B"
        assert result["reason"]["code"] == "recommend.reason.quality_slow"
    readable = [item("14B", 14, (7.8, 9.1), status=compat.SLOW), item("8B", 8, (2, 4))]
    result = rank(readable, "balanced")
    assert result["best"]["name"] == "14B"
    assert result["reason"]["code"] == "recommend.reason.balanced_slow"


def test_filters_runnable_and_task_and_counts_excluded() -> None:
    items = [
        item("8B", 8, (50, 59)),
        item("no-tools", 14, (8, 9), suits=False),
        item("too-big", 70, None, status=compat.NO_RAM),
        item("unknown", 4, None, status="compat.unknown"),
    ]
    result = rank(items, "quality")
    assert result["best"]["name"] == "8B"
    assert result["excluded"] == {"unknown": 1, "not_runnable": 1, "not_for_task": 1}
    assert [a["name"] for a in result["alternatives"]] == []


def test_nothing_suitable_explains_why() -> None:
    result = rank([item("too-big", 70, None, status=compat.NO_VRAM)], "balanced")
    assert result["best"] is None and result["alternatives"] == []
    assert text(result["reason"]) == (
        "подходящей модели нет: не запустятся на этом ПК — 1, не подходят задаче — 0, нет данных — 0"
    )


def test_same_file_from_two_sources_is_one_model_installed_first() -> None:
    items = [
        item("hf 8B", 8, (50, 59), key="qwen3-8b-q4_k_m"),
        item("local 8B", 8, (50, 59), key="qwen3-8b-q4_k_m", installed=True),
        item("4B", 4, (81, 95)),
    ]
    result = rank(items, "balanced")
    assert result["best"]["name"] == "local 8B" and result["best"]["installed"] is True
    assert [a["name"] for a in result["alternatives"]] == ["4B"]


def test_alternatives_state_measurable_differences() -> None:
    result = rank(LAPTOP_LIKE, "balanced")
    reasons = {a["name"]: text(a["reason"]) for a in result["alternatives"]}
    assert reasons["4B"].endswith("меньше; быстрее")
    assert reasons["14B"].endswith("крупнее; медленнее; больше RAM")


def test_economy_mode_ranks_by_ram_and_bad_priority_is_rejected() -> None:
    assert TaskProfile("economy").ranking == "ram"
    assert TaskProfile("economy", "speed").ranking == "speed"
    assert TaskProfile("chat").ranking == "balanced"
    with pytest.raises(ValueError):
        rank(LAPTOP_LIKE, "smartest")


def test_api_recommendations_follow_profile(tmp_path: Path) -> None:
    from test_download_manager import App, FileServer, entry
    from test_model_fit import tiny_qwen3

    files = FileServer()
    data = tiny_qwen3(tmp_path / "src.gguf").read_bytes()
    files.files["/tiny-Q4_K_M.gguf"] = data
    app = App(tmp_path, [entry(files.base, "/tiny-Q4_K_M.gguf", data)], (0.05,), headers=True)
    try:
        answer = app.client.ok("GET", "/recommendations")
        assert answer["profile"]["priority"] == "balanced"
        assert answer["priority"] == "balanced"
        assert set(answer) >= {"best", "reason", "alternatives", "excluded"}
        # На этом компьютере модель либо запускается (тогда она лучшая), либо нет — без NVIDIA.
        if answer["best"] is not None:
            assert answer["best"]["id"] == "test:/tiny-Q4_K_M.gguf"
        else:
            assert answer["excluded"]["not_runnable"] == 1
        app.client.ok("POST", "/settings", {"task_mode": "economy"})
        assert app.client.ok("GET", "/recommendations")["priority"] == "ram"
    finally:
        app.stop()
        files.close()
