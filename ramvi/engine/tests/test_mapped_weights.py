"""3.18 ENG-MAPPED-01: слои вне видеокарты из файла модели, без собственных копий.

Решаемые вручную примеры: матрица 256x256 F16 = 131072 байта, 7 матриц в слое.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_engine.planning.resources import STAGE_RAM, MemoryUsage, estimate_gguf

MATRIX = 256 * 256 * 2


class F16GGUF:
    meta = {
        "general.architecture": "qwen3",
        "qwen3.block_count": 3,
        "qwen3.embedding_length": 256,
        "qwen3.attention.head_count": 2,
        "qwen3.attention.head_count_kv": 1,
        "qwen3.attention.key_length": 128,
        "qwen3.feed_forward_length": 256,
    }
    tensors = {"output.weight": None}

    def __init__(self, kind: str = "F16") -> None:
        self.kind = kind

    def raw(self, name: str) -> tuple[SimpleNamespace, int, int, str]:
        size = 256 * 144 if self.kind == "Q4_K" else MATRIX
        return SimpleNamespace(size=size), 256, 256, self.kind


def test_streamed_f16_is_file_working_set_not_owned_ram() -> None:
    e = estimate_gguf(F16GGUF(), ["gpu", "stream", "stream"], 10)
    assert e.streaming == MemoryUsage(2 * 7 * MATRIX, STAGE_RAM)  # два слота в VRAM + кольцо
    assert e.weights.mapped == (1 + 14) * MATRIX  # embedding + 14 подкачиваемых матриц из файла
    assert e.weights.vram == 8 * MATRIX  # output + 7 матриц слоя в видеокарте


def test_streamed_quantized_keeps_pinned_copy() -> None:
    e = estimate_gguf(F16GGUF("Q4_K"), ["gpu", "stream", "stream"], 10)
    assert e.streaming.ram == 14 * 256 * 148  # repack в закреплённой RAM, как раньше
    assert e.weights.mapped == 256 * 144  # только embedding


def test_cpu_f16_copy_only_without_native_kernel() -> None:
    plain = estimate_gguf(F16GGUF(), ["cpu"] * 3, 7)
    native = estimate_gguf(F16GGUF(), ["cpu"] * 3, 7, native_f16=True)
    assert plain.persistent.ram - native.persistent.ram == 3 * 7 * 256 * 256 * 4
    assert native.weights.mapped == plain.weights.mapped == (1 + 21) * MATRIX
    assert native.streaming == MemoryUsage()  # процессорные слои не подкачиваются


def test_hybrid_f16_counted_once_in_file() -> None:
    e = estimate_gguf(F16GGUF(), ["gpu", "hyb", "stream"], 10, native_f16=True)
    assert e.weights.mapped == (1 + 14) * MATRIX
    assert e.streaming.ram == STAGE_RAM


def test_owned_ram_now_below_file_size_for_offloaded_model() -> None:
    """Суть пункта: вне видеокарты своей RAM почти нет, остальное — файл."""
    e = estimate_gguf(F16GGUF(), ["stream"] * 3, 2048, native_f16=True)
    assert e.allocated.ram - STAGE_RAM < MATRIX  # кроме кольца — меньше одной матрицы из 21
    assert e.allocated.mapped >= 21 * MATRIX


@pytest.mark.parametrize(
    "features,platform,no_cpp,expected",
    [
        ({"X86_V4": True, "F16C": True}, "win32", None, True),
        ({"X86_V4": False, "F16C": True}, "win32", None, False),
        ({"X86_V4": True, "F16C": True}, "darwin", None, False),
        ({"X86_V4": True, "F16C": True}, "win32", "1", False),
    ],
)
def test_native_check(
    monkeypatch: pytest.MonkeyPatch,
    features: dict[str, bool],
    platform: str,
    no_cpp: str | None,
    expected: bool,
) -> None:
    from ramvi_engine.cpu import native_check

    if no_cpp is None:
        monkeypatch.delenv("RAMVI_NO_CPP", raising=False)
    else:
        monkeypatch.setenv("RAMVI_NO_CPP", no_cpp)
    assert native_check.native_cpu_available(features, platform) is expected


def test_native_check_needs_library_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ramvi_engine.cpu import native_check

    monkeypatch.delenv("RAMVI_NO_CPP", raising=False)
    monkeypatch.setattr(native_check, "__file__", str(tmp_path / "cpu" / "native_check.py"))
    assert not native_check.native_cpu_available({"X86_V4": True, "F16C": True}, "win32")


def test_host_weights_maps_file_without_copy(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from ramvi_engine.core.engine import _unpack
    from ramvi_engine.core.gengine import host_weights

    values = (np.arange(256 * 256, dtype=np.float32) % 97 - 48).astype(np.float16)
    path = tmp_path / "w.bin"
    values.tofile(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    src, lay = host_weights((mm, 256, 256, "F16"), arena=None)  # type: ignore[no-untyped-call]  # арена не нужна: копии нет
    assert src.dtype == torch.uint8 and not src.is_pinned()
    assert src.data_ptr() == mm.ctypes.data  # те же страницы файла
    w = _unpack(src.clone(), lay)  # type: ignore[no-untyped-call]
    assert (w["t"], w["N"], w["K"]) == ("F16", 256, 256)
    assert torch.equal(w["W"], torch.from_numpy(values.reshape(256, 256)))


def test_loader_stages_pageable_source_exactly() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("нужна CUDA")
    from ramvi_engine.core.engine import Loader

    loader = Loader()  # type: ignore[no-untyped-call]
    try:
        n = 37 * 2**20 + 123  # несколько кусков по 10 МиБ и неполный хвост; больше кольца (40 МиБ)
        src = torch.randint(0, 256, (n,), dtype=torch.uint8)
        assert not src.is_pinned()
        for _ in range(2):  # второй проход переиспользует буферы кольца
            dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
            import threading

            job: dict[str, Any] = {
                "pairs": [(src, dst)],
                "wait": None,
                "done": threading.Event(),
                "ev": None,
            }
            loader.q.put(job)
            Loader.wait(job)  # type: ignore[no-untyped-call]
            job["ev"].synchronize()
            assert torch.equal(dst.cpu(), src)
    finally:
        loader.close()  # type: ignore[no-untyped-call]


def test_offload_fraction_rule() -> None:
    from ramvi_engine.planning.automatic import CPU_FRACTION_QUANTIZED, offload_cpu_fraction

    assert offload_cpu_fraction(True) == 1.0  # несжатые слои — только процессору
    assert offload_cpu_fraction(False) == CPU_FRACTION_QUANTIZED == 0.58  # прежний замер для Q4
    assert offload_cpu_fraction(True, "0.25") == 0.25  # явная настройка сильнее правила
    with pytest.raises(ValueError):
        offload_cpu_fraction(False, "1.5")


def test_unquantized_layers_from_metadata() -> None:
    from ramvi_engine.planning.automatic import unquantized_layers

    assert unquantized_layers(F16GGUF("F16"))
    assert not unquantized_layers(F16GGUF("Q4_K"))
    moe = F16GGUF("F16")
    moe.meta = {**F16GGUF.meta, "general.architecture": "qwen3moe"}
    assert not unquantized_layers(moe)


def test_no_draft_while_unquantized_layers_are_offloaded() -> None:
    """Вручную: VRAM 100. С draft помещается resident=2 (10+60+15=85), все 3 слоя с draft — 115.
    Без draft все 3 слоя — 100: при draft_with_offload=False выбирается он, а не draft + hyb."""
    from ramvi_engine.planning.automatic import PlanningOptions, choose_plan
    from ramvi_engine.planning.resources import ResourceEstimate

    def estimate(route: list[str], use_draft: bool, reserve: int) -> ResourceEstimate:
        return ResourceEstimate(
            weights=MemoryUsage(vram=10 + route.count("gpu") * 30, ram=5, mapped=10),
            draft=MemoryUsage(vram=15 if use_draft else 0),
            reserve=MemoryUsage(vram=reserve),
        )

    def choose(**kwargs: Any) -> Any:
        args: dict[str, Any] = dict(
            layers=3,
            estimate=estimate,
            vram_available=100,
            ram_available=100,
            maxt=2048,
            architecture="qwen3",
            model_format="gguf",
            cpu=True,
            draft_requested=True,
            options=PlanningOptions(reserve_bytes=0, max_batch=16),
        )
        args.update(kwargs)
        return choose_plan(**args)

    before = choose(cpu_fraction=1.0)
    assert (before.resident, before.draft_enabled) == (2, True)
    assert before.placement[2] == "hyb"  # с draft процессорный слой становится гибридным
    after = choose(draft_with_offload=False)
    assert (after.resident, after.draft_enabled) == (3, False)
    small = choose(draft_with_offload=False, vram_available=60, cpu_fraction=1.0)
    assert (small.resident, small.draft_enabled) == (1, False)
    assert small.placement == ("gpu", "cpu", "cpu")  # без draft нет hyb; при доле 1.0 нет stream
    forced = choose(
        draft_with_offload=False,
        cpu_fraction=1.0,
        options=PlanningOptions(reserve_bytes=0, max_batch=16, draft="on"),
    )
    assert (forced.resident, forced.draft_enabled) == (2, True)  # явное «всегда» сильнее правила
    assert forced.placement[2] == "hyb"
    full = choose(draft_with_offload=False, vram_available=200)
    assert (full.resident, full.draft_enabled) == (3, True)  # всё в видеокарте — draft разрешён
