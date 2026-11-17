"""Оценка ресурсов на решаемых вручную примерах, без GPU."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from ramvi_engine.planning.resources import (
    MemoryUsage,
    ResourceEstimate,
    assess_candidates,
    estimate_gguf,
    estimate_hf,
)


def hf() -> SimpleNamespace:
    return SimpleNamespace(
        L=3,
        D=256,
        H=2,
        KV=1,
        HD=128,
        FF=512,
        cfg={"vocab_size": 256},
        tied=True,
        layer_params=lambda: {
            "qkv": (512, 256),
            "o": (256, 256),
            "gu": (1024, 256),
            "down": (256, 512),
        },
    )


def test_manual_budget_and_candidate_selection() -> None:
    estimate = ResourceEstimate(
        weights=MemoryUsage(100, 30, 20),
        kv=MemoryUsage(50, 10),
        streaming=MemoryUsage(20, 5),
        cpu_workspace=MemoryUsage(ram=15),
        draft=MemoryUsage(40, 8),
        reserve=MemoryUsage(10, 2),
    )
    assert estimate.required == MemoryUsage(220, 70, 20)
    assert estimate.allocated == MemoryUsage(210, 68, 20)
    result = assess_candidates({"one": estimate}, vram_available=219, ram_available=89)["one"]
    assert result["vram_shortfall"] == result["ram_shortfall"] == 1
    assert not result["estimated_fit"] and not result["guaranteed"]
    assert estimate.assess(vram_available=220, ram_available=90)["estimated_fit"]
    assert json.loads(json.dumps(estimate.to_dict()))["required"]["vram"] == 220


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_memory(bad: int) -> None:
    with pytest.raises(ValueError):
        MemoryUsage(vram=bad)
    with pytest.raises(ValueError):
        assess_candidates({}, vram_available=bad, ram_available=0)


def test_hf_kv_and_buffers_follow_actual_placement() -> None:
    ck = hf()
    e = estimate_hf(ck, ["gpu", "cpu", "stream"], 10)
    # Один GPU и один stream KV + один общий staging KV fp32, CPU KV отдельно.
    kv = 2 * 1 * 10 * 128 * 4
    assert e.kv == MemoryUsage(3 * kv, kv)
    all_gpu = estimate_hf(ck, ["gpu"] * 3, 10)
    assert all_gpu.streaming == MemoryUsage()
    assert all_gpu.cpu_workspace == MemoryUsage()
    assert all_gpu.kv == MemoryUsage(3 * kv)
    bigger = estimate_hf(ck, ["gpu", "cpu", "stream"], 10003)
    assert bigger.kv.vram == e.kv.vram * 10003 // 10
    assert bigger.kv.ram == e.kv.ram * 10003 // 10
    untied = hf()
    untied.tied = False
    assert estimate_hf(untied, ["gpu"] * 3, 10).weights.mapped == 2 * all_gpu.weights.mapped
    with pytest.raises(ValueError):
        estimate_hf(ck, ["hyb"] * 3, 10)


class FakeGGUF:
    def __init__(self, moe: bool = False) -> None:
        arch = "qwen3moe" if moe else "qwen3"
        self.meta: dict[str, Any] = {"general.architecture": arch}
        for key, value in {
            "block_count": 3,
            "embedding_length": 256,
            "attention.head_count": 2,
            "attention.head_count_kv": 1,
            "attention.key_length": 128,
            "feed_forward_length": 512,
            "expert_feed_forward_length": 512,
            "expert_count": 4,
            "expert_used_count": 2,
        }.items():
            self.meta[f"{arch}.{key}"] = value
        self.tensors = {"token_embd.weight": None}

    def raw(self, name: str) -> tuple[SimpleNamespace, int, int, str]:
        # 256x256 Q4_K: 256 блоков по 144 байта; репак GPU = 256*148.
        return SimpleNamespace(size=256 * 144), 256, 256, "Q4_K"


def test_gguf_repack_alignment_and_duplicate_hybrid_kv() -> None:
    g = FakeGGUF()
    e = estimate_gguf(g, ["gpu", "hyb", "stream"], 10)
    gpu_matrix = 256 * 148
    assert e.weights.vram == 8 * gpu_matrix  # output + 7 matrices
    assert e.weights.mapped == 8 * 256 * 144  # embedding + 7 CPU matrices
    assert e.streaming.ram == 14 * gpu_matrix
    assert e.streaming.vram == 14 * gpu_matrix  # two slots, one max per matrix
    assert e.kv == MemoryUsage(3 * 2 * 10 * 128 * 2, 2 * 10 * 128 * 4)
    assert e.persistent.ram > 7 * 256 * 256 // 4  # expanded CPU scales
    moe = estimate_gguf(FakeGGUF(True), ["cpu", "cpu", "gpu"], 10)
    assert moe.kv == MemoryUsage(2 * 10 * 128 * 2, 2 * 2 * 10 * 128 * 4)
    assert moe.streaming == MemoryUsage()
    with pytest.raises(ValueError):
        estimate_gguf(FakeGGUF(True), ["gpu", "stream", "cpu"], 10)


def test_plan_and_selection_use_identical_estimate() -> None:
    pytest.importorskip("torch")
    from ramvi_engine.planning.plan import make_plan

    ck = hf()
    with (
        patch("torch.cuda.mem_get_info", return_value=(10**12, 10**12)),
        patch("ramvi_engine.planning.plan.ram_info", return_value=(10**12, 10**12)),
    ):
        plan = make_plan(ck, 10003, force_resident=1, log=lambda _: None)  # type: ignore[no-untyped-call]
    e = estimate_hf(ck, plan["place"], 10003, reserve=MemoryUsage(vram=int(0.8 * 2**30)))
    assert plan["resources"] == e.to_dict()
    assert (
        plan["resource_fit"]
        == assess_candidates({"test": e}, vram_available=10**12, ram_available=10**12)["test"]
    )


def test_gguf_plan_reports_the_same_budget() -> None:
    pytest.importorskip("torch")
    from ramvi_engine.planning.plan import plan_gguf

    g = FakeGGUF()
    with (
        patch("ramvi_engine.formats.gguf.GGUF", return_value=g),
        patch("torch.cuda.is_available", return_value=False),
        patch("ramvi_engine.planning.plan.ram_info", return_value=(10**12, 10**12)),
    ):
        plan = plan_gguf("unused.gguf", maxt=10003, force_resident=1, log=lambda _: None)  # type: ignore[no-untyped-call]
    expected = estimate_gguf(g, plan["place"], 10003, reserve=MemoryUsage(vram=int(0.9 * 2**30)))
    assert plan["resources"] == expected.to_dict()
    assert not plan["resource_fit"]["estimated_fit"]  # no CUDA, even with ample RAM


def test_draft_reserve_and_batch_are_counted_once() -> None:
    base = estimate_gguf(FakeGGUF(), ["gpu"] * 3, 23, batch=1)
    draft = ResourceEstimate(weights=MemoryUsage(100, 50, 25), kv=MemoryUsage(20, 10))
    bigger = estimate_gguf(
        FakeGGUF(), ["gpu"] * 3, 23, batch=16, draft=draft.allocated, reserve=MemoryUsage(123, 456)
    )
    assert bigger.draft == MemoryUsage(120, 60, 25)
    assert bigger.required.vram == bigger.allocated.vram + 123
    assert bigger.required.ram == bigger.allocated.ram + 456
    assert bigger.kv == base.kv
    assert bigger.persistent.vram > base.persistent.vram
    assert base.cpu_workspace.ram == bigger.cpu_workspace.ram == 0
    for _ in range(3):
        assert bigger == estimate_gguf(
            FakeGGUF(),
            ["gpu"] * 3,
            23,
            batch=16,
            draft=draft.allocated,
            reserve=MemoryUsage(123, 456),
        )


def test_plain_cpu_matrices_include_decoded_copies() -> None:
    class PlainGGUF(FakeGGUF):
        def raw(self, name: str) -> tuple[SimpleNamespace, int, int, str]:
            return SimpleNamespace(size=256 * 256 * 2), 256, 256, "F16"

    e = estimate_gguf(PlainGGUF(), ["cpu"] * 3, 7)
    assert e.persistent.ram >= 3 * 7 * 256 * 256 * 4
    assert e.weights.mapped == (3 * 7 + 1) * 256 * 256 * 2


@pytest.mark.parametrize("kind,block_bytes", [("Q6_K", 210), ("F32", 1024)])
def test_other_supported_weight_formats(kind: str, block_bytes: int) -> None:
    class OtherGGUF(FakeGGUF):
        def raw(self, name: str) -> tuple[SimpleNamespace, int, int, str]:
            return SimpleNamespace(size=256 * block_bytes), 256, 256, kind

    gpu = estimate_gguf(OtherGGUF(), ["gpu"] * 3, 100003)
    assert gpu.weights.vram == (3 * 7 + 1) * 256 * block_bytes
    assert gpu.kv.vram == 3 * 2 * 100003 * 128 * 2
    assert gpu.kv.ram == 0


def test_resource_contract_imports_without_compute_libraries() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'triton', 'transformers'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from ramvi_engine.planning.resources import ResourceEstimate, MemoryUsage, assess_candidates
assert assess_candidates({'empty': ResourceEstimate()}, vram_available=0, ram_available=0)['empty']['estimated_fit']
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
