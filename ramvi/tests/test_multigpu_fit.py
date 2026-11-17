"""4.10: оценка совместимости учитывает несколько видеокарт для плотной Qwen3 GGUF."""

from pathlib import Path
from typing import Any

from ramvi_service import compat
from ramvi_service.catalog.headers import HeaderStore
from ramvi_service.gguf_meta import GGUFMeta
from ramvi_service.hardware import GPUInfo
from ramvi_service.model_fit import ModelFit
from test_compat import LAPTOP

Q4_K, Q6_K, F32 = 12, 14, 0


def qwen3(layers: int, d: int, heads: int, kv: int, ff: int, arch: str = "qwen3") -> GGUFMeta:
    """Заголовок с размерами настоящей модели (веса не нужны)."""
    hd, vocab = 128, 151936
    meta = {
        "general.architecture": arch,
        "general.size_label": "32B",
        "general.file_type": 15,
        f"{arch}.block_count": layers,
        f"{arch}.embedding_length": d,
        f"{arch}.attention.head_count": heads,
        f"{arch}.attention.head_count_kv": kv,
        f"{arch}.attention.key_length": hd,
        f"{arch}.attention.value_length": hd,
        f"{arch}.feed_forward_length": ff,
        f"{arch}.context_length": 40960,
    }
    tensors: dict[str, tuple[tuple[int, ...], int]] = {
        "token_embd.weight": ((d, vocab), Q4_K),
        "output_norm.weight": ((d,), F32),
        "output.weight": ((d, vocab), Q6_K),
    }
    for i in range(layers):
        for name, dims in (
            ("attn_norm", (d,)),
            ("ffn_norm", (d,)),
            ("attn_q_norm", (hd,)),
            ("attn_k_norm", (hd,)),
        ):
            tensors[f"blk.{i}.{name}.weight"] = (dims, F32)
        matrices: tuple[tuple[str, tuple[int, ...]], ...] = (
            ("attn_q", (d, heads * hd)),
            ("attn_k", (d, kv * hd)),
            ("attn_v", (d, kv * hd)),
            ("attn_output", (heads * hd, d)),
            ("ffn_gate", (d, ff)),
            ("ffn_up", (d, ff)),
            ("ffn_down", (ff, d)),
        )
        for name, shape in matrices:
            tensors[f"blk.{i}.{name}.weight"] = (shape, Q4_K)
    return GGUFMeta(meta, tensors)


def profile(*cards_mb: int) -> Any:
    gpus = tuple(GPUInfo(f"RTX {i}", "NVIDIA", mb, mb, "580") for i, mb in enumerate(cards_mb))
    return LAPTOP.__class__(
        **{**LAPTOP.__dict__, "gpus": gpus, "ram_total_mb": 32768, "ram_available_mb": 30000}
    )


def verdict(meta: GGUFMeta, hw: Any) -> dict[str, Any]:
    fit = ModelFit(HeaderStore(Path("unused")), lambda: hw, native_f16=lambda: True)
    return fit.verdict(meta, "32B", 20 * 10**9, False, ("chat",))


def test_model_too_big_for_one_card_fits_two() -> None:
    meta = qwen3(64, 5120, 64, 8, 25600)  # Qwen3-32B Q4: ~19 ГБ
    one = verdict(meta, profile(12288))
    two = verdict(meta, profile(12288, 12288))
    assert one["status"] == compat.SLOW and one["plan"]["gpu"] < 64
    assert two["status"] == compat.RECOMMENDED, two
    assert two["plan"]["gpu"] == 64 and two["plan"]["off_gpu_bytes"] == 0


def test_moe_stays_on_one_card() -> None:
    """Несколько карт умеет только GGUF-движок плотных Qwen3; для MoE — одна карта."""
    meta = qwen3(48, 2048, 32, 4, 768, arch="qwen3moe")
    meta.meta.update(
        {
            "qwen3moe.expert_count": 128,
            "qwen3moe.expert_used_count": 8,
            "qwen3moe.expert_feed_forward_length": 768,
        }
    )
    for i in range(48):
        for name in ("gate", "up", "down"):
            del meta.tensors[f"blk.{i}.ffn_{name}.weight"]
        meta.tensors[f"blk.{i}.ffn_gate_exps.weight"] = ((2048, 768, 128), Q4_K)
        meta.tensors[f"blk.{i}.ffn_up_exps.weight"] = ((2048, 768, 128), Q4_K)
        meta.tensors[f"blk.{i}.ffn_down_exps.weight"] = ((768, 2048, 128), Q4_K)
        meta.tensors[f"blk.{i}.ffn_gate_inp.weight"] = ((2048, 128), F32)
    one = verdict(meta, profile(8192))
    two = verdict(meta, profile(8192, 8192))
    assert one["plan"]["gpu"] == two["plan"]["gpu"]
