"""Общая оценка памяти для плана и предварительного отбора моделей.

Только метаданные: без импорта torch/numpy, выделения памяти и проверки железа.
"""

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any


def _bytes(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: требуется неотрицательное целое число байт")


@dataclass(frozen=True)
class MemoryUsage:
    """RAM — собственные массивы; mapped — желательный файловый working set."""

    vram: int = 0
    ram: int = 0
    mapped: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _bytes(value, name)

    def __add__(self, other: "MemoryUsage") -> "MemoryUsage":
        return MemoryUsage(self.vram + other.vram, self.ram + other.ram, self.mapped + other.mapped)


@dataclass(frozen=True)
class ResourceEstimate:
    """Байты на один движок и его draft; reserve не является выделенной памятью."""

    weights: MemoryUsage = MemoryUsage()
    kv: MemoryUsage = MemoryUsage()
    persistent: MemoryUsage = MemoryUsage()
    streaming: MemoryUsage = MemoryUsage()
    cpu_workspace: MemoryUsage = MemoryUsage()
    draft: MemoryUsage = MemoryUsage()
    reserve: MemoryUsage = MemoryUsage()
    notes: tuple[str, ...] = ()

    @property
    def allocated(self) -> MemoryUsage:
        """Оценка массивов/working set без запаса."""
        return (
            self.weights
            + self.kv
            + self.persistent
            + self.streaming
            + self.cpu_workspace
            + self.draft
        )

    @property
    def required(self) -> MemoryUsage:
        """Бюджет с запасом, используемый планом и подбором."""
        return self.allocated + self.reserve

    def assess(self, *, vram_available: int, ram_available: int) -> dict[str, Any]:
        """Предварительное соответствие бюджету; это не гарантия загрузки."""
        _bytes(vram_available, "vram_available")
        _bytes(ram_available, "ram_available")
        need = self.required
        ram_need = need.ram + need.mapped
        return {
            "estimated_fit": need.vram <= vram_available and ram_need <= ram_available,
            "vram_shortfall": max(0, need.vram - vram_available),
            "ram_shortfall": max(0, ram_need - ram_available),
            "includes_file_working_set": True,
            "guaranteed": False,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый отчёт, общий для CLI, каталога и сервиса."""
        return {
            "schema_version": 1,
            "unit": "bytes",
            "kind": "estimate",
            "components": {
                name: asdict(getattr(self, name))
                for name in (
                    "weights",
                    "kv",
                    "persistent",
                    "streaming",
                    "cpu_workspace",
                    "draft",
                    "reserve",
                )
            },
            "allocated": asdict(self.allocated),
            "required": asdict(self.required),
            "notes": list(self.notes),
        }


def assess_candidates(
    candidates: dict[str, ResourceEstimate], *, vram_available: int, ram_available: int
) -> dict[str, dict[str, Any]]:
    """Общий примитив подбора: одинаковые бюджеты и правила для всех моделей."""
    _bytes(vram_available, "vram_available")
    _bytes(ram_available, "ram_available")
    return {
        name: estimate.assess(vram_available=vram_available, ram_available=ram_available)
        for name, estimate in candidates.items()
    }


def _validate(place: list[str], layers: int, maxt: int, batch: int) -> None:
    if len(place) != layers or any(p not in {"gpu", "cpu", "stream", "hyb"} for p in place):
        raise ValueError("неверное размещение слоёв")
    for name, value in (("maxt", maxt), ("batch", batch)):
        _bytes(value, name)
        if value < 1:
            raise ValueError(f"{name} должно быть положительным")


def estimate_hf(
    ck: Any,
    place: list[str],
    maxt: int,
    *,
    batch: int = 16,
    draft: MemoryUsage = MemoryUsage(),
    reserve: MemoryUsage = MemoryUsage(),
) -> ResourceEstimate:
    """HF Qwen3: 12-битные веса приблизительны, CPU KV/буферы fp32."""
    _validate(place, ck.L, maxt, batch)
    if "hyb" in place:
        raise ValueError("HF не имеет размещения hyb")
    shapes = ck.layer_params()
    elements = sum(a * b for a, b in shapes.values())
    packed = ceil(elements * 1.505)  # прежняя эмпирическая оценка исключений compression
    bf16 = elements * 2
    d, h, kv, hd, ff = ck.D, ck.H, ck.KV, ck.HD, ck.FF
    v = ck.cfg["vocab_size"]
    nc, ns, ng = (place.count(p) for p in ("cpu", "stream", "gpu"))
    kv_layer = 2 * kv * maxt * hd * 4
    qkv = (h + 2 * kv) * hd
    small = (2 * d + 2 * hd) * ck.L * 2 + d * 2
    scratch = batch * (2 * d + qkv + 2 * h * hd + 2 * ff) * 4 + 8
    head = v * d + v * 16
    host_small = nc * (2 * d + 2 * hd) * 4
    cpu_work = batch * (2 * d + qkv + h * hd + 3 * ff) * 4 if nc else 0
    return ResourceEstimate(
        weights=MemoryUsage(
            ng * packed + head,
            mapped=(nc * bf16 + (2 if getattr(ck, "tied", False) else 4) * v * d),
        ),
        kv=MemoryUsage((ng + ns) * kv_layer + (kv_layer if nc else 0), nc * kv_layer),
        persistent=MemoryUsage(
            small + scratch + batch * v * 4 + batch * 4 + hd * 2,
            host_small + (batch * d * 4 if nc else 0) + hd * 2,
        ),
        streaming=MemoryUsage((2 * packed if ns else 0) + (bf16 if nc else 0), ns * packed),
        cpu_workspace=MemoryUsage(ram=cpu_work),
        draft=draft,
        reserve=reserve,
        notes=(
            "HF compression/exceptions and stream padding are approximate",
            "mapped includes CPU layer files and embedding/head BF16 working set (tied tensor counted once)",
            "excludes loading peaks, allocator overhead, graphs, JIT and long-prefill temporaries; reserve required",
        ),
    )


def _aligned(value: int) -> int:
    return (value + 255) // 256 * 256


def _gpu_tensor(raw: Any, packed: bool = False) -> int:
    buf, n, k, kind = raw
    count = int(n) * int(k)
    if kind == "Q4_K":
        parts = [count // 2, count // 32, count // 32, count // 64]
    elif kind == "Q6_K":
        parts = [count // 2, count // 4, count // 16, count // 128]
    elif kind in ("F16", "F32"):
        parts = [count * (2 if kind == "F16" else 4)]
    else:
        raise NotImplementedError(f"GPU estimate: {kind}")
    return sum(_aligned(x) if packed else x for x in parts)


def _cpu_scales(raw: Any, native_f16: bool = False) -> int:
    """Собственная RAM процессорной матрицы сверх файла: масштабы Q4_K/Q6_K или копия в float32.

    3.18: с ядром C++ (`native_f16`) F16 считается прямо из файла — копии нет."""
    _, n, k, kind = raw
    count = int(n) * int(k)
    if kind in ("Q4_K", "Q6_K"):
        return count // 4
    if kind == "F16" and native_f16 and int(k) % 256 == 0:
        return 0
    return count * 4


MAPPED_STREAM_KINDS = ("F16", "F32")
STAGE_RAM = 4 * 10 * 2**20  # кольцо закреплённых буферов загрузчика (core.engine.Loader)


PREFILL_ROWS = 64  # B.6: ширина куска чтения запроса у плотного GGUF (gengine)


def estimate_gguf(
    g: Any,
    place: list[str],
    maxt: int,
    *,
    batch: int = 16,
    draft: MemoryUsage = MemoryUsage(),
    reserve: MemoryUsage = MemoryUsage(),
    native_f16: bool = False,
) -> ResourceEstimate:
    """GGUF Qwen3 dense/MoE по размерам тензоров без распаковки весов.

    3.18: подкачиваемые F16/F32 читаются из файла — это working set файла (`mapped`, может
    дочитываться с диска), а не собственная RAM; `native_f16` — процессор считает F16 из файла."""
    arch = g.meta["general.architecture"]
    if arch not in ("qwen3", "qwen3moe"):
        raise NotImplementedError(arch)

    def m(key: str) -> int:
        return int(g.meta[f"{arch}.{key}"])

    layers, d, h, kv, hd = (
        m(k)
        for k in (
            "block_count",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
        )
    )
    _validate(place, layers, maxt, batch)
    moe = arch == "qwen3moe"
    if moe and any(p in ("stream", "hyb") for p in place):
        raise ValueError("MoE не поддерживает stream/hyb")
    ff = m("expert_feed_forward_length" if moe else "feed_forward_length")
    experts = m("expert_count") if moe else 0
    topk = m("expert_used_count") if moe else 0
    out_name = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
    out = g.raw(out_name)
    v = out[1]
    gpu_w = _gpu_tensor(out)
    mapped_names = {"token_embd.weight"}
    gpu_kv = cpu_kv = cpu_aux = stream_ram = 0
    staged = False
    stream_max: dict[str, int] = {}
    attn = ("attn_q", "attn_k", "attn_v", "attn_output")
    matrices = attn + (
        ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
        if moe
        else ("ffn_gate", "ffn_up", "ffn_down")
    )
    small_per = (2 * d + 2 * hd) * 4  # g.full returns float32
    small_gpu = 0
    for i, mode in enumerate(place):
        cpu = mode in ("cpu", "hyb")
        if mode != "cpu":
            gpu_kv += 2 * kv * maxt * hd * 2
        if cpu:
            cpu_kv += 2 * kv * maxt * hd * 4
            cpu_aux += small_per + (experts * d * 4 if moe else 0)
        if not moe or mode == "gpu":
            small_gpu += small_per
        for matrix in matrices:
            name = f"blk.{i}.{matrix}.weight"
            raw = g.raw(name)
            if cpu:
                mapped_names.add(name)
                if not moe or matrix in attn:
                    cpu_aux += _cpu_scales(raw, native_f16)
            if mode == "gpu":
                gpu_w += _gpu_tensor(raw)
            elif mode in ("stream", "hyb"):
                size = _gpu_tensor(raw, packed=True)
                if raw[3] in MAPPED_STREAM_KINDS:
                    mapped_names.add(name)
                    staged = True
                else:
                    stream_ram += size
                stream_max[matrix] = max(stream_max.get(matrix, 0), size)
        if moe and mode == "gpu":
            gpu_w += experts * d * 4  # router is decoded fp32
    mapped = sum(g.raw(name)[0].size for name in mapped_names)
    nc = sum(mode in ("cpu", "hyb") for mode in place)
    qkv = (h + 2 * kv) * hd
    scratch = batch * (2 * d + qkv + 2 * h * hd + v) * 4 + 8
    if moe:
        scratch += batch * topk * (2 * ff + d) * 4
        cpu_aux += batch * (2 * d + topk * (ff + d)) * 4
    else:
        scratch += batch * 3 * ff * 4
        # B.6: рабочие буферы слоя — на 64 слова (чтение запроса кусками по 64), логиты — на 16;
        # входы тензорных блоков: 3 части bf16 по 64 строки и суммы подблоков для каждой длины K.
        wide = PREFILL_ROWS - batch if batch == 16 else 0
        scratch += wide * (2 * d + qkv + 2 * h * hd + 3 * ff) * 4
        scratch += sum(PREFILL_ROWS * (k * 3 * 2 + k // 32 * 4) for k in {d, h * hd, ff})
        cpu_aux += batch * d * 4 if nc else 0
    cpu_work = batch * (2 * d + qkv + h * hd + 3 * ff + (d + h * hd + ff) // 32) * 4 if nc else 0
    if moe and nc:
        cpu_work += batch * (4 * experts + topk * (2 * ff + d)) * 4
    return ResourceEstimate(
        weights=MemoryUsage(gpu_w, mapped=mapped),
        kv=MemoryUsage(gpu_kv, cpu_kv),
        persistent=MemoryUsage(small_gpu + scratch + d * 4 + hd * 2, cpu_aux + hd * 2),
        streaming=MemoryUsage(
            2 * sum(stream_max.values()), stream_ram + (STAGE_RAM if staged else 0)
        ),
        cpu_workspace=MemoryUsage(ram=cpu_work),
        draft=draft,
        reserve=reserve,
        notes=(
            "stream RAM excludes Arena chunk slack/pinning overhead",
            "mapped counts distinct embedding and CPU-layer tensors once; GPU repack includes Q4 scale expansion",
            "excludes loading peaks, allocator overhead, graphs, JIT and long-prefill temporaries; reserve required",
        ),
    )
