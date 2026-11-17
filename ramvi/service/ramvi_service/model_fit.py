"""4.4 MOD-03: статус модели для этого компьютера по плану движка.

Заголовок GGUF (каталог — прочитанный по сети, установленная модель — из файла)
отдаётся тому же автоматическому планировщику, что выбирает размещение слоёв при
загрузке (`planning.automatic.choose_plan` + `planning.resources.estimate_gguf`).
Бюджет — возможности компьютера: VRAM самой большой NVIDIA и вся RAM. Нехватка
свободной RAM сейчас — пометка («закройте программы»), а не отказ (`compat.py`).
Неполное размещение в VRAM — «Медленно», не «неподходящая» (приёмка MOD-03).

Угадыватель (draft) в оценку не входит: каталог показывает саму модель.
"""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_engine.planning.automatic import (
    NoMemoryPlan,
    PlanningOptions,
    automatic_reserve,
    choose_plan,
    offload_cpu_fraction,
    placement,
    unquantized_layers,
)
from ramvi_engine.planning.resources import MemoryUsage, ResourceEstimate, estimate_gguf

from ramvi_service import compat, speed
from ramvi_service.catalog.headers import HeaderStore, read_file_header
from ramvi_service.gguf_meta import GGUFMeta, NotGGUF
from ramvi_service.hardware import HardwareProfile
from ramvi_service.i18n import msg
from ramvi_service.onboarding import FILE_TYPES, SUPPORTED_ARCHITECTURES

MAXT = 2048  # контекст чата по умолчанию (ModelConfig.maxt)
MIB = 1024 * 1024
UNKNOWN = "compat.unknown"


def model_key(meta: GGUFMeta) -> str:
    """Ключ таблицы замеров `compat.MEASURED`: «qwen3-14b-q4_k_m»."""
    arch = str(meta.meta.get("general.architecture", ""))
    size = str(meta.meta.get("general.size_label", ""))
    quant = FILE_TYPES.get(meta.meta.get("general.file_type"), "")  # type: ignore[arg-type]
    return "-".join(p for p in (arch, size, quant) if p).lower()


def plan(
    meta: GGUFMeta, hw: HardwareProfile, *, maxt: int = MAXT, native_f16: bool = False
) -> tuple[dict[str, Any] | None, ResourceEstimate]:
    """План движка для этого железа; не помещается — None и оценка самого лёгкого плана."""
    arch = str(meta.meta["general.architecture"])
    layers = int(meta.meta[f"{arch}.block_count"])
    unquantized = unquantized_layers(meta)
    fraction = offload_cpu_fraction(unquantized)
    vram = max((g.vram_total_mb or 0 for g in hw.gpus if g.vendor == "NVIDIA"), default=0) * MIB
    ram = (hw.ram_total_mb or 0) * MIB

    def estimate(route: list[str], use_draft: bool, reserve: int) -> ResourceEstimate:
        value = estimate_gguf(meta, route, maxt, batch=16, native_f16=native_f16)
        return replace(value, reserve=MemoryUsage(vram=reserve))

    try:
        decision = choose_plan(
            layers,
            estimate,
            vram_available=vram,
            ram_available=ram,
            maxt=maxt,
            architecture=arch,
            model_format="gguf",
            cpu=True,
            draft_requested=False,
            options=PlanningOptions(),
            cpu_fraction=fraction,
            draft_with_offload=not unquantized,
        )
    except NoMemoryPlan:
        route = placement(
            layers,
            0,
            architecture=arch,
            model_format="gguf",
            cpu=True,
            draft_enabled=False,
            cpu_fraction=fraction,
        )
        return None, estimate(list(route), False, automatic_reserve(vram))
    counts = {mode: decision.placement.count(mode) for mode in ("gpu", "stream", "cpu", "hyb")}
    off_gpu = sum(
        meta.raw(name)[0].size
        for name in meta.tensors
        if name.startswith("blk.") and decision.placement[int(name.split(".")[1])] != "gpu"
    )
    layout: dict[str, Any] = {
        "layers": layers,
        "resident": decision.resident,
        **counts,
        "off_gpu_bytes": off_gpu,
        "placement": list(decision.placement),
    }
    return layout, decision.resources


def traffic(meta: GGUFMeta, route: list[str] | tuple[str, ...], paging: int = 0) -> speed.Traffic:
    """Байты весов на один токен по местам. У MoE из экспертов читаются только выбранные."""
    arch = str(meta.meta["general.architecture"])
    moe = arch == "qwen3moe"
    share = 1.0
    if moe:
        share = int(meta.meta[f"{arch}.expert_used_count"]) / int(meta.meta[f"{arch}.expert_count"])
    where = {"vram": 0, "cpu": 0, "bus": 0}
    for name in meta.tensors:
        if name.startswith("blk."):
            mode = route[int(name.split(".")[1])]
            place = {"gpu": "vram", "stream": "bus"}.get(mode, "cpu")  # hyb: один шаг — CPU
        elif name == "output.weight" or (
            name == "token_embd.weight" and "output.weight" not in meta.tensors
        ):
            place = "vram"  # выходной словарь в видеокарте; входные эмбеддинги — одна строка
        else:
            continue
        size = meta.raw(name)[0].size
        where[place] += int(size * share) if "_exps" in name else size
    return speed.Traffic(where["vram"], where["cpu"], where["bus"], paging, moe, len(route))


def _native_f16() -> bool:
    try:
        from ramvi_engine.cpu.native_check import native_cpu_available

        return bool(native_cpu_available())
    except Exception:  # нет numpy/ядра — считаем без C++ (оценка RAM выше, не ниже)
        return False


class ModelFit:
    """Статусы моделей каталога и установленных моделей; профиль железа кэшируется."""

    def __init__(
        self,
        headers: HeaderStore,
        hardware: Callable[[], HardwareProfile],
        *,
        ttl: float = 30.0,
        native_f16: Callable[[], bool] = _native_f16,
    ) -> None:
        self.headers, self.hardware, self.ttl = headers, hardware, ttl
        self._native = native_f16
        self._lock = threading.Lock()
        self._hw: tuple[float, HardwareProfile, bool] | None = None
        self._files: dict[tuple[str, int, int], GGUFMeta | None] = {}

    def _profile(self) -> tuple[HardwareProfile, bool]:
        with self._lock:
            if self._hw is None or time.monotonic() - self._hw[0] > self.ttl:
                self._hw = (time.monotonic(), self.hardware(), self._native())
            return self._hw[1], self._hw[2]

    def verdict(
        self, meta: GGUFMeta | None, name: str, file_bytes: int, downloaded: bool
    ) -> dict[str, Any]:
        if meta is None:
            return {"status": UNKNOWN, "reasons": [msg("compat.reason.no_header").json()]}
        hw, native = self._profile()
        arch = str(meta.meta.get("general.architecture", ""))
        try:
            layout, estimate = plan(meta, hw, native_f16=native)
        except NotImplementedError as error:  # тип матриц без ядра движка
            reason = msg("compat.reason.format", format=str(error).rsplit(" ", 1)[-1])
            return {"status": compat.UNSUPPORTED, "reasons": [reason.json()]}
        except (KeyError, ValueError) as error:
            if arch not in SUPPORTED_ARCHITECTURES:
                reason = msg("compat.reason.architecture")
            else:
                reason = msg("compat.reason.format", format=type(error).__name__)
            return {"status": compat.UNSUPPORTED, "reasons": [reason.json()]}
        candidate = compat.Candidate(
            name=name,
            file_format="gguf",
            file_bytes=file_bytes,
            estimate=estimate,
            downloaded=downloaded,
            architecture_supported=arch in SUPPORTED_ARCHITECTURES,
            model_id=model_key(meta),
            off_gpu_bytes=None if layout is None else int(layout["off_gpu_bytes"]),
            estimated_tok_s=self._estimate(meta, layout, estimate, hw),
        )
        result = compat.assess(hw, candidate)
        need = estimate.required
        return {
            "status": result.status,
            "reasons": [r.json() for r in result.reasons],
            "speed": result.speed.json(),
            "plan": None
            if layout is None
            else {k: v for k, v in layout.items() if k != "placement"},
            "memory": {"vram": need.vram, "ram": need.ram, "mapped": need.mapped},
        }

    @staticmethod
    def _estimate(
        meta: GGUFMeta,
        layout: dict[str, Any] | None,
        estimate: ResourceEstimate,
        hw: HardwareProfile,
    ) -> tuple[float, float] | None:
        if layout is None:
            return None
        need = estimate.required
        total = (hw.ram_total_mb or 0) * MIB
        paging = max(0, need.ram + need.mapped - total) if total else 0
        value = speed.tokens_per_second(
            traffic(meta, layout["placement"], paging), hw.cores_physical
        )
        if value is None:
            return None
        return speed.rounded(value[0]), speed.rounded(value[1])

    def for_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Запись каталога: только по сохранённому заголовку, без сети."""
        meta = self.headers.cached(entry["checksum"])
        return self.verdict(meta, entry["name"], int(entry["bytes"]), downloaded=False)

    def for_file(self, path: str | os.PathLike[str], name: str) -> dict[str, Any]:
        """Установленная модель: заголовок из файла (кэш по пути, размеру и времени)."""
        file = Path(path)
        try:
            stat = file.stat()
        except OSError:
            return {"status": UNKNOWN, "reasons": [msg("compat.reason.no_header").json()]}
        key = (str(file), stat.st_size, stat.st_mtime_ns)
        if key not in self._files:
            try:
                self._files[key] = read_file_header(file)
            except (OSError, NotGGUF):
                self._files[key] = None
        return self.verdict(self._files[key], name, stat.st_size, downloaded=True)
