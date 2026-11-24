"""4.4 MOD-03: статус модели для этого компьютера по плану движка.

Заголовок GGUF (каталог — прочитанный по сети, установленная модель — из файла)
отдаётся тому же автоматическому планировщику, что выбирает размещение слоёв при
загрузке (`planning.automatic.choose_plan` + `planning.resources.estimate_gguf`).
Бюджет — возможности компьютера: VRAM самой большой NVIDIA и вся RAM. Нехватка
свободной RAM сейчас — пометка («закройте программы»), а не отказ (`compat.py`).
Неполное размещение в VRAM — «Медленно», не «неподходящая» (приёмка MOD-03).

Угадыватель (draft) в оценку не входит: если он не помещается, план движка его
выключает (`choose_plan`, draft=auto), так что выполнимость от него не зависит.

4.5 AUTO-02/03: контекст берётся из профиля задачи (KV-кэш); кроме оценки на весь
компьютер считается план на **свободную сейчас** VRAM и RAM (`now`) и проверяется,
подходит ли модель задаче (`task`: нужные возможности и длина контекста модели).
"""

import os
import re
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
from ramvi_engine.planning.multigpu import usable_vram
from ramvi_engine.planning.resources import MemoryUsage, ResourceEstimate, estimate_gguf

from ramvi_service import compat, speed
from ramvi_service.catalog.headers import HeaderStore, read_file_header
from ramvi_service.catalog.support import capabilities as engine_capabilities
from ramvi_service.gguf_meta import GGUFMeta, NotGGUF
from ramvi_service.hardware import HardwareProfile
from ramvi_service.i18n import msg
from ramvi_service.onboarding import FILE_TYPES, SUPPORTED_ARCHITECTURES
from ramvi_service.task_profile import TaskProfile

MAXT = 2048  # контекст чата по умолчанию (ModelConfig.maxt)
MIB = 1024 * 1024
UNKNOWN = "compat.unknown"
MISSING = "compat.missing"  # B.5: файла модели нет по сохранённому пути


def size_billions(meta: GGUFMeta) -> float | None:
    """Число параметров из `general.size_label` («14B», «0.6B», «30B-A3B» → 30)."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*([BM])", str(meta.meta.get("general.size_label", "")))
    if match is None:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2) == "M" else value


def model_key(meta: GGUFMeta) -> str:
    """Ключ таблицы замеров `compat.MEASURED`: «qwen3-14b-q4_k_m»."""
    arch = str(meta.meta.get("general.architecture", ""))
    size = str(meta.meta.get("general.size_label", ""))
    quant = FILE_TYPES.get(meta.meta.get("general.file_type"), "")  # type: ignore[arg-type]
    return "-".join(p for p in (arch, size, quant) if p).lower()


def budget(hw: HardwareProfile, *, now: bool = False, multi: bool = False) -> tuple[int, int]:
    """(VRAM, RAM) в байтах: возможности компьютера или свободное прямо сейчас.

    4.10: `multi` — модель, которую движок умеет раскладывать по нескольким картам (плотная
    Qwen3 GGUF): основная карта плюс остальные за вычетом их резерва, как в плане движка.
    Иначе — одна, самая большая карта."""
    cards = [g for g in hw.gpus if g.vendor == "NVIDIA"]
    sizes = [(g.vram_free_mb if now else g.vram_total_mb) or 0 for g in cards]
    ram = (hw.ram_available_mb if now else hw.ram_total_mb) or 0
    if multi and len(sizes) > 1:
        vram = usable_vram([s * MIB for s in sizes], automatic_reserve)
        return vram, ram * MIB
    return max(sizes, default=0) * MIB, ram * MIB


def plan(
    meta: GGUFMeta,
    hw: HardwareProfile,
    *,
    maxt: int = MAXT,
    native_f16: bool = False,
    now: bool = False,
) -> tuple[dict[str, Any] | None, ResourceEstimate]:
    """План движка для этого железа; не помещается — None и оценка самого лёгкого плана."""
    arch = str(meta.meta["general.architecture"])
    layers = int(meta.meta[f"{arch}.block_count"])
    unquantized = unquantized_layers(meta)
    fraction = offload_cpu_fraction(unquantized)
    vram, ram = budget(hw, now=now, multi=arch == "qwen3")

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
            hybrid_prefill=not unquantized,  # B.6: как движок — длинный запрос на видеокарте
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


def apply_control(verdict: dict[str, Any], result: dict[str, Any]) -> None:
    """4.7: замер на этом ПК вместо оценки; нестабильный прогон — пометка и причина."""
    low_high = result.get("tok_s_range")
    if low_high:
        low, high = speed.rounded(low_high[0]), speed.rounded(low_high[1])
        verdict["speed"] = msg("compat.speed_here", low=f"{low:g}", high=f"{high:g}").json()
        verdict["tok_s"] = {"low": low_high[0], "high": low_high[1], "measured": True}
    stable = bool(result.get("stable"))
    verdict["control"] = {
        "stable": stable,
        "tok_s": result.get("tok_s"),
        "ttft_s": result.get("ttft_s"),
        "peak_ram": result.get("peak_ram"),
        "peak_vram": result.get("peak_vram"),
        "page_faults": result.get("page_faults"),
        "resident": result.get("resident"),
        "rolled_back_from": result.get("rolled_back_from"),
    }
    if not stable:
        verdict["reasons"].append(msg("compat.reason.unstable").json())
    if result.get("rolled_back_from") is not None:
        reason = msg(
            "compat.reason.rolled_back",
            before=result["rolled_back_from"],
            after=result.get("resident"),
        )
        verdict["reasons"].append(reason.json())


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
        profile: Callable[[], TaskProfile] = TaskProfile,
        local_result: Callable[[str, HardwareProfile, int], dict[str, Any] | None] | None = None,
    ) -> None:
        self.headers, self.hardware, self.ttl = headers, hardware, ttl
        self.profile = profile
        self.local_result = local_result  # 4.7: замер контрольного прогона на этом ПК
        self._native = native_f16
        self._lock = threading.Lock()
        self._hw: tuple[float, HardwareProfile, bool] | None = None
        self._files: dict[tuple[str, int, int], GGUFMeta | None] = {}

    def hardware_profile(self) -> HardwareProfile:
        """Профиль железа из кэша (обновляется раз в `ttl` секунд)."""
        return self._profile()[0]

    def _profile(self) -> tuple[HardwareProfile, bool]:
        with self._lock:
            if self._hw is None or time.monotonic() - self._hw[0] > self.ttl:
                self._hw = (time.monotonic(), self.hardware(), self._native())
            return self._hw[1], self._hw[2]

    def verdict(
        self,
        meta: GGUFMeta | None,
        name: str,
        file_bytes: int,
        downloaded: bool,
        capabilities: tuple[str, ...] | list[str] = (),
        task: TaskProfile | None = None,
    ) -> dict[str, Any]:
        """Статус модели; `task` — другой профиль вместо сохранённого (4.8: «а если…»)."""
        if meta is None:
            return {"status": UNKNOWN, "reasons": [msg("compat.reason.no_header").json()]}
        hw, native = self._profile()
        task = task or self.profile()
        context = task.effective_context
        arch = str(meta.meta.get("general.architecture", ""))
        try:
            layout, estimate = plan(meta, hw, maxt=context, native_f16=native)
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
            # Замер снят при контексте 2048; при другом контексте — оценка по плану.
            model_id=model_key(meta) if context == compat.MEASURED_CONTEXT else "",
            off_gpu_bytes=None if layout is None else int(layout["off_gpu_bytes"]),
            estimated_tok_s=self._estimate(meta, layout, estimate, hw),
            vram_mb=budget(hw, multi=arch == "qwen3")[0] // MIB,
        )
        result = compat.assess(hw, candidate)
        tok_s = (
            compat.speed_range(hw, candidate, result.status)
            if result.status in compat.RUNNABLE
            else None
        )
        need = estimate.required
        reasons = [r.json() for r in result.reasons]
        now = self._now(meta, hw, context, native, result.status)
        if now is not None and not now["fits"]:
            reasons.append(msg("compat.reason.busy_now").json())
        return {
            "status": result.status,
            "reasons": reasons,
            "speed": result.speed.json(),
            "plan": None
            if layout is None
            else {k: v for k, v in layout.items() if k != "placement"},
            "memory": {"vram": need.vram, "ram": need.ram, "mapped": need.mapped},
            "context": context,
            "now": now,
            "task": self._task(meta, task, capabilities),
            # 4.6: числа для ранжирования
            "tok_s": None
            if tok_s is None
            else {"low": tok_s[0], "high": tok_s[1], "measured": tok_s[2]},
            "model": {
                "key": model_key(meta),
                "size_b": size_billions(meta),
                "quant": FILE_TYPES.get(meta.meta.get("general.file_type"), ""),  # type: ignore[arg-type]
            },
        }

    @staticmethod
    def _now(
        meta: GGUFMeta, hw: HardwareProfile, context: int, native: bool, status: str
    ) -> dict[str, Any] | None:
        """AUTO-03: помещается ли план в свободную сейчас память (а не только во всю)."""
        if status not in compat.RUNNABLE:
            return None
        layout, _ = plan(meta, hw, maxt=context, native_f16=native, now=True)
        if layout is None:
            return {"fits": False}
        return {"fits": True, "gpu": layout["gpu"], "layers": layout["layers"]}

    @staticmethod
    def _task(
        meta: GGUFMeta, task: TaskProfile, capabilities: tuple[str, ...] | list[str]
    ) -> dict[str, Any]:
        """Подходит ли модель профилю: нужные возможности и длина контекста модели."""
        reasons = [
            msg("task.reason.capability", capability=need).json()
            for need in task.needs
            if need not in capabilities
        ]
        arch = meta.meta.get("general.architecture", "")
        limit = meta.meta.get(f"{arch}.context_length")
        if isinstance(limit, int) and limit < task.effective_context:
            reasons.append(
                msg("task.reason.context", model=limit, need=task.effective_context).json()
            )
        return {"suits": not reasons, "reasons": reasons}

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
        return self.verdict(
            meta,
            entry["name"],
            int(entry["bytes"]),
            downloaded=False,
            capabilities=entry.get("capabilities", ()),
        )

    def for_file(
        self,
        path: str | os.PathLike[str],
        name: str,
        model_id: str | None = None,
        task: TaskProfile | None = None,
    ) -> dict[str, Any]:
        """Установленная модель: заголовок из файла (кэш по пути, размеру и времени);
        есть замер контрольного прогона на этом ПК (4.7) — скорость из него."""
        file = Path(path)
        try:
            stat = file.stat()
        except FileNotFoundError:  # B.5: перенос на другой ПК, отключённый диск
            return {
                "status": MISSING,
                "reasons": [msg("compat.reason.missing", path=str(file)).json()],
            }
        except OSError:
            return {"status": UNKNOWN, "reasons": [msg("compat.reason.no_header").json()]}
        key = (str(file), stat.st_size, stat.st_mtime_ns)
        if key not in self._files:
            try:
                self._files[key] = read_file_header(file)
            except (OSError, NotGGUF):
                self._files[key] = None
        meta = self._files[key]
        arch = "" if meta is None else str(meta.meta.get("general.architecture", ""))
        caps = engine_capabilities("gguf", arch)
        verdict = self.verdict(meta, name, stat.st_size, True, caps, task=task)
        if model_id is not None and self.local_result is not None and "context" in verdict:
            found = self.local_result(model_id, self.hardware_profile(), verdict["context"])
            if found is not None:
                apply_control(verdict, found)
        return verdict
