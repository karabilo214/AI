"""4.4 MOD-03: статус совместимости модели с этим компьютером и фильтры.

Неполное размещение в VRAM — не «неподходящая», а «Медленно»: движок
считает остальные слои на CPU. «Не хватает RAM» — только когда не помещается
собственная RAM движка; веса, читаемые из файла (`mapped`), при нехватке RAM
дочитываются с диска — это «Медленно» с причиной (так работают qwen3:30b и
qwen3-8b-f16 на 16 ГБ RAM, планировщик движка тоже это допускает). 4.10: модель, которую движок раскладывает по нескольким картам, получает их общую
видеопамять (`Candidate.vram_mb`); иначе — одна, самая большая NVIDIA.

3.14: статусы — ключи каталога `compat.*`, причины и объяснения — сообщения `Msg`
(ключ + параметры); текст на языке пользователя собирает оболочка.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ramvi_engine.planning.resources import ResourceEstimate

from ramvi_service.hardware import HardwareProfile
from ramvi_service.i18n import Msg, msg

MIB = 1024 * 1024
DISK_RESERVE = 512 * MIB
SUPPORTED_FORMATS = ("gguf", "safetensors")

RECOMMENDED = "compat.recommended"
SLOW = "compat.slow"
NO_VRAM = "compat.no_vram"
NO_RAM = "compat.no_ram"
NO_DISK = "compat.no_disk"
UNSUPPORTED = "compat.unsupported"
ORDER = (RECOMMENDED, SLOW, NO_VRAM, NO_RAM, NO_DISK, UNSUPPORTED)


@dataclass(frozen=True)
class Candidate:
    name: str
    file_format: str
    file_bytes: int
    estimate: ResourceEstimate  # план для этого железа (2.2)
    downloaded: bool = False
    architecture_supported: bool = True
    model_id: str = ""  # ключ для таблицы замеров, например "qwen3-14b"
    measured_tok_s: tuple[float, float] | None = None  # контрольный прогон на этом ПК (4.7)
    params_b: float | None = None  # число параметров, млрд; не оценка качества
    # Байт весов слоёв вне видеокарты по плану; None — судить по estimate.weights.
    # Таблица входных эмбеддингов движка всегда читается из файла и сюда не входит.
    off_gpu_bytes: int | None = None
    # Оценка по плану и пропускной способности памяти (speed.py); замер важнее.
    estimated_tok_s: tuple[float, float] | None = None
    # 4.10: видеопамять, которую движок может отдать модели (несколько карт); None — одна карта.
    vram_mb: int | None = None


# Только реальные замеры: (подстрока имени NVIDIA, model_id, статус) -> (мин, макс) ток/с.
# Все сняты при контексте MEASURED_CONTEXT; при другом контексте план другой — тогда оценка.
MEASURED_CONTEXT = 2048
MEASURED = {
    # ASUS RTX 5060 8 ГБ + 16 ГБ RAM, greedy, контекст 2048, 30 вопросов,
    # полные ряды этапов 1–2: 7.80–9.11 ток/с (docs/reports/STAGE2_ACCEPTANCE.md, CONTEXT.md).
    # Ключ — `model_fit.model_key`: архитектура, размер и квант из заголовка GGUF.
    # Замер с автоугадывателем qwen3:0.6b, как работает чат.
    ("RTX 5060", "qwen3-14b-q4_k_m", SLOW): (7.8, 9.1),
}


@dataclass(frozen=True)
class Verdict:
    name: str
    status: str  # ключ compat.*
    reasons: tuple[Msg, ...]
    speed: Msg = msg("compat.speed_unknown")


def _gpu_vram_mb(hw: HardwareProfile) -> int | None:
    sizes = [g.vram_total_mb for g in hw.gpus if g.vendor == "NVIDIA" and g.vram_total_mb]
    return max(sizes) if sizes else None


def _range(hw: HardwareProfile, c: Candidate, status: str) -> tuple[float, float] | None:
    if c.measured_tok_s is not None:
        return c.measured_tok_s
    gpu = " ".join(g.name for g in hw.gpus if g.vendor == "NVIDIA")
    return next(
        (v for (g, m, st), v in MEASURED.items() if g in gpu and m == c.model_id and st == status),
        None,
    )


def _speed(hw: HardwareProfile, c: Candidate, status: str) -> Msg:
    rng = _range(hw, c, status)
    if rng is None and c.estimated_tok_s is not None:
        low, high = c.estimated_tok_s
        return msg("compat.speed_estimate", low=f"{low:g}", high=f"{high:g}")
    if rng is None:
        return msg("compat.speed_unknown")
    return msg("compat.speed_range", low=f"{rng[0]:g}", high=f"{rng[1]:g}")


def speed_range(hw: HardwareProfile, c: Candidate, status: str) -> tuple[float, float, bool] | None:
    """(нижняя, верхняя, замер ли) ток/с: замер, иначе оценка; None — ничего нет."""
    rng = _range(hw, c, status)
    if rng is not None:
        return rng[0], rng[1], True
    if c.estimated_tok_s is not None:
        return c.estimated_tok_s[0], c.estimated_tok_s[1], False
    return None


def assess(hw: HardwareProfile, c: Candidate) -> Verdict:
    v = _assess(hw, c)
    if v.status in (RECOMMENDED, SLOW):
        return Verdict(v.name, v.status, v.reasons, _speed(hw, c, v.status))
    return v


def _assess(hw: HardwareProfile, c: Candidate) -> Verdict:
    vram_mb = _gpu_vram_mb(hw)
    if vram_mb is not None and c.vram_mb is not None:
        vram_mb = max(vram_mb, c.vram_mb)
    if c.file_format.lower() not in SUPPORTED_FORMATS:
        return Verdict(c.name, UNSUPPORTED, (msg("compat.reason.format", format=c.file_format),))
    if not c.architecture_supported:
        return Verdict(c.name, UNSUPPORTED, (msg("compat.reason.architecture"),))
    if vram_mb is None:
        return Verdict(c.name, UNSUPPORTED, (msg("compat.reason.no_nvidia"),))
    if not c.downloaded and hw.disk_free_mb is not None:
        need_mb = (c.file_bytes + DISK_RESERVE) // MIB
        if need_mb > hw.disk_free_mb:
            reason = msg("compat.reason.disk", need_mb=need_mb, free_mb=hw.disk_free_mb)
            return Verdict(c.name, NO_DISK, (reason,))
    need = c.estimate.required
    owned_mb = need.ram // MIB
    ram_need_mb = (need.ram + need.mapped) // MIB
    if hw.ram_total_mb is not None and owned_mb > hw.ram_total_mb:
        reason = msg("compat.reason.ram", need_mb=owned_mb, total_mb=hw.ram_total_mb)
        return Verdict(c.name, NO_RAM, (reason,))
    if need.vram // MIB > vram_mb:
        reason = msg("compat.reason.vram", need_mb=need.vram // MIB, total_mb=vram_mb)
        return Verdict(c.name, NO_VRAM, (reason,))
    reasons = []
    paging = hw.ram_total_mb is not None and ram_need_mb > hw.ram_total_mb
    if paging:
        assert hw.ram_total_mb is not None
        reasons.append(msg("compat.reason.paging", mb=ram_need_mb - hw.ram_total_mb))
    elif hw.ram_available_mb is not None and ram_need_mb > hw.ram_available_mb:
        reasons.append(
            msg("compat.reason.ram_busy", free_mb=hw.ram_available_mb, need_mb=ram_need_mb)
        )
    weights = c.estimate.weights
    off_gpu = weights.ram + weights.mapped if c.off_gpu_bytes is None else c.off_gpu_bytes
    if off_gpu:
        reason = msg("compat.reason.on_cpu", mb=off_gpu // MIB)
        return Verdict(c.name, SLOW, (reason, *reasons))
    if paging:
        return Verdict(c.name, SLOW, tuple(reasons))
    return Verdict(c.name, RECOMMENDED, (msg("compat.reason.in_vram"), *reasons))


def filter_models(
    hw: HardwareProfile,
    candidates: Iterable[Candidate],
    statuses: Iterable[str] | None = None,
    max_file_bytes: int | None = None,
) -> list[Verdict]:
    """Фильтр по статусам и размеру; порядок — от лучшего статуса."""
    wanted = set(statuses) if statuses is not None else set(ORDER)
    result = [
        assess(hw, c)
        for c in candidates
        if max_file_bytes is None or c.file_bytes <= max_file_bytes
    ]
    return sorted((v for v in result if v.status in wanted), key=lambda v: ORDER.index(v.status))


RUNNABLE = (RECOMMENDED, SLOW)  # 4.6: выбор — recommend.py
