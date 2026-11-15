"""4.4 MOD-03: статус совместимости модели с этим компьютером и фильтры.

Неполное размещение в VRAM — не «неподходящая», а «Медленно»: движок
считает остальные слои на CPU. Учитывается одна NVIDIA (самая большая):
несколько карт движок пока не использует (пункт 4.10).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ramvi_engine.planning.resources import ResourceEstimate

from ramvi_service.hardware import HardwareProfile

MIB = 1024 * 1024
DISK_RESERVE = 512 * MIB
SUPPORTED_FORMATS = ("gguf", "safetensors")

RECOMMENDED = "Рекомендуется"
SLOW = "Медленно"
NO_VRAM = "Не хватает VRAM"
NO_RAM = "Не хватает RAM"
NO_DISK = "Не хватает диска"
UNSUPPORTED = "Не поддерживается"
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


# Только реальные замеры: (подстрока имени NVIDIA, model_id, статус) -> (мин, макс) ток/с.
MEASURED = {
    # ASUS RTX 5060 8 ГБ + 16 ГБ RAM, greedy, контекст 2048, 30 вопросов,
    # полные ряды этапов 1–2: 7.80–9.11 ток/с (docs/reports/STAGE2_ACCEPTANCE.md, CONTEXT.md).
    ("RTX 5060", "qwen3-14b", "Медленно"): (7.8, 9.1),
}


@dataclass(frozen=True)
class Verdict:
    name: str
    status: str
    reasons: tuple[str, ...]
    speed: str = "нет замера"


def _gpu_vram_mb(hw: HardwareProfile) -> int | None:
    sizes = [g.vram_total_mb for g in hw.gpus if g.vendor == "NVIDIA" and g.vram_total_mb]
    return max(sizes) if sizes else None


def _range(hw: HardwareProfile, c: Candidate, status: str) -> tuple[float, float] | None:
    if c.measured_tok_s is not None:
        return c.measured_tok_s
    gpu = " ".join(g.name for g in hw.gpus if g.vendor == "NVIDIA")
    return next((v for (g, m, st), v in MEASURED.items() if g in gpu and m == c.model_id and st == status), None)


def _speed(hw: HardwareProfile, c: Candidate, status: str) -> str:
    rng = _range(hw, c, status)
    return f"примерно {rng[0]:g}–{rng[1]:g} токенов/с" if rng else "нет замера"


def assess(hw: HardwareProfile, c: Candidate) -> Verdict:
    v = _assess(hw, c)
    if v.status in (RECOMMENDED, SLOW):
        return Verdict(v.name, v.status, v.reasons, _speed(hw, c, v.status))
    return v


def _assess(hw: HardwareProfile, c: Candidate) -> Verdict:
    vram_mb = _gpu_vram_mb(hw)
    if c.file_format.lower() not in SUPPORTED_FORMATS:
        return Verdict(c.name, UNSUPPORTED, (f"формат {c.file_format} не поддерживается",))
    if not c.architecture_supported:
        return Verdict(c.name, UNSUPPORTED, ("архитектура модели не поддерживается движком",))
    if vram_mb is None:
        return Verdict(c.name, UNSUPPORTED, ("нужна видеокарта NVIDIA (CUDA)",))
    if not c.downloaded and hw.disk_free_mb is not None:
        need_mb = (c.file_bytes + DISK_RESERVE) // MIB
        if need_mb > hw.disk_free_mb:
            return Verdict(c.name, NO_DISK, (f"нужно {need_mb} МБ на диске, свободно {hw.disk_free_mb}",))
    need = c.estimate.required
    ram_need_mb = (need.ram + need.mapped) // MIB
    if hw.ram_total_mb is not None and ram_need_mb > hw.ram_total_mb:
        return Verdict(c.name, NO_RAM, (f"нужно {ram_need_mb} МБ RAM, всего {hw.ram_total_mb}",))
    if need.vram // MIB > vram_mb:
        return Verdict(c.name, NO_VRAM, (f"план требует {need.vram // MIB} МБ VRAM, у карты {vram_mb}",))
    reasons = []
    if hw.ram_available_mb is not None and ram_need_mb > hw.ram_available_mb:
        reasons.append(f"сейчас свободно {hw.ram_available_mb} МБ RAM из нужных {ram_need_mb} — закройте программы")
    weights = c.estimate.weights
    if weights.ram or weights.mapped:
        on_cpu = (weights.ram + weights.mapped) // MIB
        return Verdict(c.name, SLOW, (f"{on_cpu} МБ весов не помещаются в VRAM и считаются на CPU", *reasons))
    return Verdict(c.name, RECOMMENDED, ("модель целиком в видеопамяти", *reasons))


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


# --- 4.6 AUTO-04: выбор с объяснением ------------------------------------

FAST, LARGE = "быстрее", "крупнее"
RUNNABLE = (RECOMMENDED, SLOW)


@dataclass(frozen=True)
class Choice:
    best: Verdict | None
    reason: str
    alternatives: tuple[tuple[Verdict, str], ...]


def _params(c: Candidate) -> float:
    return c.params_b if c.params_b is not None else c.file_bytes / 1e9


def _speed_mid(hw: HardwareProfile, c: Candidate, status: str) -> float | None:
    rng = _range(hw, c, status)
    return (rng[0] + rng[1]) / 2 if rng else None


def choose(hw: HardwareProfile, candidates: list[Candidate], prefer: str = FAST, alternatives: int = 3) -> Choice:
    """Без общего рейтинга качества: порядок задаёт только выбор пользователя.

    FAST: модель целиком в VRAM раньше «Медленно», затем измеренная скорость,
    затем меньше параметров. LARGE: больше параметров среди запускаемых,
    при равенстве — лучший статус.
    """
    if prefer not in (FAST, LARGE):
        raise ValueError(f"prefer: {FAST!r} или {LARGE!r}")
    rated = [(c, assess(hw, c)) for c in candidates]
    ok = [(c, v) for c, v in rated if v.status in RUNNABLE]
    if not ok:
        why = "; ".join(f"{v.name}: {v.status}" for _, v in rated) or "список пуст"
        return Choice(None, f"ни одна модель не запустится на этом ПК ({why})", ())

    def key(item: tuple[Candidate, Verdict]) -> tuple:
        c, v = item
        status_rank = RUNNABLE.index(v.status)
        speed = _speed_mid(hw, c, v.status)
        if prefer == FAST:
            return (status_rank, -(speed or 0.0), _params(c))
        return (-_params(c), status_rank)

    ok.sort(key=key)

    def explain(c: Candidate, v: Verdict, first: Candidate | None) -> str:
        size = f"{_params(c):g}B" if c.params_b is not None else f"{c.file_bytes / 1e9:.1f} ГБ файл"
        parts = [v.status.lower(), size, v.speed]
        if first is not None:
            if prefer == FAST:
                parts.append("крупнее, но медленнее" if _params(c) > _params(first) else "медленнее")
            else:
                parts.append("быстрее, но меньше" if v.status == RECOMMENDED else "меньше")
        return ", ".join(parts)

    best_c, best_v = ok[0]
    lead = "самая быстрая из запускаемых" if prefer == FAST else "самая крупная из запускаемых"
    alts = tuple((v, explain(c, v, best_c)) for c, v in ok[1 : 1 + alternatives])
    return Choice(best_v, f"{lead}: {explain(best_c, best_v, None)}", alts)
