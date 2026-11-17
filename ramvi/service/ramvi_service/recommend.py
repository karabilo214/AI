"""4.6 AUTO-04: выбор модели с объяснением и 2–3 альтернативами.

Сначала отсев: модель должна запускаться на этом компьютере («Рекомендуется» или
«Медленно») и подходить задаче профиля (4.5). Неизвестные (заголовок не прочитан)
не предлагаются. Одинаковые файлы из разных источников (один и тот же размер и
квант) считаются одной моделью: установленная важнее.

Потом порядок по приоритету профиля — только по проверяемым величинам:

- `balanced` (по умолчанию) — самая крупная среди комфортных (верхняя оценка
  скорости ≥ 15 ток/с — заметно быстрее чтения), если таких нет — среди
  интерактивных (≥ 3 ток/с). Без этого «скорость» выбирает самую маленькую модель,
  а «качество» — самую крупную на грани удобства;
- `speed` — выше скорость (замер, иначе оценка по плану);
- `quality` — больше параметров и точнее квант среди интерактивных (верхняя оценка
  скорости ≥ 3 ток/с). Это не рейтинг «умнее»: проверенных измерений качества у нас
  нет, и объяснение говорит об этом прямо;
- `ram` — меньше RAM (своя + веса из файла), затем VRAM;
- `ttft` — меньше весов вне видеопамяти (чтение запроса быстрее на GPU), затем меньше
  модель.

При равенстве — установленная, затем быстрее.
"""

from collections.abc import Callable
from typing import Any

from ramvi_service import compat
from ramvi_service.i18n import Msg, msg

MIN_INTERACTIVE = 3.0  # ток/с: медленнее — неудобно читать по мере генерации (выбор UX)
COMFORTABLE = 15.0  # ток/с: заметно быстрее чтения (выбор UX, не замер)
MIB = 1024 * 1024
UNKNOWN = "compat.unknown"
# Приблизительная точность кванта, бит на вес — только для сравнения при равном размере.
BITS = {
    "F32": 32,
    "BF16": 16,
    "F16": 16,
    "Q8_0": 8,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q4_K_M": 4.8,
    "Q4_K_S": 4.6,
    "Q4_0": 4.5,
}


def _speed(item: dict[str, Any]) -> float:
    tok = item["compat"].get("tok_s")
    return (tok["low"] + tok["high"]) / 2 if tok else 0.0


def _size(item: dict[str, Any]) -> float:
    return float(item["compat"]["model"].get("size_b") or 0.0)


def _bits(item: dict[str, Any]) -> float:
    return float(BITS.get(item["compat"]["model"].get("quant", ""), 0))


def _ram(item: dict[str, Any]) -> int:
    memory = item["compat"]["memory"]
    return int(memory["ram"] + memory["mapped"])


def _off_gpu(item: dict[str, Any]) -> int:
    return int((item["compat"].get("plan") or {}).get("off_gpu_bytes", 0))


def _interactive(item: dict[str, Any], threshold: float = MIN_INTERACTIVE) -> bool:
    tok = item["compat"].get("tok_s")
    return bool(tok) and tok["high"] >= threshold


ORDER: dict[str, Callable[[dict[str, Any]], tuple[float, ...]]] = {
    "balanced": lambda i: (
        not _interactive(i, COMFORTABLE),
        not _interactive(i),
        -_size(i),
        -_bits(i),
        -_speed(i),
    ),
    "speed": lambda i: (-_speed(i), _size(i)),
    "quality": lambda i: (not _interactive(i), -_size(i), -_bits(i), -_speed(i)),
    "ram": lambda i: (_ram(i), i["compat"]["memory"]["vram"], -_speed(i)),
    "ttft": lambda i: (_off_gpu(i), _size(i), -_speed(i)),
}


def _facts(item: dict[str, Any]) -> dict[str, Any]:
    c = item["compat"]
    return {
        "status": msg(c["status"] + ".inline"),
        "size": msg("recommend.size", size=f"{_size(item):g}", quant=c["model"].get("quant", "")),
        "speed": Msg(c["speed"]["code"], c["speed"]["params"]),
        "ram_mb": _ram(item) // MIB,
        "vram_mb": c["memory"]["vram"] // MIB,
        "off_gpu_mb": _off_gpu(item) // MIB,
    }


def _why_best(item: dict[str, Any], priority: str) -> Msg:
    facts = _facts(item)
    if priority in ("quality", "balanced") and not _interactive(item):
        return msg("recommend.reason.quality_slow", **facts)
    if priority == "balanced" and not _interactive(item, COMFORTABLE):
        return msg("recommend.reason.balanced_slow", **facts)
    if priority == "ttft" and not _off_gpu(item):
        return msg("recommend.reason.ttft_vram", **facts)
    return msg(f"recommend.reason.{priority}", **facts)


def _versus(item: dict[str, Any], best: dict[str, Any]) -> Msg:
    """Чем альтернатива отличается от выбранной — только проверяемые различия."""
    parts: list[Msg] = []
    if _size(item) != _size(best):
        parts.append(
            msg("recommend.vs.larger" if _size(item) > _size(best) else "recommend.vs.smaller")
        )
    if _speed(item) and _speed(best) and abs(_speed(item) - _speed(best)) > 0.1 * _speed(best):
        parts.append(
            msg("recommend.vs.faster" if _speed(item) > _speed(best) else "recommend.vs.slower")
        )
    if abs(_ram(item) - _ram(best)) > 256 * MIB:
        parts.append(
            msg("recommend.vs.more_ram" if _ram(item) > _ram(best) else "recommend.vs.less_ram")
        )
    if not parts:
        parts.append(msg("recommend.vs.similar"))
    return msg("recommend.alternative", **_facts(item), differences=parts)


def _brief(item: dict[str, Any]) -> dict[str, Any]:
    return {k: item[k] for k in ("id", "name", "installed")} | {"compat": item["compat"]}


def rank(items: list[dict[str, Any]], priority: str, alternatives: int = 3) -> dict[str, Any]:
    """`items`: {id, name, installed, compat}. Ответ: best, reason, alternatives, excluded."""
    if priority not in ORDER:
        raise ValueError(f"priority: {priority}")
    excluded = {"unknown": 0, "not_runnable": 0, "not_for_task": 0}
    usable: dict[str, dict[str, Any]] = {}
    for item in items:
        c = item["compat"]
        if c["status"] == UNKNOWN:
            excluded["unknown"] += 1
        elif c["status"] not in compat.RUNNABLE:
            excluded["not_runnable"] += 1
        elif not c["task"]["suits"]:
            excluded["not_for_task"] += 1
        else:
            key = c["model"]["key"] or item["id"]
            if key not in usable or (item["installed"] and not usable[key]["installed"]):
                usable[key] = item  # один файл из разных источников — одна модель
    if not usable:
        return {
            "priority": priority,
            "best": None,
            "reason": msg("recommend.none", **excluded).json(),
            "alternatives": [],
            "excluded": excluded,
        }
    ordered = sorted(
        usable.values(), key=lambda i: (*ORDER[priority](i), not i["installed"], -_speed(i))
    )
    best = ordered[0]
    return {
        "priority": priority,
        "best": _brief(best),
        "reason": _why_best(best, priority).json(),
        "alternatives": [
            {**_brief(i), "reason": _versus(i, best).json()} for i in ordered[1 : 1 + alternatives]
        ],
        "excluded": excluded,
    }
