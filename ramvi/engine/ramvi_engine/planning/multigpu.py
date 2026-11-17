"""4.10 ENG-MULTIGPU: раскладка слоёв модели по нескольким видеокартам NVIDIA.

Слои в видеопамяти идут подряд блоками: первые — на первой карте, следующие — на второй
и т. д.; между картами передаётся только состояние слоя (вектор `x`, T×D чисел), веса и
KV-кэш слоя лежат на его карте. Слои вне видеопамяти (по шине, на процессоре) считаются,
как и раньше, через первую карту.

Если все слои помещаются, они делятся пропорционально бюджетам карт (запас памяти на
каждой карте примерно одинаковый); если нет — карты заполняются по очереди, сколько
влезет. Модуль без torch: те же правила проверяются тестами без видеокарты.
"""

import os
from collections.abc import Callable, Sequence

MiB = 1024 * 1024
# Рабочие буферы движка и контекст CUDA на дополнительной карте (сверх весов и KV).
SECONDARY_OVERHEAD = 384 * MiB


def visible_devices(count: int, override: str | None = None) -> list[int]:
    """Какие карты использовать: `RAMVI_DEVICES="0,1"` (или пусто — все видимые)."""
    text = override if override is not None else os.environ.get("RAMVI_DEVICES", "")
    if not text.strip():
        return list(range(count))
    chosen = []
    for part in text.split(","):
        index = int(part.strip())
        if not 0 <= index < count:
            raise ValueError(f"RAMVI_DEVICES: карты {index} нет (видно {count})")
        if index not in chosen:
            chosen.append(index)
    return chosen


def usable_vram(free: Sequence[int], reserve: "Callable[[int], int]", gguf: bool = True) -> int:
    """Бюджет видеопамяти для плана: основная карта целиком (её резерв учитывает план),
    плюс дополнительные — за вычетом своего резерва и рабочих буферов. Несколько карт
    использует только GGUF-движок плотных Qwen3 (4.10); для остальных — только основная."""
    if not free:
        return 0
    if not gguf:
        return int(free[0])
    extra = sum(max(0, int(f) - reserve(int(f)) - SECONDARY_OVERHEAD) for f in free[1:])
    return int(free[0]) + extra


def split_layers(
    layer_bytes: Sequence[int], budgets: Sequence[int], resident: int | None = None
) -> list[int]:
    """Номер карты (позиция в `budgets`) для каждого из первых слоёв, лежащих в видеопамяти.

    `layer_bytes[i]` — веса слоя i вместе с его KV-кэшем; `budgets[d]` — сколько байт карте d
    можно отдать под слои. `resident=None` — столько слоёв, сколько помещается; иначе не
    больше `resident`. Возвращается не больше слоёв, чем помещается: длина ответа —
    фактическое число слоёв в видеопамяти.
    """
    if any(b < 0 for b in layer_bytes):
        raise ValueError("отрицательный размер слоя")
    limit = len(layer_bytes) if resident is None else min(resident, len(layer_bytes))
    wanted = list(layer_bytes[:limit])
    room = [max(0, int(b)) for b in budgets]
    if not wanted or not room:
        return []
    total = sum(room)
    need = sum(wanted)
    if need <= total and need > 0:
        scale = need / total
        proportional = _greedy(wanted, [r * scale for r in room], room)
        if proportional is not None and len(proportional) == len(wanted):
            return proportional
    return _greedy(wanted, room, room) or []


def _greedy(
    wanted: Sequence[int], targets: Sequence[float], room: Sequence[int]
) -> list[int] | None:
    """Карта d берёт слои по порядку, пока не превысит цель `targets[d]`; последняя занятая
    карта может взять сверх цели, но не сверх настоящего бюджета `room[d]`."""
    out: list[int] = []
    used = [0] * len(room)
    device = 0
    for size in wanted:
        while device < len(room):
            fits_target = used[device] + size <= targets[device] + 1
            last = device == len(room) - 1
            fits_room = used[device] + size <= room[device]
            if fits_room and (fits_target or last):
                used[device] += size
                out.append(device)
                break
            device += 1
        else:
            return out  # дальше не помещается
    return out
