"""Заголовок GGUF без весов: метаданные и таблица тензоров (4.4 MOD-03, 4.5 AUTO-03).

Нужен, чтобы оценить память модели **до скачивания**: заголовок — первые мегабайты
файла (запрос Range), веса не читаются. Объект повторяет интерфейс, который нужен
оценке движка (`planning.resources.estimate_gguf`, `planning.automatic`): `meta`,
`tensors`, `raw(name) -> (байты, N строк, K в строке, тип)`. Поэтому статус в каталоге
считается той же функцией, что и план при загрузке.

Только стандартная библиотека; повреждённые размеры отвергаются, как в движке.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any

# Тип ggml -> (имя, байт в блоке, значений в блоке). Неизвестный тип — оценка невозможна.
TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 4, 1),
    1: ("F16", 2, 1),
    2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32),
    7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32),
    9: ("Q8_1", 36, 32),
    10: ("Q2_K", 84, 256),
    11: ("Q3_K", 110, 256),
    12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256),
    14: ("Q6_K", 210, 256),
    15: ("Q8_K", 292, 256),
    30: ("BF16", 2, 1),
}
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
MAX_ITEMS = 10_000_000  # записей и элементов массива: защита от испорченного заголовка


class NotGGUF(ValueError):
    pass


class Truncated(Exception):
    """Данных меньше заголовка; `need` — сколько байт прочитать минимум."""

    def __init__(self, need: int) -> None:
        super().__init__(need)
        self.need = need


@dataclass(frozen=True)
class _Bytes:
    size: int  # как у memmap-среза движка: estimate_gguf берёт только .size


class GGUFMeta:
    """Метаданные (скаляры и строки; массивы — только тип и длина) и тензоры."""

    def __init__(self, meta: dict[str, Any], tensors: dict[str, tuple[tuple[int, ...], int]]):
        self.meta, self.tensors = meta, tensors

    def raw(self, name: str) -> tuple[_Bytes, int, int, str]:
        dims, kind = self.tensors[name]
        if kind not in TYPES:
            raise NotImplementedError(f"GGUF tensor type {kind}")
        type_name, block_bytes, block = TYPES[kind]
        k = dims[0]
        n = math.prod(dims[1:]) if len(dims) > 1 else 1
        return _Bytes(n * (k // block) * block_bytes), n, k, type_name

    def json(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "tensors": {name: [list(dims), kind] for name, (dims, kind) in self.tensors.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GGUFMeta":
        tensors = {name: (tuple(dims), int(kind)) for name, (dims, kind) in data["tensors"].items()}
        return cls(dict(data["meta"]), tensors)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0:
            raise NotGGUF("negative length")
        if end > len(self.data):
            raise Truncated(end)
        value = self.data[self.pos : end]
        self.pos = end
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def string(self) -> str:
        (n,) = self.unpack("Q")
        if n > 1 << 24:
            raise NotGGUF("string too long")
        return self.take(n).decode("utf-8", "replace")

    def value(self, kind: int, depth: int = 0) -> Any:
        if depth > 8:
            raise NotGGUF("nesting too deep")
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind])[0]
        if kind == 8:
            return self.string()
        if kind == 9:
            item, n = self.unpack("IQ")
            if n > MAX_ITEMS:
                raise NotGGUF("array too long")
            if item in _SCALARS:
                self.take(n * struct.calcsize("<" + _SCALARS[item]))
            else:
                for _ in range(n):  # словарь токенизатора: строки пропускаются по длине
                    self.value(item, depth + 1)
            return ["array", item, n]
        raise NotGGUF(f"metadata type {kind}")


def parse(data: bytes) -> GGUFMeta:
    """Разобрать заголовок из начала файла; мало данных — `Truncated(need)`."""
    reader = _Reader(data)
    if reader.take(4) != b"GGUF":
        raise NotGGUF("no GGUF signature")
    version, count_tensors, count_kv = reader.unpack("IQQ")
    if version not in (2, 3) or count_tensors > MAX_ITEMS or count_kv > MAX_ITEMS:
        raise NotGGUF("bad GGUF header")
    meta: dict[str, Any] = {}
    for _ in range(count_kv):
        key = reader.string()
        (kind,) = reader.unpack("I")
        meta[key] = reader.value(kind)
    tensors: dict[str, tuple[tuple[int, ...], int]] = {}
    for _ in range(count_tensors):
        name = reader.string()
        (ndims,) = reader.unpack("I")
        if not 1 <= ndims <= 4:
            raise NotGGUF("bad tensor rank")
        dims = reader.unpack(f"{ndims}Q")
        kind, _offset = reader.unpack("IQ")
        if any(d <= 0 for d in dims) or name in tensors:
            raise NotGGUF("bad tensor entry")
        if kind in TYPES and dims[0] % TYPES[kind][2]:
            raise NotGGUF("row is not a multiple of the quantization block")
        tensors[name] = (tuple(dims), kind)
    return GGUFMeta(meta, tensors)
