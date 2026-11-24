"""B.7 (ENG-KV-SESS-01): кэш k, v неактивных чатов.

Движок держит k, v одного запроса (``engine.ctx`` — слова, чьи строки уже в кэше). Запрос
другого чата перезаписывал их, и возврат в длинный чат читал всю историю заново. Теперь перед
перезаписью готовые строки кэша сохраняются в RAM (не помещаются — на диск, давние вытесняются),
а запрос, чьё начало совпадает с сохранённым дальше, чем с текущим, сначала получает его обратно.

Текст не меняется: восстанавливаются те же числа, что были посчитаны для тех же слов; дальше —
обычное продолжение с общего начала, как для текущего чата.

Буферы кэша находятся по форме: трёхмерный массив со второй осью длины ``engine.maxt``
(``[KV, maxt, HD]``) у видеокарты (``Kc``/``Vc``), у слоёв на процессоре и гибридных
(``W[l].K``/``.V``), у MoE (``cpul``/``attn``). Вспомогательные ``Ks``/``Vs`` — копия на время
прохода, их не храним.

Размеры — ``RAMVI_KV_RAM_MB`` (по умолчанию 2048: чат 14B на 4 тыс. слов ≈ 0,7 ГБ, остаётся RAM
под слои на процессоре у ноутбука на 16 ГБ) и ``RAMVI_KV_DISK_MB`` (по умолчанию 8192, папка
``cache/kv`` данных RAMVI); 0 — уровень выключен.
"""

from __future__ import annotations

import atexit
import collections
import itertools
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Меньше этого не сохраняем и не восстанавливаем: копия дороже, чем прочитать слова заново.
MIN_TOKENS = 256
SKIP = frozenset({"Ks", "Vs"})


def _is_buffer(value: Any, maxt: int) -> bool:
    shape = getattr(value, "shape", None)
    return (
        shape is not None
        and len(shape) == 3
        and shape[1] == maxt
        and (isinstance(value, np.ndarray) or type(value).__module__.startswith("torch"))
    )


def kv_buffers(engine: Any) -> list[Any]:
    """Все буферы k, v движка в постоянном порядке (без копий на время прохода)."""
    found: list[Any] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if _is_buffer(value, engine.maxt) and id(value) not in seen:
            seen.add(id(value))
            found.append(value)

    def scan(holder: Any) -> None:
        items = holder.values() if isinstance(holder, dict) else holder
        for item in items:
            add(item)
            if hasattr(item, "__dict__") and not isinstance(item, np.ndarray):
                for name, value in sorted(vars(item).items()):
                    if name not in SKIP:
                        add(value)

    for name, value in sorted(vars(engine).items()):
        if name in SKIP or name == "kv_store":
            continue
        if isinstance(value, dict):
            scan([value[k] for k in sorted(value, key=str)])
        elif isinstance(value, (list, tuple)):
            scan(value)
        else:
            add(value)
    return found


def signature(engine: Any) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Форма и тип каждого буфера: после переноса слоёв (нехватка памяти) сохранённое не годится."""
    return tuple((tuple(b.shape), str(b.dtype)) for b in kv_buffers(engine))


def snapshot(engine: Any, n: int) -> list[np.ndarray]:
    """Строки 0..n каждого буфера — копия в RAM (numpy, тот же тип чисел)."""
    rows = []
    for buffer in kv_buffers(engine):
        part = buffer[:, :n]
        if isinstance(part, np.ndarray):
            rows.append(part.copy())
        else:
            rows.append(part.detach().to("cpu", copy=True).contiguous().numpy())
    return rows


def restore(engine: Any, tokens: tuple[int, ...], rows: list[np.ndarray]) -> None:
    buffers = kv_buffers(engine)
    if len(buffers) != len(rows):
        raise ValueError("kv snapshot does not match engine buffers")
    n = len(tokens)
    for buffer, saved in zip(buffers, rows, strict=True):
        if isinstance(buffer, np.ndarray):
            buffer[:, :n] = saved
        else:
            import torch

            buffer[:, :n].copy_(torch.from_numpy(saved).to(buffer.device, buffer.dtype))
    engine.ctx = list(tokens)


@dataclass
class Entry:
    tokens: tuple[int, ...]
    size: int
    sig: tuple[Any, ...] = ()
    rows: list[np.ndarray] | None = None  # в RAM
    path: Path | None = None  # на диске
    serial: int = field(default=0)


def common(a: tuple[int, ...] | list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class KvStore:
    """Сохранённые кэши: RAM, затем диск; вытесняется давно не нужный."""

    def __init__(self, ram_bytes: int, disk_bytes: int, folder: Path | None) -> None:
        self.ram_bytes, self.disk_bytes = max(0, ram_bytes), max(0, disk_bytes)
        self.folder = folder
        self.entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()
        self._ids = itertools.count(1)
        self.saved = self.restored = 0
        self._owner = _hold(folder / OWNER) if folder is not None else None

    @classmethod
    def from_env(cls) -> KvStore:
        from ramvi_engine.config import cache_dir, env

        ram = int(float(env("RAMVI_KV_RAM_MB", "2048")) * 2**20)
        disk = int(float(env("RAMVI_KV_DISK_MB", "8192")) * 2**20)
        folder = None
        if disk:
            root = _ensure(cache_dir() / "kv")
            _remove_stale(root)
            folder = Path(tempfile.mkdtemp(prefix=f"kv-{os.getpid()}-", dir=root))
        store = cls(ram, disk, folder)
        atexit.register(store.close)  # обычный выход — файлы кэша удаляются сразу
        return store

    @property
    def enabled(self) -> bool:
        return bool(self.ram_bytes or self.disk_bytes)

    def used(self, where: str) -> int:
        return sum(
            e.size for e in self.entries.values() if (e.rows is not None) == (where == "ram")
        )

    def best(self, prompt: list[int], sig: tuple[Any, ...] = ()) -> tuple[Entry | None, int]:
        for key in [k for k, e in self.entries.items() if e.sig != sig]:
            self._drop(key)  # движок перестроил кэш — сохранённое не подходит
        best, length = None, 0
        for entry in self.entries.values():
            n = common(entry.tokens, prompt)
            if n > length:
                best, length = entry, n
        return best, length

    def put(self, tokens: list[int], rows: list[np.ndarray], sig: tuple[Any, ...] = ()) -> None:
        tokens_t = tuple(tokens)
        # Тот же чат короче — устарел.
        for key in [k for k, e in self.entries.items() if tokens_t[: len(e.tokens)] == e.tokens]:
            self._drop(key)
        size = sum(r.nbytes for r in rows)
        if size > max(self.ram_bytes, self.disk_bytes):
            return
        key = next(self._ids)
        self.entries[key] = Entry(tokens_t, size, sig=sig, rows=rows, serial=key)
        self.saved += 1
        self._fit()

    def take(self, entry: Entry) -> list[np.ndarray]:
        """Строки записи (с диска — в RAM); запись становится самой свежей."""
        key = entry.serial
        if entry.rows is None:
            assert entry.path is not None
            with np.load(entry.path) as data:
                entry.rows = [data[f"a{i}"] for i in range(len(data.files))]
            entry.path.unlink(missing_ok=True)
            entry.path = None
        self.entries.move_to_end(key)
        self.restored += 1
        rows = entry.rows
        self._fit()  # может отправить её же на диск — строки уже у нас
        return rows

    def _fit(self) -> None:
        # RAM переполнена — давние записи на диск (или вон, если диска нет).
        for key in list(self.entries):
            if self.used("ram") <= self.ram_bytes:
                break
            entry = self.entries[key]
            if entry.rows is None or key == next(reversed(self.entries)):
                continue
            if not self._to_disk(key):
                self._drop(key)
        # Самая свежая запись больше RAM — держим её на диске, если он есть.
        last = next(reversed(self.entries), None)
        if last is not None and self.used("ram") > self.ram_bytes:
            if self.entries[last].rows is not None and not self._to_disk(last):
                self._drop(last)
        for key in list(self.entries):
            if self.used("disk") <= self.disk_bytes:
                break
            if self.entries[key].rows is None:
                self._drop(key)

    def _to_disk(self, key: int) -> bool:
        entry = self.entries[key]
        if not self.disk_bytes or self.folder is None or entry.size > self.disk_bytes:
            return False
        assert entry.rows is not None
        path = self.folder / f"{key}.npz"
        np.savez(path, **{f"a{i}": r for i, r in enumerate(entry.rows)})
        entry.rows, entry.path = None, path
        return True

    def _drop(self, key: int) -> None:
        entry = self.entries.pop(key)
        if entry.path is not None:
            entry.path.unlink(missing_ok=True)

    def close(self) -> None:
        for key in list(self.entries):
            self._drop(key)
        if self._owner is not None:
            self._owner.close()
            self._owner = None
        if self.folder is not None:
            shutil.rmtree(self.folder, ignore_errors=True)


OWNER = "owner"


def _hold(path: Path) -> Any:
    """Открытый файл-владелец папки: пока процесс жив, файл занят (Windows — не удалить,
    POSIX — блокировка flock)."""
    handle = open(path, "a+b")  # noqa: SIM115 — держим открытым всё время работы
    if os.name != "nt":
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _remove_stale(root: Path) -> None:
    """Папки кэша процессов, которые завершились без уборки (сбой, принудительная остановка)."""
    for folder in root.glob("kv-*"):
        owner = folder / OWNER
        try:
            if os.name == "nt":
                owner.unlink(missing_ok=True)  # занят живым процессом — PermissionError
            else:
                import fcntl

                with open(owner, "a+b") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        shutil.rmtree(folder, ignore_errors=True)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def switch(engine: Any, prompt: list[int]) -> None:
    """Перед запросом: сохранить кэш, который запрос перезапишет; вернуть сохранённый, если
    его начало совпадает с запросом дальше текущего."""
    store = getattr(engine, "kv_store", None)
    if store is None:
        store = engine.kv_store = KvStore.from_env()
    if not store.enabled:
        return
    ctx = list(engine.ctx)
    sig = signature(engine)
    current = common(ctx, prompt)
    entry, length = store.best(prompt, sig)
    lose = len(ctx) - current
    if entry is not None and length >= current + MIN_TOKENS:
        if lose >= MIN_TOKENS:
            store.put(ctx, snapshot(engine, len(ctx)), sig)
        rows = store.take(entry)
        engine.ctx = []  # частичное восстановление не должно выглядеть готовым кэшем
        restore(engine, entry.tokens, rows)
        return
    if lose >= MIN_TOKENS:
        store.put(ctx, snapshot(engine, len(ctx)), sig)
