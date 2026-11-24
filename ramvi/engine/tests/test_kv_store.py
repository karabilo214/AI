"""B.7: кэш k, v неактивных чатов — сохранить перед перезаписью, вернуть при возврате.

Поддельный движок с буферами того же вида, что у настоящего: видеокарта (torch ``Kc``/``Vc``),
слой на процессоре (numpy ``W[l].K``/``.V``) и временная копия ``Ks`` (не хранится).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ramvi_engine.generation import kv_store
from ramvi_engine.generation.kv_store import KvStore, kv_buffers, switch

MAXT = 64
torch = pytest.importorskip("torch")  # лёгкие проверки CI идут без torch


class FakeEngine:
    def __init__(self, store: KvStore) -> None:
        self.maxt, self.kv_store = MAXT, store
        self.ctx: list[int] = []
        self.Kc = {i: torch.zeros(2, MAXT, 4, dtype=torch.float16) for i in range(2)}
        self.Vc = {i: torch.zeros(2, MAXT, 4, dtype=torch.float16) for i in range(2)}
        self.W = [
            {"weight": np.zeros((4, 4))},
            SimpleNamespace(
                K=np.zeros((2, MAXT, 4), np.float32), V=np.zeros((2, MAXT, 4), np.float32)
            ),
        ]
        self.Ks = torch.zeros(2, MAXT, 4)
        self.inv = torch.zeros(8)

    def process(self, prompt: list[int]) -> None:
        """Как движок: общее начало оставить, остальное «посчитать» (число из слова и позиции)."""
        start = kv_store.common(self.ctx, prompt)
        for pos in range(start, len(prompt)):
            value = prompt[pos] * 1000 + pos
            for buffer in kv_buffers(self):
                buffer[:, pos] = value % 2048
        self.ctx = list(prompt)

    def generate(self, prompt: list[int]) -> int:
        switch(self, prompt)
        recomputed = len(prompt) - kv_store.common(self.ctx, prompt)
        self.process(prompt)
        return recomputed


@pytest.fixture(autouse=True)
def small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kv_store, "MIN_TOKENS", 4)


def state(engine: FakeEngine, n: int) -> list[Any]:
    return [np.array(b[:, :n]) for b in kv_buffers(engine)]


def test_buffers_are_found_by_shape_without_scratch_copies() -> None:
    engine = FakeEngine(KvStore(2**20, 0, None))
    buffers = kv_buffers(engine)
    assert len(buffers) == 6  # Kc×2, Vc×2, W[1].K, W[1].V — без Ks и весов
    assert all(b is not engine.Ks for b in buffers)


def test_switching_chats_saves_and_restores_the_same_numbers() -> None:
    engine = FakeEngine(KvStore(2**20, 0, None))
    chat_a = list(range(1, 41))
    chat_b = [7] * 30
    assert engine.generate(chat_a) == 40
    before = state(engine, 40)
    assert engine.generate(chat_b) == 30  # чат B перезаписал кэш…
    assert engine.kv_store.saved == 1  # …но кэш A сохранён
    assert engine.generate([*chat_a, 99, 98]) == 2  # возврат: читаются только новые слова
    assert engine.kv_store.restored == 1
    assert all(np.array_equal(a, b) for a, b in zip(state(engine, 40), before, strict=True))
    # Сохранённый B тоже возвращается.
    assert engine.generate([*chat_b, 5]) == 1


def test_ram_overflow_goes_to_disk_and_disk_limit_evicts_oldest(tmp_path: Path) -> None:
    engine = FakeEngine(KvStore(1, 10**6, tmp_path))  # RAM меньше одной записи
    chats = [[c] * 20 for c in range(1, 5)]
    for chat in chats:
        engine.generate(chat)
    assert engine.kv_store.used("ram") == 0 and engine.kv_store.used("disk") > 0
    assert len(list(tmp_path.glob("*.npz"))) == 3
    assert engine.generate([*chats[0], 9]) == 1  # с диска
    size = next(iter(engine.kv_store.entries.values())).size
    (tmp_path / "t").mkdir()
    tight = FakeEngine(KvStore(0, 2 * size, tmp_path / "t"))
    for chat in chats:
        tight.generate(chat)
    kept = [e.tokens[0] for e in tight.kv_store.entries.values()]
    assert kept == [2, 3]  # самая давняя (1) вытеснена, текущая (4) ещё в движке


def test_rebuilt_engine_drops_saved_caches_and_disabled_store_does_nothing() -> None:
    engine = FakeEngine(KvStore(2**20, 0, None))
    engine.generate(list(range(1, 21)))
    engine.generate([3] * 20)
    assert len(engine.kv_store.entries) == 1
    engine.W.append(
        SimpleNamespace(K=np.zeros((2, MAXT, 4), np.float32), V=np.zeros((2, MAXT, 4), np.float32))
    )
    assert engine.generate(list(range(1, 25))) == 24  # слой перенесён — сохранённое не годится
    off = FakeEngine(KvStore(0, 0, None))
    off.generate(list(range(1, 21)))
    off.generate([3] * 20)
    assert not off.kv_store.entries and off.kv_store.saved == 0


def test_short_chats_are_not_copied() -> None:
    engine = FakeEngine(KvStore(2**20, 0, None))
    engine.generate([1, 2, 3])
    engine.generate([4, 5, 6])
    assert engine.kv_store.saved == 0


def test_disk_folder_is_owned_and_stale_folders_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAMVI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAMVI_KV_DISK_MB", "16")
    stale = tmp_path / "cache" / "kv" / "kv-1-dead"
    stale.mkdir(parents=True)
    (stale / "owner").write_bytes(b"")
    (stale / "1.npz").write_bytes(b"old")
    alive = KvStore.from_env()
    assert alive.folder is not None and alive.folder.is_dir()
    assert not stale.exists()  # владелец не занят — процесс завершился
    other = KvStore.from_env()  # второй движок не трогает папку живого
    assert alive.folder.is_dir() and other.folder is not None
    alive.close()
    other.close()
    assert not alive.folder.exists() and not other.folder.exists()
