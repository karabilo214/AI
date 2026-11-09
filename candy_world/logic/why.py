"""Почему одни поиски находят правило, а другие застревают. Разбор истории поисков (только анализ).

История: candy_world/living/results_history (лучшая таблица каждые 100 поколений, счёт, обычная цепочка).

    python why.py
"""
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

LIVING = Path(__file__).resolve().parent.parent / "living"
sys.path.insert(0, str(LIVING))
from engine import Design  # noqa: E402
from fast_engine import run_days  # noqa: E402
import task_count as T  # noqa: E402

DESIGN = Design()


def bits(text):
    return np.array([int(c) for c in text], dtype=np.uint8)


def exam(table, width=16, high=200, seed=99):
    rng = random.Random(seed)
    days, right, _ = T.make_days([(rng.randint(0, high), rng.randint(0, high)) for _ in range(300)], rng)
    got, _ = run_days(table, DESIGN, days, width, T.CODES, max_fetches=int(right.max()) * 2 + 10)
    need = right > 0
    return float((got[need] == right[need]).mean())


def small(table, high, width=8, seed=5):
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(high + 1) for b in range(high + 1)] * 2
    days, right, _ = T.make_days(pairs, rng)
    got, _ = run_days(table, DESIGN, days, width, T.CODES)
    return float((got == right).mean())


def used_rows():
    return [(c << 3) | (b << 2) | (a << 1) | bb for c in (1, 2, 3) for b in (0, 1) for a in (0, 1) for bb in (0, 1)]


def distance(x, y):
    """Различие в битах только по используемым строкам (24 строки по 3 бита)."""
    rows = used_rows()
    return int((x.reshape(32, 3)[rows] != y.reshape(32, 3)[rows]).sum())


runs = {}
for f in sorted(glob.glob(str(LIVING / "results_history" / "count__base__seed_*.json"))):
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    runs[d["seed"]] = d
winners = {s: bits(d["таблица"]) for s, d in runs.items() if all(v["когда надо идти"] == 1.0 for v in d["экзамен"].values())}
print("удачные поиски:", sorted(winners), "из", len(runs))

print("\n1. Когда появляется правило (экзамен до 200 на каждой точке истории):")
for s in sorted(winners):
    line = []
    for generation, stage, table in runs[s]["таблицы по ходу"]:
        line.append(f"{generation}:{stage}:{exam(bits(table)):.2f}")
    print(f"  поиск {s}: (поколение:ступень:экзамен) " + " ".join(line))

print("\n2. Застрявшие: на чём ошибаются и как далеко от удачных (бит из 72):")
for s, d in sorted(runs.items()):
    if s in winners:
        continue
    t = bits(d["таблица"])
    nearest = min(distance(t, w) for w in winners.values())
    print(f"  поиск {s}: ступень {d['ступень']}, точно до 3: {small(t, 3):.2f}, до 6: {small(t, 6):.2f}, "
          f"до 10: {small(t, 10):.2f}, экзамен до 200: {exam(t):.2f}, до ближайшей удачной: {nearest} бит")

print("\n3. Удачные между собой и с ручной:")
sys.path.insert(0, str(Path(__file__).parent))
import logic  # noqa: E402
hand = logic.hand_written_table()
for s, w in sorted(winners.items()):
    print(f"  поиск {s}: до ручной {distance(w, hand)} бит; " + ", ".join(f"до поиска {o}: {distance(w, x)}" for o, x in winners.items() if o != s))

print("\n4. Как часто таблица меняется в удачных и застрявших (бит между соседними точками истории):")
for s, d in sorted(runs.items()):
    tables = [bits(t) for _, _, t in d["таблицы по ходу"]]
    changes = [distance(a, b) for a, b in zip(tables, tables[1:])]
    mark = "УДАЧНЫЙ" if s in winners else "застрял"
    print(f"  поиск {s} ({mark}): изменений по ходу {changes[:20]}{' ...' if len(changes) > 20 else ''}")
