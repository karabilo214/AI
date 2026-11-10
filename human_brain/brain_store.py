"""Точки сохранения мозга между запусками: не переучивать прежние ступени заново.

save(b, rng, meta, path): мозг (Brain.save, до бита) в path.npz и рядом path.json
    с состоянием генератора дней (random.Random) и сведениями (ступень, дни, seed).
load(path): (мозг, генератор, сведения). Продолжение после загрузки идёт так же,
    как обучение подряд (проверка: check_store.py).
"""
import json
import os
import random

from Brain import Brain

HERE = os.path.dirname(os.path.abspath(__file__))
# HB_BRAINS: другая папка мозгов (проверка цепочки с нуля, не трогая сохранённые)
DIR = os.environ.get("HB_BRAINS") or os.path.join(HERE, "brains")


def path_for(name):
    os.makedirs(DIR, exist_ok=True)
    return os.path.join(DIR, name)


def save(b, rng, meta, path):
    b.save(path + ".npz")
    v, state, gauss = rng.getstate()
    with open(path + ".json", "w") as f:
        json.dump({"rng": [v, list(state), gauss], "meta": meta}, f, ensure_ascii=False)


def exists(path):
    return os.path.exists(path + ".npz") and os.path.exists(path + ".json")


def load(path):
    b = Brain.load(path + ".npz")
    with open(path + ".json") as f:
        d = json.load(f)
    rng = random.Random()
    v, state, gauss = d["rng"]
    rng.setstate((v, tuple(state), gauss))
    return b, rng, d["meta"]
