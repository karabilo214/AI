"""Длинный замер генерации: контекст 512, затем 256 слов жадно; 3 круга, полный словарь и угадыватель S=2048."""
import json
import statistics
import sys
import time

import numba
import numpy as np

import numba_engine2 as mod
from core import wiki_tokens

ids = wiki_tokens("test")[:600].numpy()
eng = mod.Engine(max_t=1024, predictor="results/vocab_r64.pt")
eng.step(int(ids[0]), 0)
res = {"threading": numba.threading_layer(), "threads": numba.get_num_threads()}
for S in (0, 2048) * 5:
    for p in range(512):
        eng.layers(int(ids[p]), p)
    tok, out = int(ids[512]), []
    t0 = time.perf_counter()
    for k in range(256):
        tok = eng.step(tok, 512 + k, S)
        out.append(tok)
    dt = (time.perf_counter() - t0) / 256 * 1000
    res.setdefault(str(S), []).append(dt)
    res.setdefault(f"tokens_{S}", out)
same = next((k for k in range(256) if res["tokens_0"][k] != res["tokens_2048"][k]), 256)
for S in ("0", "2048"):
    print(f"{res['threading']}, {res['threads']} потоков, словарь {'полный' if S == '0' else 'S=2048'}: "
          f"{statistics.median(res[S]):.1f} мс на слово (круги {[round(x, 1) for x in res[S]]})")
print(f"генерация 256 слов: полный и угадыватель совпали на первых {same}")
json.dump({k: v for k, v in res.items() if not k.startswith("tokens")}, open("results/bench_numba.json", "w"), indent=1)
