"""Замер 1: сколько стоит проверить T слов одним проходом по сравнению с одним словом (контекст 512).
Сначала проверка: verify даёт те же ответы, что пословный расчёт."""
import json
import statistics
import time

import numpy as np

import numba_engine2 as mod
from core import wiki_tokens

ids = wiki_tokens("test")[:700].numpy()
eng = mod.Engine(max_t=1100)
eng.step(int(ids[0]), 0)
for p in range(512):
    eng.layers(int(ids[p]), p)
# проверка: пословно предсказания для позиций 512..519
seq = []
for k in range(8):
    seq.append(eng.step(int(ids[512 + k]), 512 + k))
got = list(eng.verify(ids[512:520], 512))
print(f"проверка verify против пословного на 8 словах: {sum(a == b for a, b in zip(seq, got))}/8 совпало")
Ts = (1, 2, 4, 8, 16)
for T in Ts:
    eng.verify(ids[512:512 + T], 512)  # компиляция/прогрев
res = {T: [] for T in Ts}
res["step"] = []
for rnd in range(15):
    t0 = time.perf_counter(); eng.step(int(ids[512]), 512); res["step"].append((time.perf_counter() - t0) * 1000)
    for T in Ts:
        t0 = time.perf_counter(); eng.verify(ids[512:512 + T], 512); res[T].append((time.perf_counter() - t0) * 1000)
st = statistics.median(res["step"])
ratio = {T: statistics.median([a / b for a, b in zip(res[T], res["step"])]) for T in Ts}
print(f"одно слово (обычный шаг): медиана {st:.1f} мс, лучший {min(res['step']):.1f}")
for T in Ts:
    md = statistics.median(res[T])
    print(f"проверка {T:2d} слов одним проходом: медиана {md:6.1f} мс (лучший {min(res[T]):6.1f}), "
          f"отношение к одному слову в том же круге: медиана {ratio[T]:.2f}")
res["ratio"] = {str(T): ratio[T] for T in Ts}
json.dump({str(k): v for k, v in res.items()}, open("results/bench_verify.json", "w"), indent=1)
