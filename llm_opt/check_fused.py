"""Проверка своих ядер: пачка из 8 слов даёт те же ответы, что 8 шагов по одному; и время T=1..16."""
import statistics
import time

import torch

from fused_decode import FusedEngine
from spec_gpu import prompts

eng = FusedEngine()
p = prompts[0]
out = eng.prefill(p)
seq = [int(out[-1])]
for k in range(8):
    seq.append(int(eng.run([seq[-1]], len(p) + k)[0]))
eng.prefill(p)
many = eng.run(seq[:8], len(p)).tolist()
print(f"пачка 8 слов против 8 шагов по одному: {sum(a == b for a, b in zip(many, seq[1:9]))}/8")
res = {T: [] for T in eng.graphs}
for _ in range(15):
    for T in eng.graphs:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        eng.run(seq[:1] * T, len(p))
        torch.cuda.synchronize(); res[T].append((time.perf_counter() - t0) * 1000)
for T in eng.graphs:
    print(f"  T={T:2d}: медиана {statistics.median(res[T]):.2f} мс, к одному слову {statistics.median(res[T]) / statistics.median(res[1]):.2f}")
