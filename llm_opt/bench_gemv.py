"""Скорость gemv_k (T=1) на каждой форме весов: 28 разных матриц подряд (как слои, мимо кэша), ГБ/с весов.
Предел памяти RTX 5060 ~448 ГБ/с. Модуль с ядром — аргумент."""
import importlib
import statistics
import sys

import torch
import triton

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "fused_decode")
torch.manual_seed(0)
shapes = {"qkv": (4096, 1024, 0), "o": (1024, 2048, 1), "gu": (6144, 1024, 0), "down": (1024, 3072, 2),
          "словарь": (151936, 1024, 0)}


def bench(f, n=30):
    for _ in range(3): f()  # подбор настроек до записи графа
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(3): g.replay()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        r.append(s.elapsed_time(e))
    return statistics.median(r)


tot = 0
for name, (N, K, mode) in shapes.items():
    reps = 1 if N > 100000 else 28
    Ws = [torch.randn(N, K, device="cuda").bfloat16() for _ in range(reps)]
    x = torch.randn(1, K * (2 if mode == 2 else 1), device="cuda")
    y = torch.zeros(1, N, device="cuda")
    t = bench(lambda: [mod.gemv(x, w, y, mode, 1) for w in Ws])
    tot += t
    cfg = mod.gemv_k.best_config if hasattr(mod.gemv_k, "best_config") else ""
    print(f"{name:8s} {N}x{K}: {t:.3f} мс на {reps}, {reps * N * K * 2 / t / 1e6:.0f} ГБ/с  {cfg}", flush=True)
    del Ws
    torch.cuda.empty_cache()
print(f"всего {tot:.3f} мс (предел при 448 ГБ/с: {(28 * 15.7e6 + 155.6e6) * 2 / 448e6:.3f} мс)")
