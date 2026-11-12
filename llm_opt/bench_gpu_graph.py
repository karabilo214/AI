"""Проверка и замер движка с графами на видеокарте.
1) fp32: жадная генерация 64 слов совпадает с core.Qwen (torch, без графов) — заведомо решаемая проверка;
2) время одного слова и проверки T слов (контекст 512), fp32 и bf16; медиана 15 кругов."""
import statistics
import sys
import time

import torch

from core import L, Qwen
from gpu_engine import GpuEngine

torch.manual_seed(0)
ids = torch.randint(1000, 50000, (600,)).tolist()
for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    eng = GpuEngine(dt)
    # 1) генерация после контекста 128
    out = eng.prefill(ids[:128])
    tok = int(out[-1]); gen = [tok]
    for k in range(63):
        tok = int(eng.run([tok], 128 + k)[0]); gen.append(tok)
    if name == "fp32":
        m = Qwen("cuda", torch.float32)
        cache = [None] * L
        lg = m.forward(torch.tensor(ids[:128], device="cuda")[None], cache=cache)
        t = int(lg[0, -1].argmax()); ref = [t]
        for _ in range(63):
            lg = m.forward(torch.tensor([[t]], device="cuda"), cache=cache)
            t = int(lg[0, -1].argmax()); ref.append(t)
        same = next((k for k in range(64) if gen[k] != ref[k]), 64)
        print(f"fp32: генерация 64 слов с графами совпала с обычным torch на первых {same}")
        del m
    # проверка: run(T слов) даёт те же ответы, что по одному
    eng.prefill(ids[:512])
    one = [int(eng.run([ids[512 + k]], 512 + k)[0]) for k in range(8)]
    eng.prefill(ids[:512])
    many = eng.run(ids[512:520], 512).tolist()
    print(f"{name}: проверка 8 слов одним проходом против по одному: {sum(a == b for a, b in zip(one, many))}/8")
    # 2) время
    eng.prefill(ids[:512])
    res = {T: [] for T in eng.graphs}
    for _ in range(15):
        for T in eng.graphs:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            eng.run(ids[512:512 + T], 512)
            torch.cuda.synchronize(); res[T].append((time.perf_counter() - t0) * 1000)
    for T in eng.graphs:
        rat = statistics.median([a / b for a, b in zip(res[T], res[1])])
        print(f"  {name} T={T:2d}: медиана {statistics.median(res[T]):6.2f} мс, лучший {min(res[T]):6.2f}, к одному слову {rat:.2f}")
    del eng
    torch.cuda.empty_cache()
