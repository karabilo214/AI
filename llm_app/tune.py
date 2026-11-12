"""Автонастройка ядер под видеокарту и формы матриц модели: время в CUDA-графе на нескольких копиях матрицы
(мимо кэша). Результат сохраняется в ~/.cache/llm_app/tune.json (ключ: видеокарта + форма) и при следующих
запусках берётся оттуда."""
import json
import os
import statistics

import torch

from . import wcomp
from .wcomp import compress, gemvc1, gemvc_tc

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "llm_app", "tune.json")


def _bench(f, n=8):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    g.replay(); torch.cuda.synchronize()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        r.append(s.elapsed_time(e))
    return statistics.median(r)


def _try(f):
    try:
        return _bench(f)
    except Exception:
        return float("inf")


def tune(shapes, log=print):
    """shapes: список (N, K, MODE). Заполняет wcomp.CFG1 и wcomp.CFGT."""
    gpu = torch.cuda.get_device_name()
    db = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [s for s in shapes if f"{gpu}|{s}" not in db]
    for N, K, mode in todo:
        log(f"  настройка ядер под {N}x{K} (вид {mode})...")
        cs = [compress((torch.randn(N, K, device="cuda") * 0.02).bfloat16()) for _ in range(3)]
        x = torch.randn(16, K * (2 if mode == 2 else 1), device="cuda")
        y = torch.zeros(16, N, device="cuda")
        r1 = min((_try(lambda: [gemvc1(x[:1], c, y[:1], mode, bn, w) for c in cs]), bn, w)
                 for bn in (1, 2, 4, 8, 16) for w in (1, 2, 4) if N % bn == 0)
        rt = min((_try(lambda: [gemvc_tc(x, c, y, mode, 16, bn, w, S) for c in cs]), bn, w, S)
                 for bn in (16, 32, 64) for w in (2, 4) for S in (1, 2, 4))
        db[f"{gpu}|{(N, K, mode)}"] = {"one": list(r1[1:]), "batch": list(rt[1:])}
        del cs
        torch.cuda.empty_cache()
    if todo:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(db, open(CACHE, "w"), indent=1)
    for N, K, mode in shapes:
        d = db[f"{gpu}|{(N, K, mode)}"]
        wcomp.CFG1[(N, K, mode)] = tuple(d["one"])
        wcomp.CFGT[(N, K, mode)] = tuple(d["batch"])
