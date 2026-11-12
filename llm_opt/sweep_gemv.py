"""Подбор настроек gemv (T=1) по времени в CUDA-графе (28 разных матриц подряд) и предел чтения памяти."""
import statistics

import torch
import triton
import triton.language as tl

torch.manual_seed(0)


def bench_graph(f, n=20):
    f(); torch.cuda.synchronize()
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


# предел: чтение одного большого массива (сумма), 28 массивов по 8 МБ, копирование
big = torch.randn(600_000_000 // 4, device="cuda")
t = bench_graph(lambda: big.sum())
print(f"сумма 600 МБ: {600 / t:.0f} ГБ/с")
small = [torch.randn(8_000_000 // 4, device="cuda") for _ in range(28)]
out = torch.empty(28, device="cuda")
t = bench_graph(lambda: [torch.sum(s, 0, out=out[i]) for i, s in enumerate(small)])
print(f"сумма 28 x 8 МБ: {28 * 8 / t:.0f} ГБ/с")
del big, small


@triton.jit
def gemv_t(X, W, Y, N, K, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rk)
        w = tl.load(W + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rn, acc)


shapes = {"qkv": (4096, 1024), "o": (1024, 2048), "gu": (6144, 1024), "down": (1024, 3072), "словарь": (151936, 1024)}
for name, (N, K) in shapes.items():
    reps = 1 if N > 100000 else 28
    Ws = [torch.randn(N, K, device="cuda").bfloat16() for _ in range(reps)]
    x = torch.randn(K, device="cuda")
    y = torch.zeros(N, device="cuda")
    res = []
    for bn in (1, 2, 4, 8, 16, 32):
        for bk in (128, 256, 512, 1024):
            if bk > K or N % bn: continue
            for w in (1, 2, 4, 8):
                for st in (1, 2, 3, 4):
                    if bn * bk / (32 * w) > 128: continue  # слишком много регистров на поток
                    try:
                        t = bench_graph(lambda: [gemv_t[(N // bn,)](x, W, y, N, K, BN=bn, BK=bk, num_warps=w, num_stages=st) for W in Ws], n=10)
                    except Exception:
                        continue
                    res.append((t, bn, bk, w, st))
    res.sort()
    gb = reps * N * K * 2 / 1e6
    print(f"{name:8s}: " + "; ".join(f"{gb / t:.0f} ГБ/с BN={bn} BK={bk} w={w} st={st}" for t, bn, bk, w, st in res[:3]), flush=True)
    del Ws
    torch.cuda.empty_cache()
