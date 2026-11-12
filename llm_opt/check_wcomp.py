"""Проверка сжатия весов: обратное восстановление бит в бит, результат gemvc против gemv_k, скорость в графе (28 слоёв)."""
import statistics

import torch

from fused_decode import FusedEngine, gemv
from wcomp import compress, decompress, gemvc

eng = FusedEngine(Ts=())
C = [{n: compress(Wn[n]) for n in Wn} for Wn in eng.Wn]
ok = all(torch.equal(decompress(c[n]), eng.Wn[l][n]) for l, c in enumerate(C) for n in c)
ne = sum(c[n]["ev"].numel() for c in C for n in c)
by = sum(c[n]["m8"].numel() + c[n]["e4"].numel() + c[n]["ev"].numel() * 12 + c[n]["rp"].numel() * 4 for c in C for n in c)
by0 = sum(w.numel() * 2 for Wn in eng.Wn for w in Wn.values())
print(f"восстановление бит в бит: {ok}; исключений {ne}; байт {by / 1e6:.0f} МБ против {by0 / 1e6:.0f} МБ")


def bench_graph(f, n=30):
    f(); f(); torch.cuda.synchronize()
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


torch.manual_seed(0)
tot0 = tot1 = 0
for n, mode in (("qkv", 0), ("o", 1), ("gu", 0), ("down", 2)):
    N, K = eng.Wn[0][n].shape
    for T in (1, 2, 4):
        x = torch.randn(T, K * (2 if mode == 2 else 1), device="cuda")
        y0 = torch.randn(T, N, device="cuda")
        a, b = y0.clone(), y0.clone()
        gemv(x, eng.Wn[5][n], a, mode, T)
        gemvc(x, C[5][n], b, mode, T)
        err = ((a - b).abs().max() / a.abs().max()).item()
        y = y0.clone()
        t0 = bench_graph(lambda: [gemv(x, Wn[n], y, mode, T) for Wn in eng.Wn])
        t1 = bench_graph(lambda: [gemvc(x, c[n], y, mode, T) for c in C])
        if T == 1:
            tot0 += t0; tot1 += t1
        print(f"{n:5s} T={T}: разница {err:.1e}; 28 матриц: bf16 {t0:.3f} мс, сжатые {t1:.3f} мс", flush=True)
print(f"итого T=1: bf16 {tot0:.3f} мс, сжатые {tot1:.3f} мс")
