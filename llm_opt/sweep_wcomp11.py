"""Вариант ~11 бит: восстановление бит в бит, бит на вес, ответ против bf16, скорость в графе (28 слоёв) против 12 бит."""
import statistics

import torch

from fused_decode import FusedEngine, gemv
from wcomp import CFG1, bits11, compress, compress11, decompress11, gemvc1, gemvc11

eng = FusedEngine(Ts=())
C12 = [{n: compress(Wn[n]) for n in Wn} for Wn in eng.Wn]
C11 = [{n: compress11(Wn[n]) for n in Wn} for Wn in eng.Wn]
ok = all(torch.equal(decompress11(c[n]), eng.Wn[l][n]) for l, c in enumerate(C11) for n in c)
nw = sum(c[n]["N"] * c[n]["K"] for c in C11 for n in c)
b = sum(bits11(c[n]) * c[n]["N"] * c[n]["K"] for c in C11 for n in c) / nw
print(f"восстановление бит в бит: {ok}; бит на вес {b:.2f}; размер {b * nw / 8e6:.0f} МБ (bf16 {nw * 2 / 1e6:.0f}, 12 бит {nw * 1.5 / 1e6:.0f})")
CFG1.update({(4096, 1024, 0): (8, 1), (1024, 2048, 1): (2, 1), (6144, 1024, 0): (2, 1), (1024, 3072, 2): (2, 1)})


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


torch.manual_seed(0)
best = {}
t12 = t11 = 0
for n, mode in (("qkv", 0), ("o", 1), ("gu", 0), ("down", 2)):
    N, K = eng.Wn[0][n].shape
    x = torch.randn(1, K * (2 if mode == 2 else 1), device="cuda")
    y = torch.zeros(1, N, device="cuda")
    a = y.clone(); gemv(x, eng.Wn[3][n], a, mode, 1)
    b_ = y.clone(); gemvc11(x, C11[3][n], b_, mode)
    lnw = eng.m.layers[3]["ln1"]
    a2 = y.clone(); b2 = y.clone()
    if mode == 0:
        gemvc1(x, C12[3][n], a2, 0, lnw=lnw); gemvc11(x, C11[3][n], b2, 0, lnw=lnw)
    tb = bench_graph(lambda: [gemvc1(x, c[n], y, mode) for c in C12])
    res = []
    for BN in (1, 2, 4, 8, 16):
        for w in (1, 2, 4, 8):
            res.append((bench_graph(lambda: [gemvc11(x, c[n], y, mode, BN, w) for c in C11]), BN, w))
    res.sort()
    best[(N, K, mode)] = res[0][1:]
    t12 += tb; t11 += res[0][0]
    print(f"{n:5s}: разница с bf16 {((a - b_).abs().max() / a.abs().max()).item():.1e}, с нормировкой 11 против 12 бит "
          f"{((a2 - b2).abs().max() / (a2.abs().max() + 1e-30)).item():.1e}; 12 бит {tb:.3f} мс; 11 бит лучшие: "
          + ", ".join(f"{t:.3f} (BN={bn}, w={w})" for t, bn, w in res[:3]), flush=True)
print(f"итого 28 слоёв: 12 бит {t12:.3f} мс, 11 бит {t11:.3f} мс; CFG11 = {best}")
