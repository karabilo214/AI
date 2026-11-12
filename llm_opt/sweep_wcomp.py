"""Подбор BN и числа warp для gemvc1_k (одно слово, сжатые веса) по времени в графе на 28 настоящих матрицах."""
import statistics

import torch

from fused_decode import FusedEngine, gemv
from wcomp import compress, gemvc1

eng = FusedEngine(Ts=())
import sys
CH = int(sys.argv[1]) if len(sys.argv) > 1 else 256
C = [{n: compress(Wn[n], CH) for n in Wn} for Wn in eng.Wn]
print("кусок", CH)


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
tb = tc = 0
for n, mode in (("qkv", 0), ("o", 1), ("gu", 0), ("down", 2)):
    N, K = eng.Wn[0][n].shape
    x = torch.randn(1, K * (2 if mode == 2 else 1), device="cuda")
    y = torch.zeros(1, N, device="cuda")
    a = y.clone(); gemv(x, eng.Wn[3][n], a, mode, 1)
    b = y.clone(); gemvc1(x, C[3][n], b, mode)
    t0 = bench_graph(lambda: [gemv(x, Wn[n], y, mode, 1) for Wn in eng.Wn])
    res = []
    for BN in (1, 2, 4, 8):
        for w in (1, 2, 4):
            res.append((bench_graph(lambda: [gemvc1(x, c[n], y, mode, BN, w) for c in C]), BN, w))
    res.sort()
    best[(N, K, mode)] = res[0][1:]
    tb += t0; tc += res[0][0]
    print(f"{n:5s}: разница {((a - b).abs().max() / a.abs().max()).item():.1e}; bf16 {t0:.3f} мс; сжатые лучшие: "
          + ", ".join(f"{t:.3f} (BN={bn}, w={w})" for t, bn, w in res[:3]), flush=True)
print(f"итого: bf16 {tb:.3f} мс, сжатые {tc:.3f} мс; CFG1 = {best}")
