"""Подбор BN/warp для пачечного ядра на сжатых весах (gemvc_tc) и одно-словного (gemvc1) на формах модели:
python sweep_tc.py <модель> [слоёв]. Время в графе на нескольких слоях подряд (мимо кэша)."""
import statistics
import sys

import torch

from big_engine import MATS, Ckpt
from wcomp import compress, gemvc1, gemvc_tc

name = sys.argv[1]
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ck = Ckpt(name)
C = []
for i in range(NL):
    w = ck.layer(i)
    C.append({n: compress(w[n].cuda()) for n in MATS})
    torch.cuda.empty_cache()


def bench_graph(f, n=10):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(2): g.replay()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        r.append(s.elapsed_time(e))
    return statistics.median(r)


torch.manual_seed(0)
cfg1, cfgt = {}, {}
s1 = st = 0
for n, mode in (("qkv", 0), ("o", 1), ("gu", 0), ("down", 2)):
    N, K = C[0][n]["N"], C[0][n]["K"]
    x = torch.randn(16, K * (2 if mode == 2 else 1), device="cuda")
    y = torch.zeros(16, N, device="cuda")
    r1 = sorted((bench_graph(lambda: [gemvc1(x[:1], c[n], y[:1], mode, bn, w) for c in C]), bn, w)
                for bn in (1, 2, 4, 8) for w in (1, 2, 4))
    rt = []
    for bn in (16, 32, 64):
        for w in (2, 4):
            for S in (1, 2, 4, 8):
                try:
                    rt.append((bench_graph(lambda: [gemvc_tc(x, c[n], y, mode, 16, bn, w, S) for c in C]), bn, w, S))
                except Exception as ex:
                    pass
    rt.sort()
    # проверка: пачка = по одному
    a = torch.zeros(16, N, device="cuda"); b = torch.zeros(16, N, device="cuda"); a2 = torch.zeros(16, N, device="cuda")
    gemvc_tc(x, C[0][n], a2, mode, 16, *rt[0][1:])
    gemvc_tc(x, C[0][n], a, mode, 16, *rt[0][1:])
    for t in range(16):
        gemvc1(x[t:t + 1], C[0][n], b[t:t + 1], mode, r1[0][1], r1[0][2])
    cfg1[(N, K, mode)] = r1[0][1:]; cfgt[(N, K, mode)] = rt[0][1:]
    s1 += r1[0][0]; st += rt[0][0]
    print(f"{n:5s} {N}x{K}: T=1 {r1[0][0]:.3f} мс {r1[0][1:]}, T=16 {rt[0][0]:.3f} мс {rt[0][1:]} "
          f"(в {rt[0][0] / r1[0][0]:.2f} раза); пачка против по одному {((a - b).abs().max() / b.abs().max()).item():.1e}, повтор бит в бит {torch.equal(a, a2)}", flush=True)
print(f"на {NL} слоёв: T=1 {s1:.3f} мс, T=16 {st:.3f} мс (в {st / s1:.2f} раза)")
print(f"CFG1 = {cfg1}\nCFGT = {cfgt}")
