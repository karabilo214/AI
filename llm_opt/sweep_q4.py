"""Ядро q4 на формах модели: точность восстановления, подбор BN/warp в графе, время шага T=1."""
import statistics, sys
import torch
from big_engine import MATS, Ckpt
from q4 import dequant4, gemvq4, quant4

ck = Ckpt(sys.argv[1]); NL = 8
C, Wr = [], []
for i in range(NL):
    w = ck.layer(i)
    C.append({n: quant4(w[n].cuda()) for n in MATS})
    if i == 0: Wr = {n: w[n].cuda() for n in MATS}
def bench_graph(f, n=10):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    for _ in range(2): g.replay()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); r.append(s.elapsed_time(e))
    return statistics.median(r)
tot = 0; cfg = {}
for n, mode in (("qkv", 0), ("o", 1), ("gu", 0), ("down", 2)):
    N, K = C[0][n]["N"], C[0][n]["K"]
    W = Wr[n].float()
    err = ((dequant4(C[0][n]) - W).norm() / W.norm()).item()
    x = torch.randn(1, K * (2 if mode == 2 else 1), device="cuda"); y = torch.zeros(1, N, device="cuda")
    xin = x if mode != 2 else torch.nn.functional.silu(x[:, :K]) * x[:, K:]
    gemvq4(x, C[0][n], y, 0 if mode != 2 else 2, 1) if False else None
    yy = torch.zeros(1, N, device="cuda"); gemvq4(x, C[0][n], yy, mode, 1)
    ref = xin @ dequant4(C[0][n]).T
    r = sorted((bench_graph(lambda: [gemvq4(x, c[n], y, mode, 1, bn, w) for c in C]), bn, w) for bn in (1, 2, 4, 8, 16) for w in (1, 2, 4))
    cfg[(N, K, mode)] = r[0][1:]; tot += r[0][0]
    by = sum(C[0][n][k].numel() * C[0][n][k].element_size() for k in ("q", "s", "m")) * NL
    print(f"{n}: ошибка весов {err:.3f}, ядро против dequant {((yy - ref).abs().max() / ref.abs().max()).item():.1e}; "
          f"{r[0][0]:.3f} мс на {NL} слоёв ({by / r[0][0] / 1e6:.0f} ГБ/с) {r[0][1:]}", flush=True)
print(f"на {NL} слоёв {tot:.3f} мс; CFGQ = {cfg}")
