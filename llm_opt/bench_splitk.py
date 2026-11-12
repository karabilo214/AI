"""split-K против gemv_k на o и down: точность (против float64), повторяемость, скорость на 28 разных матрицах."""
import statistics

import torch

from fused_decode import gemv
from splitk import SplitK, gemv_sk_k

torch.manual_seed(0)
sk = SplitK()


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


for name, (N, K, mode) in {"o": (1024, 2048, 1), "down": (1024, 3072, 2)}.items():
    Ws = [torch.randn(N, K, device="cuda").bfloat16() for _ in range(28)]
    x = torch.randn(1, K * (2 if mode == 2 else 1), device="cuda")
    y0 = torch.randn(1, N, device="cuda")
    xin = x.double() if mode == 1 else (torch.nn.functional.silu(x[:, :K].double()) * x[:, K:].double())
    ref = y0.double() + xin @ Ws[0].double().T
    y1 = y0.clone(); gemv(x, Ws[0], y1, mode, 1)
    y2 = y0.clone(); sk(x, Ws[0], y2, mode)
    outs = []
    for _ in range(5):
        y3 = y0.clone(); sk(x, Ws[0], y3, mode); outs.append(y3)
    same = all(torch.equal(outs[0], o) for o in outs)
    e1 = ((y1.double() - ref).abs().max() / ref.abs().max()).item()
    e2 = ((y2.double() - ref).abs().max() / ref.abs().max()).item()
    y = y0.clone()
    t1 = bench(lambda: [gemv(x, w, y, mode, 1) for w in Ws])
    t2 = bench(lambda: [sk(x, w, y, mode) for w in Ws])
    gb = 28 * N * K * 2 / 1e6
    print(f"{name}: ошибка gemv_k {e1:.1e}, split-K {e2:.1e}, повторяемость {same}; время на 28: gemv_k {t1:.3f} мс "
          f"({gb / t1:.0f} ГБ/с), split-K {t2:.3f} мс ({gb / t2:.0f} ГБ/с)  {gemv_sk_k.best_config}", flush=True)
