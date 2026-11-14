"""Проверка ядер GGUF: перекладка + ядро против распакованной на процессоре матрицы; скорость в графе.
python -m ramvi.engine.tools check gguf -- qwen3:8b"""
import statistics
import sys

import numpy as np
import torch

from ramvi_engine.formats.gguf import GGUF, find_ollama
from ramvi_engine.kernels.gkernels import cfg_space, cfg_space_tc, ggemv, ggemv_tc, nbytes, repack


def bench(f, n=10):
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


def main():
    g = GGUF(find_ollama(sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"))
    torch.manual_seed(0)
    for name, mode in (("blk.0.attn_q.weight", 0), ("blk.0.attn_v.weight", 1), ("blk.0.ffn_down.weight", 1),
                       ("blk.2.ffn_up.weight", 0)):
        buf, N, K, t = g.raw(name)
        w = repack(buf, N, K, t)
        W = torch.from_numpy(g.full(name)).cuda().double()
        for T in (1, 5, 16):
            x = torch.randn(T, K, device="cuda")
            y0 = torch.randn(T, N, device="cuda")
            y = y0.clone() if mode else torch.zeros(T, N, device="cuda")
            (ggemv if T == 1 else ggemv_tc)(x, w, y, mode, T)
            ref = x.double() @ W.T + (y0.double() if mode else 0)
            err = ((y.double() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-4, (name, T, err)
            yb = torch.zeros(T, N, device="cuda")
            res = []
            for cf in (cfg_space(t) if T == 1 else cfg_space_tc(t)):
                try:
                    res.append((bench(lambda: (ggemv if T == 1 else ggemv_tc)(x, w, yb, 0, T, cf)), cf))
                except Exception:
                    pass
            res.sort()
            if not res:
                raise RuntimeError("нет рабочей конфигурации")
            tm, cf = res[0]
            selected = y0.clone() if mode else torch.zeros_like(y0)
            (ggemv if T == 1 else ggemv_tc)(x, w, selected, mode, T, cf)
            selected_err = ((selected.double() - ref).abs().max() / ref.abs().max()).item()
            assert selected_err < 1e-4, (name, T, cf, selected_err)
            print(f"{name} {t} {N}x{K} T={T}: ошибка {err:.1e}; {tm * 1000:.0f} мкс ({nbytes(w) / tm / 1e6:.0f} ГБ/с) {cf}", flush=True)


if __name__ == "__main__":
    main()
