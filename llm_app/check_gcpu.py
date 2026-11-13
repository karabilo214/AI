"""Матрицы GGUF на процессоре: точность против распакованной матрицы и время слоя. python -m llm_app.check_gcpu qwen3:14b"""
import sys
import time

import numpy as np

from .gcpu import CpuMat, sub_sums
from .gguf import GGUF, find_ollama


def main():
    g = GGUF(find_ollama(sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b"))
    rng = np.random.default_rng(0)
    names = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
    import os
    from .gcpu import _dll
    print("ядра:", "C++ (qmm.dll)" if _dll() is not None else "Numba")
    for T in (1, 8):
        tot, by = 0.0, 0
        for i in (0, 1):
            for n in names:
                m = CpuMat(g, f"blk.{i}.{n}.weight")
                X = rng.standard_normal((T, m.K)).astype(np.float32)
                SX = np.empty((T, m.K // 32), np.float32); sub_sums(X, SX)
                Y = np.zeros((T, m.N), np.float32)
                m.mm(X, Y, SX)
                if i == 0 and T == 8:
                    W = g.full(f"blk.{i}.{n}.weight").astype(np.float64)
                    ref = X.astype(np.float64) @ W.T
                    print(f"  {n} {m.t}: ошибка {np.abs(Y - ref).max() / np.abs(ref).max():.1e}")
                t0 = time.perf_counter()
                for _ in range(3):
                    m.mm(X, Y, SX)
                dt = (time.perf_counter() - t0) / 3
                if i == 1:
                    tot += dt
                    if T == 1:
                        print(f"    {n} {m.t} {m.N}x{m.K}: {dt * 1000:.2f} мс ({g.raw(f'blk.{i}.{n}.weight')[0].size / dt / 1e9:.1f} ГБ/с)")
                    by += g.raw(f"blk.{i}.{n}.weight")[0].size
        print(f"T={T}: слой {tot * 1000:.1f} мс ({by / tot / 1e9:.1f} ГБ/с весов)", flush=True)


if __name__ == "__main__":
    main()
