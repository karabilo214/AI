"""Проверка ядер Triton: точность против fp32-умножения torch (на тех же bf16-весах) и скорость против torch.
Скорость: 28 разных матриц подряд (как слои), чтобы не сидеть в кэше видеокарты."""
import statistics

import torch

from triton_kernels import mm16

torch.manual_seed(0)
shapes = {"qkv 4096x1024": (4096, 1024), "o 1024x2048": (1024, 2048), "gate+up 6144x1024": (6144, 1024),
          "down 1024x3072": (1024, 3072), "словарь 151936x1024": (151936, 1024)}


def bench(f, n=20):
    for _ in range(3): f()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize()
        r.append(s.elapsed_time(e))
    return statistics.median(r)


for name, (N, K) in shapes.items():
    reps = 1 if N > 100000 else 28
    Ws = [torch.randn(N, K, device="cuda").bfloat16() for _ in range(reps)]  # [N, K]
    WsT32 = [w.float().T.contiguous() for w in Ws]  # для torch fp32 [K, N]
    WsT16 = [w.T.contiguous() for w in Ws]
    for T in (1, 8):
        x = torch.randn(T, K, device="cuda")
        ref = x.double() @ Ws[0].double().T
        y = mm16(x, Ws[0])
        y32 = x @ WsT32[0]
        err_t = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        err_32 = ((y32.double() - ref).abs().max() / ref.abs().max()).item()
        t_tr = bench(lambda: [mm16(x, w) for w in Ws])
        t_32 = bench(lambda: [x @ w for w in WsT32])
        xb = x.bfloat16()
        t_16 = bench(lambda: [xb @ w for w in WsT16])
        gbs = reps * N * K * 2 / (t_tr / 1000) / 1e9
        print(f"{name:20s} T={T}: ошибка triton {err_t:.1e} (torch fp32 {err_32:.1e}); время на {reps}: "
              f"triton {t_tr:.3f} мс ({gbs:.0f} ГБ/с весов), torch fp32 {t_32:.3f}, torch bf16 {t_16:.3f}", flush=True)
    del Ws, WsT32, WsT16
    torch.cuda.empty_cache()
