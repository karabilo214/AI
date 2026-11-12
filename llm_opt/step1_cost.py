"""Шаг 1. Карта затрат: умножения по частям (расчёт) и время по частям (замер).

Два режима: чтение текста (512 токенов сразу) и генерация (1 токен при контексте 512).
"""
import json
import os
import time
from collections import defaultdict

import torch

from core import FF, HD, H, KV, L, D, Qwen, wiki_tokens

V = 151936


def macs_per_token(ctx):
    """Умножений-со-сложением на 1 токен при длине контекста ctx."""
    return {
        "attn_proj": L * D * (H * HD + 2 * KV * HD + H * HD),
        "attn_core": L * 2 * H * HD * ctx,
        "mlp": L * 3 * D * FF,
        "head": D * V,
    }


class Timer:
    def __init__(self, sync):
        self.sync, self.t, self.acc = sync, {}, defaultdict(float)

    def start(self, n):
        self.sync(); self.t[n] = time.perf_counter()

    def stop(self, n):
        self.sync(); self.acc[n] += time.perf_counter() - self.t[n]


def measure(device, reps):
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    m = Qwen(device)
    ids = wiki_tokens("test")[:600].to(device)[None]
    res = {}
    for mode in ["prefill512", "decode@512"]:
        def run(tm):
            if mode == "prefill512":
                m.forward(ids[:, :512], {"timer": tm} if tm else None)
            else:
                cache = [None] * L
                m.forward(ids[:, :512], cache=cache)
                sync()
                t0 = time.perf_counter()
                tmr = tm
                for t in range(512, 544):
                    m.forward(ids[:, t:t + 1], {"timer": tmr} if tmr else None, cache=cache)
                sync()
                return time.perf_counter() - t0
        run(None); sync()  # прогрев
        # общее время без таймера
        tot = []
        for _ in range(reps):
            sync(); t0 = time.perf_counter()
            r = run(None); sync()
            tot.append(r if mode != "prefill512" else time.perf_counter() - t0)
        tm = Timer(sync)
        for _ in range(reps):
            run(tm)
        ntok = 512 if mode == "prefill512" else 32
        parts = {k: v / reps for k, v in tm.acc.items()}
        res[mode] = {"total_s_no_timer": min(tot), "ms_per_token": 1000 * min(tot) / ntok,
                     "parts_s": parts, "parts_share": {k: v / sum(parts.values()) for k, v in parts.items()}}
    return res


if __name__ == "__main__":
    out = {"macs_prefill_avgctx256": macs_per_token(256), "macs_decode_ctx512": macs_per_token(512)}
    for k in ["macs_prefill_avgctx256", "macs_decode_ctx512"]:
        tot = sum(out[k].values())
        print(f"\n{k}: всего {tot / 1e6:.0f} млн умножений на токен")
        for n, v in out[k].items():
            print(f"  {n:10s} {v / 1e6:7.1f} млн  {100 * v / tot:5.1f}%")
    for dev, reps in [("cpu", 3), ("mps", 5)]:
        r = measure(dev, reps)
        out[dev] = r
        for mode, d in r.items():
            print(f"\n{dev} {mode}: {d['ms_per_token']:.2f} мс/токен (без таймера)")
            for n, s in sorted(d["parts_share"].items(), key=lambda x: -x[1]):
                print(f"  {n:10s} {100 * s:5.1f}%")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/step1_cost.json", "w"), indent=1)
