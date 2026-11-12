"""Проверка и замер чтения текста пачкой.

1) пачка против пословного движка: конечное состояние последнего слова и кэш k, v;
2) генерация 64 слов после пачки совпадает с torch;
3) время чтения 512 слов: по одному, пачкой целиком, кусками по 128.
"""
import json
import time

import numba
import numpy as np
import torch

import numba_engine2 as mod
from core import L, Qwen, wiki_tokens

ids = wiki_tokens("test")[:1200].numpy()
eng = mod.Engine(max_t=1100)
T = 512
prompt = ids[:T]

# пословно
eng.step(int(prompt[0]), 0)
t0 = time.perf_counter()
for p, t in enumerate(prompt):
    eng.layers(int(t), p)
t_seq = time.perf_counter() - t0
h_seq, K_seq, V_seq = eng.hn.copy(), eng.K[:, :, :T].copy(), eng.V[:, :, :T].copy()

# пачкой (прогрев и компиляция на коротком тексте)
eng.prefill(prompt[:16], 0)
eng.K[:] = 0; eng.V[:] = 0
t0 = time.perf_counter()
h_b = eng.prefill(prompt, 0).copy()
t_b = time.perf_counter() - t0
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
print(f"1) пачка против пословного: состояние {rel(h_b, h_seq):.1e}, кэш k {rel(eng.K[:, :, :T], K_seq):.1e}, v {rel(eng.V[:, :, :T], V_seq):.1e}")

# кусками по 128
times_chunk = {}
for ch in (64, 128, 256):
    t0 = time.perf_counter()
    for a in range(0, T, ch):
        h_c = eng.prefill(prompt[a:a + ch], a)
    times_chunk[ch] = time.perf_counter() - t0
    print(f"   кусками по {ch}: состояние {rel(h_c, h_seq):.1e}")

# 2) генерация после пачки против torch
eng.prefill(prompt, 0)
tok, out = int(np.argmax(eng.w["emb"] @ eng.hn)), []
out.append(tok)
for k in range(63):
    tok = eng.step(tok, T + k)
    out.append(tok)
m = Qwen("cpu")
cache = [None] * L
lg = m.forward(torch.tensor(prompt)[None], cache=cache)
tt = int(lg[0, -1].argmax()); ref = [tt]
for _ in range(63):
    lg = m.forward(torch.tensor([[tt]]), cache=cache)
    tt = int(lg[0, -1].argmax()); ref.append(tt)
same = next((k for k in range(64) if out[k] != ref[k]), 64)
print(f"2) генерация 64 слов после пачки: совпало с torch первых {same}")

# 3) время, по 3 повтора
def best(f, n=3):
    r = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); r.append(time.perf_counter() - t0)
    return min(r)
t_b = best(lambda: eng.prefill(prompt, 0))
t_c = best(lambda: [eng.prefill(prompt[a:a + 128], a) for a in range(0, T, 128)])
t0 = time.perf_counter(); lg = m.forward(torch.tensor(prompt)[None]); t_torch = time.perf_counter() - t0
print(f"3) чтение {T} слов ({numba.threading_layer()}): по одному {t_seq * 1000:.0f} мс ({t_seq / T * 1000:.2f} мс/слово), "
      f"пачкой {t_b * 1000:.0f} мс ({t_b / T * 1000:.2f} мс/слово), кусками по 128 {t_c * 1000:.0f} мс, torch CPU {t_torch * 1000:.0f} мс")
json.dump({"seq_s": t_seq, "batch_s": t_b, "chunk128_s": t_c, "torch_cpu_s": t_torch, "gen_same": same,
           "chunks": times_chunk}, open("results/check_prefill.json", "w"), indent=1)
