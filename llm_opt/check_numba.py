"""Проверка движка Numba против core.Qwen (torch) и замер времени генерации.

1) конечное состояние слова совпадает с torch (на 40 позициях подряд);
2) жадная генерация 64 слов совпадает с torch;
3) время одного слова при контексте 512: полный словарь и угадыватель S=1024/2048.
"""
import json
import statistics
import time

import numpy as np
import torch

from core import L, Qwen, rms, wiki_tokens
import importlib, sys
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "numba_engine")
Engine = mod.Engine
if len(sys.argv) > 2:
    import numba; numba.set_num_threads(int(sys.argv[2]))

ids = wiki_tokens("test")[:700].numpy()
eng = Engine(predictor="results/vocab_r64.pt")
t0 = time.time()
eng.step(int(ids[0]), 0)  # компиляция
print(f"компиляция и первый шаг: {time.time() - t0:.1f} с")

# 1) состояния по 40 позициям против torch
m = Qwen("cpu")
x = torch.tensor(ids[:40])[None]
_, hid = m.forward(x, return_hidden=True)
ref = rms(hid[-1][0], m.norm).numpy()
diffs = []
for p in range(40):
    h = eng.layers(int(ids[p]), p).copy()
    diffs.append(np.abs(h - ref[p]).max() / np.abs(ref[p]).max())
print(f"1) относительная разница конечных состояний с torch: макс {max(diffs):.2e}")

# 2) жадная генерация 64 слов
def gen_numba(prompt, n, S):
    for p, t in enumerate(prompt[:-1]):
        eng.layers(int(t), p)
    tok, out = int(prompt[-1]), []
    for k in range(n):
        tok = eng.step(tok, len(prompt) - 1 + k, S)
        out.append(tok)
    return out


def gen_torch(prompt, n):
    cache = [None] * L
    lg = m.forward(torch.tensor(prompt)[None], cache=cache)
    tok = int(lg[0, -1].argmax())
    out = [tok]
    for _ in range(n - 1):
        lg = m.forward(torch.tensor([[tok]]), cache=cache)
        tok = int(lg[0, -1].argmax()); out.append(tok)
    return out


prompt = ids[100:164]
a, b = gen_numba(prompt, 64, 0), gen_torch(prompt, 64)
same = next((k for k in range(64) if a[k] != b[k]), 64)
print(f"2) генерация 64 слов: совпало первых {same} из 64")

# 3) время
res = {"state_diff_max": float(max(diffs)), "gen_same": same}
for p, t in enumerate(ids[:512]):
    eng.layers(int(t), p)
times = {0: [], 1024: [], 2048: []}
for rnd in range(6):
    for S in times:
        t0 = time.perf_counter()
        tok = int(ids[512])
        for k in range(32):
            tok = eng.step(tok, 512 + k, S)  # пишем в одни и те же позиции 512..543
        times[S].append((time.perf_counter() - t0) / 32 * 1000)
# отдельно словарь и слои
h = eng.hn.copy()
argmax_rows = getattr(mod, "argmax_rows16", mod.argmax_rows); emb_head = eng.w.get("emb16", eng.w["emb"]); predictor_argmax = getattr(mod, "predictor_argmax16", mod.predictor_argmax)
for name, f in [("словарь полный", lambda: argmax_rows(emb_head, h)),
                ("словарь S=2048", lambda: predictor_argmax(emb_head, eng.At, eng.Bt, h, 2048)),
                ("28 слоёв", lambda: eng.layers(int(ids[512]), 512))]:
    f(); t0 = time.perf_counter()
    for _ in range(20): f()
    dt = (time.perf_counter() - t0) / 20 * 1000
    res[name] = dt
    print(f"   {name}: {dt:.2f} мс")
for S, v in times.items():
    name = "полный" if S == 0 else f"S={S}"
    res[f"token_ms_{name}"] = statistics.median(v)
    print(f"3) Numba, слово целиком ({name}): медиана {statistics.median(v):.1f} мс (разброс {min(v):.1f}-{max(v):.1f})")
json.dump(res, open(f"results/check_{mod.__name__}.json", "w"), indent=1)
