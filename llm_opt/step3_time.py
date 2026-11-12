"""Шаг 3в. Реальное время генерации одного слова (контекст 512): полный словарь против угадывателя r=64.
Варианты чередуются по кругу, 6 кругов, берётся медиана — чтобы шум и нагрев не путали сравнение."""
import json
import statistics
import sys
import time

import torch

from core import L, Qwen, wiki_tokens

dev = sys.argv[1]
m = Qwen(dev)
ab = torch.load("results/vocab_r64.pt")
A, B = ab["A"].to(dev).contiguous(), ab["B"].to(dev).contiguous()
sync = torch.mps.synchronize if dev == "mps" else (lambda: None)


def make_head(S):
    def head(h):
        h2 = h.reshape(-1, h.shape[-1])
        ap = (h2 @ A) @ B
        cand = ap.topk(S, dim=-1).indices
        ex = torch.bmm(m.emb[cand], h2[:, :, None])[..., 0]
        return cand.gather(1, ex.argmax(-1, keepdim=True))[:, 0].reshape(h.shape[:-1])
    return head


heads = {"полный": lambda h: (h @ m.head).argmax(-1), "S=1024": make_head(1024), "S=2048": make_head(2048)}
x = wiki_tokens("test")[:600].to(dev)[None]
cache0 = [None] * L
m.forward(x[:, :512], cache=cache0)
sync()
tok_t = {k: [] for k in heads}
head_t = {k: [] for k in heads}
hh = torch.randn(1, 1, 1024, device=dev)
for rnd in range(6):
    for name, hf in heads.items():
        cache = [None if c is None else (c[0].clone(), c[1].clone()) for c in cache0]
        for t in range(512, 516):  # прогрев
            m.forward(x[:, t:t + 1], {"head_fn": hf}, cache=[(c[0].clone(), c[1].clone()) for c in cache])
        sync(); t0 = time.perf_counter()
        for t in range(512, 544):
            m.forward(x[:, t:t + 1], {"head_fn": hf}, cache=cache)
        sync(); tok_t[name].append((time.perf_counter() - t0) / 32 * 1000)
        sync(); t0 = time.perf_counter()
        for _ in range(50):
            hf(hh)
        sync(); head_t[name].append((time.perf_counter() - t0) / 50 * 1000)
res = {}
for name in heads:
    a, b = statistics.median(tok_t[name]), statistics.median(head_t[name])
    res[name] = {"token_ms": a, "head_ms": b, "token_all": tok_t[name]}
    print(f"{dev} {name:7s}: слово целиком медиана {a:6.1f} мс (разброс {min(tok_t[name]):.1f}-{max(tok_t[name]):.1f}), "
          f"словарь отдельно {b:6.2f} мс")
json.dump(res, open(f"results/step3_time_{dev}.json", "w"), indent=1)
