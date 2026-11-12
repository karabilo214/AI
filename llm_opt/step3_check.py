"""Шаг 3б. Проверка угадывателя словаря (r=64): несовпадения, генерация текста, реальное время.

1) На test: когда выбор расходится с полной моделью, насколько близки два слова по вероятности полной модели.
2) Жадная генерация 100 слов от 8 начал текста: полный словарь против угадывателя, где расходятся.
3) Время генерации одного слова при контексте 512 на CPU и MPS: полный словарь против угадывателя.
"""
import json
import time

import torch

from core import L, Qwen, rms, tokenizer, wiki_tokens, windows

R = 64
res = {}


def make_head(m, A, B, S):
    def head(h):  # h [B, T, D] -> номера слов [B, T]
        sh = h.shape[:-1]
        h2 = h.reshape(-1, h.shape[-1])
        ap = (h2 @ A) @ B
        cand = ap.topk(S, dim=-1).indices
        ex = torch.bmm(m.emb[cand], h2[:, :, None])[..., 0]
        return cand.gather(1, ex.argmax(-1, keepdim=True))[:, 0].reshape(sh)
    return head


def full_head(m):
    return lambda h: (h @ m.head).argmax(-1)


# ---------- 1. несовпадения ----------
m = Qwen("mps")
ab = torch.load(f"results/vocab_r{R}.pt")
A, B = ab["A"].to("mps").contiguous(), ab["B"].to("mps").contiguous()
test = windows(wiki_tokens("test").to("mps"), 8)
print("1) несовпадения на test (4096 слов):")
for S in (1024, 2048):
    gaps, ranks, pfull, pmine = [], [], [], []
    for i in range(test.shape[0]):
        _, hid = m.forward(test[i:i + 1], return_hidden=True)
        h = rms(hid[-1][0], m.norm)
        p = (h @ m.head).softmax(-1)
        ft = p.argmax(-1)
        mt = make_head(m, A, B, S)(h)
        bad = (ft != mt).nonzero()[:, 0]
        for t in bad.tolist():
            a, b = p[t, ft[t]].item(), p[t, mt[t]].item()
            pfull.append(a); pmine.append(b)
            ranks.append(int((p[t] > p[t, mt[t]]).float().sum().item()) + 1)
    n = len(pfull)
    pf, pm = torch.tensor(pfull), torch.tensor(pmine)
    rk = torch.tensor(ranks, dtype=torch.float)
    print(f"  S={S}: несовпадений {n}; вероятность слова полной модели медиана {pf.median():.3f}, выбранного {pm.median():.3f}; "
          f"выбранное слово 2-е по полной модели в {100 * (rk == 2).float().mean():.0f}% случаев, "
          f"в топ-5 в {100 * (rk <= 5).float().mean():.0f}%; разница вероятностей < 0.05 в {100 * ((pf - pm) < 0.05).float().mean():.0f}%")
    res[f"mismatch_S{S}"] = {"n": n, "p_full_median": pf.median().item(), "p_mine_median": pm.median().item(),
                             "rank2": (rk == 2).float().mean().item(), "top5": (rk <= 5).float().mean().item()}


# ---------- 2. генерация ----------
def generate(m, prompt, n, head):
    cache = [None] * L
    tok = m.forward(prompt[None], {"head_fn": head}, cache=cache)[:, -1:]
    out = [tok.item()]
    for _ in range(n - 1):
        tok = m.forward(tok, {"head_fn": head}, cache=cache)[:, -1:]
        out.append(tok.item())
    return out


tk = tokenizer()
ids = wiki_tokens("test").to("mps")
print("\n2) жадная генерация 100 слов от 8 начал (по 64 слова текста):")
div = []
for S in (1024, 2048):
    first = []
    for j in range(8):
        prompt = ids[20000 + j * 3000: 20000 + j * 3000 + 64]
        g_full = generate(m, prompt, 100, full_head(m))
        g_mine = generate(m, prompt, 100, make_head(m, A, B, S))
        d = next((k for k in range(100) if g_full[k] != g_mine[k]), 100)
        first.append(d)
        if S == 2048 and j < 2:
            print(f"  начало: ...{tk.decode(prompt[-24:].tolist())!r}")
            print(f"    полный:      {tk.decode(g_full[:40])!r}")
            print(f"    угадыватель: {tk.decode(g_mine[:40])!r}")
    print(f"  S={S}: первое расхождение на слове {first} (100 = совпали все 100)")
    res[f"gen_first_div_S{S}"] = first

# ---------- 3. время ----------
print("\n3) время генерации одного слова, контекст 512:")
for dev in ("cpu", "mps"):
    if dev == "cpu":
        m = Qwen("cpu")
    Ad, Bd = A.to(dev).contiguous(), B.to(dev).contiguous()
    sync = torch.mps.synchronize if dev == "mps" else (lambda: None)
    x = wiki_tokens("test")[:600].to(dev)[None]
    heads = {"полный": full_head(m), "угад. S=1024": make_head(m, Ad, Bd, 1024), "угад. S=2048": make_head(m, Ad, Bd, 2048)}
    # только словарь
    h = torch.randn(1, 1, 1024, device=dev)
    for name, hf in heads.items():
        for _ in range(5): hf(h)
        sync(); t0 = time.perf_counter()
        for _ in range(50): hf(h)
        sync(); dt = (time.perf_counter() - t0) / 50 * 1000
        res[f"time_head_{dev}_{name}"] = dt
        print(f"  {dev} {name:13s}: словарь {dt:6.2f} мс", end="")
        # вся модель, 32 слова с кэшем, лучшее из 3
        best = 1e9
        for rep in range(3):
            cache = [None] * L
            m.forward(x[:, :512], {"head_fn": hf}, cache=cache); sync()
            t0 = time.perf_counter()
            for t in range(512, 544):
                m.forward(x[:, t:t + 1], {"head_fn": hf}, cache=cache)
            sync(); best = min(best, (time.perf_counter() - t0) / 32 * 1000)
        res[f"time_token_{dev}_{name}"] = best
        print(f", слово целиком {best:6.1f} мс")
json.dump(res, open("results/step3_check.json", "w"), indent=1)
