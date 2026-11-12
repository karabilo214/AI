"""Шаг 2г. Пустые моменты: пары (голова, слово) и (часть слоя, слово), которые почти ничего не делают.

Решение «пусто» принимается по уже посчитанному (оракул), т.е. это предел экономии.
Экономия в умножениях считается так, будто решение бесплатно:
  голова на слове: q-срез 1024x128 + o-срез 128x1024 + внимание 2*128*256  = 327 680
  внимание слоя на слове: q + o + внимание (k, v всё равно нужны для следующих слов) = 5 242 880
  MLP слоя на слове: 3*1024*3072 = 9 437 184
Всего на слово 625 млн (шаг 1, контекст в среднем 256).
"""
import json
import time

import torch

from core import H, HD, L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
TOTAL = 625e6
COST = {"head": 1024 * 128 * 2 + 2 * 128 * 256, "attn": 1024 * 2048 * 2 + 2 * 16 * 128 * 256, "mlp": 3 * 1024 * 3072}

t0 = time.time()
m = Qwen(DEV)
test = windows(wiki_tokens("test").to(DEV), 8)
Wo = [w["o"].view(H, HD, 1024) for w in m.layers]
NTOK = test.numel()


def evaluate(opts=None):
    ls, ac = [], []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        lo, a = loss_of(m.forward(x, opts), x)
        ls.append(lo); ac.append(a)
    return sum(ls) / len(ls), sum(ac) / len(ac)


base = evaluate()
print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}")
res = {"base": base, "heads": {}, "layers": {}, "combo": {}}

# ---------- 1. распределения: доля внимания на первое слово и относительный вклад головы ----------
sink_hist = torch.zeros(L, 5)  # доля пар с долей на 1-е слово >0.5, >0.8, >0.9, >0.95, >0.99
rel_hist = torch.zeros(L, 4)   # доля пар с вкладом <0.01, <0.02, <0.05, <0.1 от |x|
lay_rel = {"attn": [[] for _ in range(L)], "mlp": [[] for _ in range(L)]}
cur_x = {}


def heads_rec(l, a, out, v):
    s = a[0, :, :, 0].float()  # [H, T]
    for j, t in enumerate((0.5, 0.8, 0.9, 0.95, 0.99)):
        sink_hist[l, j] += (s > t).float().sum().cpu()
    c = torch.einsum("htd,hdo->hto", out[0], Wo[l]).norm(dim=-1)  # [H, T]
    r = c / cur_x[l]
    for j, t in enumerate((0.01, 0.02, 0.05, 0.1)):
        rel_hist[l, j] += (r < t).float().sum().cpu()
    return out


def delta_rec(kind, l, x, out):
    r = (out[0].norm(dim=-1) / x[0].norm(dim=-1))
    lay_rel[kind][l].append(r.cpu())
    if kind == "mlp" and l + 1 < L:
        cur_x[l + 1] = (x + out)[0].norm(dim=-1)
    return out


for i in range(test.shape[0]):
    x = test[i:i + 1]
    cur_x[0] = m.emb[x][0].norm(dim=-1)
    m.forward(x, {"heads_fn": heads_rec, "delta_fn": delta_rec})
sink_hist /= NTOK * H
rel_hist /= NTOK * H
print("\nдоля пар (голова, слово) по слоям: внимание на 1-е слово >0.9 | >0.99 || вклад головы < 2% | < 5% от |x|")
for l in range(L):
    print(f"  слой {l:2d}: {sink_hist[l, 2]:.2f} {sink_hist[l, 4]:.2f} || {rel_hist[l, 1]:.2f} {rel_hist[l, 2]:.2f}")
print(f"  всего:   {sink_hist[:, 2].mean():.2f} {sink_hist[:, 4].mean():.2f} || {rel_hist[:, 1].mean():.2f} {rel_hist[:, 2].mean():.2f}")
print("\nотносительное изменение слова частью слоя (медиана |прибавка|/|x|), внимание / MLP:")
med = {k: [torch.cat(v).median().item() for v in lay_rel[k]] for k in lay_rel}
print("  attn:", " ".join(f"{x:.2f}" for x in med["attn"]))
print("  mlp: ", " ".join(f"{x:.2f}" for x in med["mlp"]))
res["sink_hist"], res["rel_hist"], res["layer_median"] = sink_hist.tolist(), rel_hist.tolist(), med

# ---------- 2. убрать пустые головы на словах ----------
def heads_skip(crit, thr, repl, cnt):
    def fn(l, a, out, v):
        if crit == "sink":
            empty = a[0, :, :, 0] > thr  # [H, T]
        else:
            c = torch.einsum("htd,hdo->hto", out[0], Wo[l]).norm(dim=-1)
            empty = c / cur_x[l] < thr
        cnt[0] += empty.float().sum().item()
        e = empty[None, :, :, None].to(out.dtype)
        if repl == "zero":
            rep = torch.zeros_like(out)
        else:  # значение первого слова у своей головы (одно на весь текст)
            rep = v[:, :, 0:1, :].repeat_interleave(H // v.shape[1], dim=1).expand_as(out)
        return out * (1 - e) + rep * e
    return fn


def track_x(kind, l, x, out):
    if kind == "mlp" and l + 1 < L:
        cur_x[l + 1] = (x + out)[0].norm(dim=-1)
    return out


def run_heads(crit, thr, repl):
    cnt = [0]
    ls, ac = [], []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        cur_x[0] = m.emb[x][0].norm(dim=-1)
        lo, a = loss_of(m.forward(x, {"heads_fn": heads_skip(crit, thr, repl, cnt), "delta_fn": track_x}), x)
        ls.append(lo); ac.append(a)
    share = cnt[0] / (NTOK * L * H)
    saved = cnt[0] / NTOK * COST["head"] / TOTAL
    return sum(ls) / len(ls), sum(ac) / len(ac), share, saved


print("\nубрать пустые головы на словах:")
for crit, thrs, repls in [("sink", (0.9, 0.95, 0.99), ("zero", "v0")), ("contrib", (0.02, 0.05, 0.1), ("zero",))]:
    for thr in thrs:
        for repl in repls:
            r = run_heads(crit, thr, repl)
            res["heads"][f"{crit}_{thr}_{repl}"] = r
            print(f"  {crit:7s} {thr:<5} замена {repl:4s}: пустых пар {100 * r[2]:4.1f}%, потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), "
                  f"топ-1 {r[1]:.4f}, экономия {100 * r[3]:.1f}% умножений")

# ---------- 3. не считать часть слоя на словах, где она почти ничего не меняет ----------
def run_layers(kinds, thr, keep_first=0):
    cnt = {"attn": 0, "mlp": 0}

    def fn(kind, l, x, out):
        if kind in kinds and l >= keep_first:
            r = out[0].norm(dim=-1) / x[0].norm(dim=-1)
            empty = r < thr
            cnt[kind] += empty.float().sum().item()
            out = out * (~empty)[None, :, None].to(out.dtype)
        return out
    q = evaluate({"delta_fn": fn})
    saved = sum(cnt[k] * COST[k] for k in cnt) / NTOK / TOTAL
    share = {k: cnt[k] / (NTOK * L) for k in cnt}
    return q[0], q[1], share, saved


print("\nне считать часть слоя на словах, где |прибавка|/|x| < порога:")
for kinds in (("attn",), ("mlp",), ("attn", "mlp")):
    for thr in (0.05, 0.1, 0.2, 0.3):
        r = run_layers(kinds, thr)
        res["layers"][f"{'+'.join(kinds)}_{thr}"] = r
        print(f"  {'+'.join(kinds):8s} < {thr:<4}: пропущено attn {100 * r[2]['attn']:4.1f}% mlp {100 * r[2]['mlp']:4.1f}% пар, "
              f"потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), топ-1 {r[1]:.4f}, экономия {100 * r[3]:.1f}%")
json.dump(res, open("results/step2_idle.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
