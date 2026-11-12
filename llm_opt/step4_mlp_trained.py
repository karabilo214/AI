"""Шаг 4б. Обученный угадыватель нейронов MLP: маленькая сеть h -> r (ReLU) -> 3072 на каждом слое.

Цель обучения: входит ли нейрон в k% самых сильных по настоящему вкладу |m_i|*|down_i| (на train).
Градиенты вручную (обратный проход torch на MPS падает). Проверка на test, выбор во всех слоях сразу.
Умножений MLP: r*(1024+3072)/(3*1024*3072) + k.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from core import FF, L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
t0 = time.time()
torch.manual_seed(0)
m = Qwen(DEV)
NW = 32
train = windows(wiki_tokens("train").to(DEV), NW)
test = windows(wiki_tokens("test").to(DEV), 8)
MLP = 3 * 1024 * FF
dn = [w["down"].norm(dim=1) for w in m.layers]
FR_T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # доля нейронов в цели обучения
RS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [128, 256]

# сбор входов MLP и настоящих вкладов (fp16), только чистый проход
Hs = [[] for _ in range(L)]
Cs = [[] for _ in range(L)]
cur = {}


def im(g, l, x):
    if g == "in_mlp":
        cur[l] = x
    return x


def mm_rec(l, mm):
    Hs[l].append(cur[l][0].half())
    Cs[l].append((mm[0].abs() * dn[l]).half())
    return mm


for i in range(NW):
    m.forward(train[i:i + 1], {"in_mask": im, "mlp_mask": mm_rec})
Hs = [torch.cat(x) for x in Hs]
Cs = [torch.cat(x) for x in Cs]
N = Hs[0].shape[0]
print(f"сбор: {N} слов x {L} слоёв, {time.time() - t0:.0f} с")


def train_layer(l, r, steps=800, bs=1024, lr=2e-3):
    kt = int(FF * FR_T)
    A = torch.randn(1024, r, device=DEV) / 32
    B = torch.zeros(r, FF, device=DEV)
    b0 = torch.zeros(FF, device=DEV)
    ps = [A, B, b0]
    mo = [torch.zeros_like(p) for p in ps]
    ve = [torch.zeros_like(p) for p in ps]
    for st in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=DEV)
        h = Hs[l][idx].float()
        c = Cs[l][idx].float()
        thr = c.kthvalue(FF - kt, dim=-1, keepdim=True).values
        y = (c > thr).float()
        z = h @ A
        a = z.clamp_min(0)
        s = a @ B + b0
        ds = (torch.sigmoid(s) - y) / bs
        gB = a.T.contiguous() @ ds
        gb = ds.sum(0)
        dz = (ds @ B.T.contiguous()) * (z > 0).float()
        gA = h.T.contiguous() @ dz
        for i, (p, g) in enumerate(zip(ps, (gA, gB, gb))):
            mo[i].mul_(0.9).add_(g, alpha=0.1)
            ve[i].mul_(0.999).addcmul_(g, g, value=0.001)
            p.sub_(lr * (mo[i] / (1 - 0.9 ** st)) / ((ve[i] / (1 - 0.999 ** st)).sqrt() + 1e-8))
    # доля совпадения выбранного с настоящим топом на последнем батче
    pred_top = s.topk(kt, dim=-1).indices
    hit = y.gather(1, pred_top).mean().item()
    return (A, B, b0), hit


def evaluate(mlp_mask):
    ls, ac = [], []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        lo, a = loss_of(m.forward(x, {"in_mask": im, "mlp_mask": mlp_mask}), x)
        ls.append(lo); ac.append(a)
    return sum(ls) / len(ls), sum(ac) / len(ac)


def keep_top(mm, score, k):
    idx = score.topk(k, dim=-1).indices
    return torch.zeros_like(mm).scatter_(-1, idx, mm.gather(-1, idx))


base = evaluate(None)
print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}; цель обучения: топ {100 * FR_T:.0f}%")
res = {"base": base, "target_frac": FR_T, "runs": []}
for r in RS:
    P, hits = [], []
    for l in range(L):
        p, h = train_layer(l, r)
        P.append(p); hits.append(h)
    print(f"r={r}: обучено за {time.time() - t0:.0f} с; угадано из настоящего топа (train, последний батч): "
          f"среднее {100 * sum(hits) / L:.0f}%, по слоям {[round(100 * x) for x in hits]}", flush=True)
    pc = r * (1024 + FF) / MLP
    for frac in (0.5, 0.4, 0.3):
        k = int(FF * frac)
        rr = evaluate(lambda l, mm: keep_top(mm, (cur[l] @ P[l][0]).clamp_min(0) @ P[l][1] + P[l][2], k))
        res["runs"].append({"r": r, "frac": frac, "mlp_cost": pc + frac, "loss": rr[0], "acc": rr[1], "hits": hits})
        print(f"  r={r} k={100 * frac:3.0f}%: потеря {rr[0]:.4f} (+{rr[0] - base[0]:.4f}), топ-1 {rr[1]:.4f}, "
              f"умножений MLP {100 * (pc + frac):5.1f}%", flush=True)
    torch.save([[t.cpu() for t in p] for p in P], f"results/mlp_pred_r{r}_t{FR_T}.pt")
json.dump(res, open(f"results/step4_mlp_trained_t{FR_T}.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
