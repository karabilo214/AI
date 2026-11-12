"""Шаг 2б. Можно ли заменить большие матрицы двумя тонкими (низкий ранг).

Для каждой матрицы W (вход in, выход out) считаем, в какой узкой части пространства лежат
её реальные выходы Y = X W на обучающих текстах. Если на r направлений приходится почти вся энергия Y,
то W ≈ (W V_r) V_r^T: умножений in*r + r*out вместо in*out.
Направления подбираются на train, качество проверяется на test.
"""
import json
import time

import torch

from core import L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
MATS = {"q": "in_qkv", "k": "in_qkv", "v": "in_qkv", "o": "in_o", "gate": "in_mlp", "up": "in_mlp", "down": "mlp_h"}

t0 = time.time()
m = Qwen(DEV)
train = windows(wiki_tokens("train").to(DEV), 8)
test = windows(wiki_tokens("test").to(DEV), 8)

cov = {}  # (имя входа, слой) -> X^T X


def rec(name, l, x):
    if name in ("in_qkv", "in_o", "in_mlp", "mlp_h"):
        x = x.reshape(-1, x.shape[-1]).float()
        cov[(name, l)] = cov.get((name, l), 0) + x.T.contiguous() @ x


for i in range(train.shape[0]):
    m.forward(train[i:i + 1], {"rec": rec})

THR = (0.99, 0.999, 0.9999)
spec = {}  # (матрица, слой) -> (энергии по убыванию, собственные векторы)
table = []
for l in range(L):
    for n, src in MATS.items():
        W = m.layers[l][n].cpu().double()  # [in, out]
        Cx = cov[(src, l)].cpu().double()
        Cy = W.T @ Cx @ W
        e, V = torch.linalg.eigh(Cy)
        e, V = e.flip(0).clamp(min=0), V.flip(1)
        cum = e.cumsum(0) / e.sum()
        sw = torch.linalg.svdvals(W) ** 2
        cw = sw.cumsum(0) / sw.sum()
        row = {"layer": l, "mat": n, "in": W.shape[0], "out": W.shape[1],
               **{f"r_out_{t}": int((cum < t).sum()) + 1 for t in THR},
               "r_weight_0.99": int((cw < 0.99).sum()) + 1}
        table.append(row)
        spec[(n, l)] = V.float()
print(f"сбор и разложение: {time.time() - t0:.0f} с")

print("\nранг, нужный для 99% / 99.9% / 99.99% энергии выходов (среднее по слоям); для сравнения ранг самих весов на 99%")
for n in MATS:
    rs = [r for r in table if r["mat"] == n]
    a = lambda k: sum(r[k] for r in rs) / len(rs)
    print(f"  {n:5s} {rs[0]['in']}x{rs[0]['out']}: {a('r_out_0.99'):6.0f} {a('r_out_0.999'):6.0f} {a('r_out_0.9999'):6.0f}"
          f"   | веса: {a('r_weight_0.99'):6.0f}   (выгодно, если ранг < {rs[0]['in'] * rs[0]['out'] // (rs[0]['in'] + rs[0]['out'])})")

base = loss_of_all = None


def evaluate():
    ls, ac = [], []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        lo, a = loss_of(m.forward(x), x)
        ls.append(lo); ac.append(a)
    return sum(ls) / len(ls), sum(ac) / len(ac)


base = evaluate()
full_macs = sum(r["in"] * r["out"] for r in table)
print(f"\nисходная на test: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}; умножений в матрицах слоёв на токен {full_macs / 1e6:.0f} млн")
orig = {(n, l): m.layers[l][n].clone() for l in range(L) for n in MATS}
res = {"base": base, "table": table, "replace": {}}
for t in THR:
    macs = 0
    for r in table:
        n, l, rr = r["mat"], r["layer"], r[f"r_out_{t}"]
        if rr * (r["in"] + r["out"]) < r["in"] * r["out"]:
            Vr = spec[(n, l)][:, :rr].contiguous().to(DEV)
            m.layers[l][n] = (orig[(n, l)] @ Vr) @ Vr.T.contiguous()
            macs += rr * (r["in"] + r["out"])
        else:
            m.layers[l][n] = orig[(n, l)]
            macs += r["in"] * r["out"]
    q = evaluate()
    res["replace"][t] = {"loss": q, "macs": macs}
    print(f"все матрицы с порогом {t}: потеря {q[0]:.4f} (+{q[0] - base[0]:.4f}), топ-1 {q[1]:.4f}, "
          f"умножений {macs / 1e6:.0f} млн ({100 * macs / full_macs:.0f}% от исходных)")
for l in range(L):
    for n in MATS:
        m.layers[l][n] = orig[(n, l)]
json.dump(res, open("results/step2_rank.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
