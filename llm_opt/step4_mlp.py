"""Шаг 4. Угадыватель нужных нейронов MLP (во всех 28 слоях сразу), без долгого обучения.

Нейрон i: m_i = silu(gate_i·h) * (up_i·h), вклад в выход |m_i| * |down_i|.
Способы выбора k% нейронов на слове:
  oracle   — по настоящему вкладу (предел, всё равно надо считать всё);
  gate     — gate считаем полностью (1/3 MLP), выбираем по |silu(g_i)| * s_i, где s_i = среднее |up_i·h| * |down_i| (train);
             up и down только для выбранных. Умножений: 1/3 + 2/3*k;
  lowrank  — грубая копия gate ранга r (подбор по состояниям train), выбор по |silu(ĝ_i)| * s_i,
             затем gate, up, down только для выбранных. Умножений: r*(1024+3072)/(3*1024*3072) + k.
Проверка на test.
"""
import json
import time

import torch
import torch.nn.functional as F

from core import FF, L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
t0 = time.time()
m = Qwen(DEV)
train = windows(wiki_tokens("train").to(DEV), 16)
test = windows(wiki_tokens("test").to(DEV), 8)
MLP = 3 * 1024 * FF

# статистика на train: ковариация входа MLP и средний |up·h|
cov = [torch.zeros(1024, 1024, device=DEV) for _ in range(L)]
upabs = [torch.zeros(FF, device=DEV) for _ in range(L)]
ntr = 0


def rec_in(g, l, x):
    if g == "in_mlp":
        x2 = x.reshape(-1, 1024)
        cov[l] += x2.T.contiguous() @ x2
        upabs[l] += (x2 @ m.layers[l]["up"]).abs().sum(0)
    return x


for i in range(train.shape[0]):
    m.forward(train[i:i + 1], {"in_mask": rec_in})
ntr = train.numel()
dn = [w["down"].norm(dim=1) for w in m.layers]  # down хранится как [FF, 1024]
s = [upabs[l] / ntr * dn[l] for l in range(L)]
print(f"статистика train: {time.time() - t0:.0f} с")


def lowrank(W, C, r):
    """W [in, out] ≈ A [in, r] @ B [r, out], лучшее для входов с ковариацией C."""
    C = C.cpu().double() / ntr
    ev, U = torch.linalg.eigh(C)
    ev = ev.clamp(min=ev.max().item() * 1e-9)
    Ch, Cih = (U * ev.sqrt()) @ U.T, (U / ev.sqrt()) @ U.T
    Wd = W.cpu().double()
    M = Ch @ Wd
    g, Ug = torch.linalg.eigh(M @ M.T)
    Ur = Ug.flip(1)[:, :r]
    return (Cih @ Ur).float().to(DEV).contiguous(), (Ur.T @ M).float().to(DEV).contiguous()


cur_h = {}


def evaluate(mlp_mask):
    def im(g, l, x):
        if g == "in_mlp":
            cur_h[l] = x
        return x
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
print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}")
res = {"base": base, "runs": []}


def report(name, frac, cost, r):
    res["runs"].append({"method": name, "frac": frac, "mlp_cost": cost, "loss": r[0], "acc": r[1]})
    print(f"  {name:12s} k={100 * frac:3.0f}%: потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), топ-1 {r[1]:.4f}, "
          f"умножений MLP {100 * cost:5.1f}%", flush=True)


for frac in (0.5, 0.3, 0.2):
    k = int(FF * frac)
    report("oracle", frac, frac, evaluate(lambda l, mm: keep_top(mm, mm.abs() * dn[l], k)))
    report("gate", frac, 1 / 3 + 2 / 3 * frac,
           evaluate(lambda l, mm: keep_top(mm, F.silu(cur_h[l] @ m.layers[l]["gate"]).abs() * s[l], k)))

for r in (64, 128, 256):
    AB = [lowrank(m.layers[l]["gate"], cov[l], r) for l in range(L)]
    pc = r * (1024 + FF) / MLP
    for frac in (0.5, 0.3):
        k = int(FF * frac)
        report(f"lowrank r={r}", frac, pc + frac,
               evaluate(lambda l, mm: keep_top(mm, F.silu((cur_h[l] @ AB[l][0]) @ AB[l][1]).abs() * s[l], k)))
json.dump(res, open("results/step4_mlp.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
