"""Шаг 2в. Считать матрицу «по необходимости» по входу.

Вход x известен до умножения. Вклад входного числа i в выход: |x_i| * длина строки i матрицы W^T.
На каждом токене оставляем только самые сильные входы, строки остальных не читаем и не умножаем.
Угадывать ничего не нужно. Проверка на test; плюс замер реального времени на CPU.
"""
import json
import time

import torch

from core import L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
m = Qwen(DEV)
test = windows(wiki_tokens("test").to(DEV), 8)
GROUPS = {"in_qkv": ["q", "k", "v"], "in_o": ["o"], "in_mlp": ["gate", "up"], "mlp_h": ["down"]}
rn = {(g, l): torch.sqrt(sum(m.layers[l][n].pow(2).sum(1) for n in ns))
      for l in range(L) for g, ns in GROUPS.items()}
SIZE = {"in_qkv": 1024 * (2048 + 1024 + 1024), "in_o": 2048 * 1024, "in_mlp": 1024 * 3072 * 2, "mlp_h": 3072 * 1024}
FULL = L * sum(SIZE.values())


def evaluate(opts=None):
    ls, ac = [], []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        lo, a = loss_of(m.forward(x, opts), x)
        ls.append(lo); ac.append(a)
    return sum(ls) / len(ls), sum(ac) / len(ac)


def make_mask(frac, only=None):
    def im(g, l, x):
        if only is not None and g not in only:
            return x
        k = max(1, int(x.shape[-1] * frac[g] if isinstance(frac, dict) else x.shape[-1] * frac))
        c = x.abs() * rn[(g, l)]
        idx = c.topk(k, dim=-1).indices
        return torch.zeros_like(x).scatter_(-1, idx, x.gather(-1, idx))
    return im


base = evaluate()
print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}")
res = {"base": base, "all": {}, "single": {}}
for frac in (0.5, 0.4, 0.3, 0.2, 0.1):
    r = evaluate({"in_mask": make_mask(frac)})
    res["all"][frac] = r
    print(f"все матрицы слоёв, оставить {100 * frac:3.0f}% входов: потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), "
          f"топ-1 {r[1]:.4f}, умножений {100 * frac:.0f}%")
for g in GROUPS:
    for frac in (0.3, 0.1):
        r = evaluate({"in_mask": make_mask(frac, only={g})})
        res["single"][f"{g}_{frac}"] = r
        print(f"только вход {g:6s} ({'/'.join(GROUPS[g])}), {100 * frac:3.0f}%: потеря {r[0]:.4f} (+{r[0] - base[0]:.4f})")
json.dump(res, open("results/step2_input.json", "w"), indent=1)

# реальное время на CPU: одно слово, полная строка весов против выбранных строк
print("\nCPU, одно слово (генерация), время умножения вектора на матрицу:")
torch.manual_seed(0)
tim = {}
for name, (i, o) in {"up 1024x3072": (1024, 3072), "down 3072x1024": (3072, 1024)}.items():
    Ws = [torch.randn(i, o) for _ in range(28)]  # 28 разных, чтобы не сидеть в кэше процессора
    x = torch.randn(i)
    for frac in (1.0, 0.5, 0.3, 0.1):
        k = int(i * frac)
        idx = torch.randperm(i)[:k].sort().values
        xs = x[idx]
        f = (lambda: [x @ W for W in Ws]) if frac == 1.0 else (lambda: [xs @ W.index_select(0, idx) for W in Ws])
        for _ in range(3): f()
        t0 = time.perf_counter()
        for _ in range(10): f()
        dt = (time.perf_counter() - t0) / 10 / 28 * 1000
        tim[f"{name}_{frac}"] = dt
        print(f"  {name}, {100 * frac:3.0f}% строк: {dt:.3f} мс")
res["cpu_ms"] = tim
json.dump(res, open("results/step2_input.json", "w"), indent=1)
