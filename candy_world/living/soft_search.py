"""Способ 2: плавная таблица и спуск по ошибке (2026-11-11).

Каждый бит таблицы — вероятность p = сигмоида(z). Цепочка считает «вероятные»
биты: вход элемента — вероятности (c, n, a, b), выход = сумма по 16
сочетаниям входов (вероятность сочетания * p строки). Дни ступени «до 6»
(add_steps) пачкой, все 4 длины цепочки. Ошибка (двоичная перекрёстная
энтропия) на событиях с сигналом мира и на походах (первые R — «идти»,
(R+1)-й — «хватит»; походы ведутся по известному ответу). Спуск Adam;
со временем добавляется штраф за неопределённость битов (тянет к 0/1).
Каждые EVAL шагов таблица округляется (p > 0.5) и проверяется точно
быстрым движком; лучшая сохраняется. Удача — точно 1.0 на «до 6».

    python soft_search.py [--restarts 8] [--steps 2000]
"""
import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

import evolve as E
from fast_engine import run_days

EVAL = 50
import os
VERBOSE = bool(os.environ.get("SOFT_VERBOSE"))


def batch(t, d):
    days, right, target, mask = d
    FETCH = t.CODES["fetch"]
    seqs, tg, tm = [], [], []
    for i, ev in enumerate(days):
        R = int(right[i])
        codes = list(ev) + [FETCH] * (R + 1)
        targ = [int(target[i, j]) for j in range(len(ev))] + [1] * R + [0]
        msk = [bool(mask[i, j]) for j in range(len(ev))] + [True] * (R + 1)
        seqs.append(codes); tg.append(targ); tm.append(msk)
    L = max(map(len, seqs))
    C = np.zeros((len(seqs), L), dtype=np.int64)
    Y = np.zeros((len(seqs), L), dtype=np.float32)
    M = np.zeros((len(seqs), L), dtype=np.float32)
    A = np.zeros((len(seqs), L), dtype=np.float32)
    for i, (s, y, m) in enumerate(zip(seqs, tg, tm)):
        C[i, :len(s)] = s; Y[i, :len(s)] = y; M[i, :len(s)] = m; A[i, :len(s)] = 1
    return C, Y, M, A


def run(seed, steps):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t, design, widths, items, _ = E.setup("add_steps", "base")
    d = E.data(t, items, 6, 0)
    C, Y, M, A = (torch.tensor(x) for x in batch(t, d))
    B, L = C.shape
    z = torch.nn.Parameter(torch.randn(8, 16, 4) * 0.5)
    opt = torch.optim.Adam([z], lr=0.05)
    combos = torch.tensor([[(k >> (3 - j)) & 1 for j in range(4)] for k in range(16)], dtype=torch.float32)
    best, best_ex, t0 = None, -1.0, time.time()
    found = None
    for it in range(steps):
        P = torch.sigmoid(z)
        loss = 0.0
        for w in widths:
            c = torch.zeros(B, w); n = torch.zeros(B, w)
            for s in range(L):
                Pc = P[C[:, s]]                         # [B,16,4]
                a = torch.ones(B); b = torch.ones(B)
                nc, nn_ = [], []
                for i in range(w):
                    x = torch.stack([c[:, i], n[:, i], a, b], 1)            # [B,4]
                    wgt = torch.prod(combos[None] * x[:, None] + (1 - combos[None]) * (1 - x[:, None]), 2)  # [B,16]
                    o = torch.einsum("bc,bck->bk", wgt, Pc)
                    nc.append(o[:, 0]); nn_.append(o[:, 1]); a, b = o[:, 2], o[:, 3]
                act = A[:, s:s + 1]
                c = act * torch.stack(nc, 1) + (1 - act) * c
                n = act * torch.stack(nn_, 1) + (1 - act) * n
                p = b.clamp(1e-5, 1 - 1e-5)
                loss = loss + (M[:, s] * -(Y[:, s] * torch.log(p) + (1 - Y[:, s]) * torch.log(1 - p))).sum()
        loss = loss / (M.sum() * len(widths))
        lam = 0.1 * it / steps
        loss = loss + lam * (P * (1 - P)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % EVAL == 0 or it == steps - 1:
            tab = (torch.sigmoid(z) > 0.5).numpy().astype(np.uint8).reshape(-1)
            ex = E.fitness(tab, design, widths, t, d)[1]
            if VERBOSE and it % 200 == 0:
                Pn = torch.sigmoid(z).detach()
                print(f"шаг {it}: ошибка {loss.item():.4f}, неопределённых битов (0.1..0.9) "
                      f"{int(((Pn > 0.1) & (Pn < 0.9)).sum())}, после округления точно {ex:.3f}", flush=True)
            if ex > best_ex:
                best_ex, best = ex, tab
            if ex >= 1.0:
                found = it
                break
    out = {"seed": seed, "точно до 6": round(best_ex, 3), "найдено на шаге": found,
           "секунд": round(time.time() - t0)}
    if found is not None:
        from sat_search import exam
        out["экзамен (до 10, до 1000)"] = exam(best, design, widths, t, items)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    with Pool(a.procs) as pool:
        for r in pool.starmap(run, [(s, a.steps) for s in range(a.restarts)]):
            print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
