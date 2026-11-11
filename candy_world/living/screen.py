"""Отсев способов поиска для «по N каждому» (2026-11-11).

Цель: на малом бюджете увидеть, даёт ли какой-то способ поиска или строения
хоть одну находку там, где обычный отбор застревает: ступень «до 6»
(до 2 гостей — нужно сохранить N для второго). Прежде: 0 из 110 поисков.
Каждый способ — RUNS запусков по BUDGET оценок (одна оценка = прогон всех
дней на 4 длинах цепочки). Удача: точно 1.0 на данных «до 6» на всех длинах.
У удачных — экзамен «до 10» (обучение такого не видело: до 10 гостей) и
«по 1-3, до 1000, цепочка 16».

    python screen.py [--budget 40000] [--runs 4] [--only base,pbil]
"""
import argparse
import json
import math
import random
import time
from multiprocessing import Pool

import numpy as np

import evolve as E
from fast_engine import run_days

BUDGET = 40000
G = {}


def init():
    t, design, widths, items, _ = E.setup("add_steps", "base")
    G.update(t=t, design=design, widths=widths, items=items)
    G["d6"] = E.data(t, items, 6, 0)
    G["d3"] = E.data(t, items, 3, 0)
    rngd = random.Random(0)
    def by_n(ns):
        it = [(a, b, n) for a in range(7) for b in range(3) for n in ns] * 2
        days, right, empty = t.make_days(it, rngd)
        length = max(map(len, days))
        mask = np.zeros((len(days), length), dtype=bool)
        target = np.zeros((len(days), length), dtype=np.uint8)
        for i, row in enumerate(empty):
            for j, v in enumerate(row):
                if v is not None:
                    mask[i, j], target[i, j] = True, v
        return days, right, target, mask
    G["dn"] = [by_n((1,)), by_n((1, 2)), G["d6"]]
    import task_probe as P
    tp, dp, wp, ip, _ = E.setup("add_probe", "base")
    G["probe"] = (tp, E.data(tp, ip, 6, 0))


class Counter:
    def __init__(self, budget):
        self.n = 0
        self.budget = budget
        self.best = None
        self.best_exact = -1.0
        self.solved_at = None

    def left(self):
        return self.n < self.budget and self.solved_at is None

    def fit(self, table, d=None, t=None):
        self.n += 1
        d = G["d6"] if d is None else d
        score, exact = E.fitness(table, G["design"], G["widths"], t or G["t"], d)
        return score, exact

    def note(self, table, exact_on_target):
        if exact_on_target > self.best_exact:
            self.best_exact = exact_on_target
            self.best = table.copy()
            if exact_on_target >= 1.0:
                self.solved_at = self.n


def target_exact(table):
    return E.fitness(table, G["design"], G["widths"], G["t"], G["d6"])[1]


def flips(rng, x, lo=1, hi=4):
    c = x.copy()
    c[rng.choice(c.size, size=rng.integers(lo, hi + 1), replace=False)] ^= 1
    return c


def row_mut(rng, x):
    c = x.copy()
    ob = G["design"].out_bits
    for _ in range(rng.integers(1, 3)):
        r = rng.integers(G["design"].rows)
        c[r * ob:(r + 1) * ob] = rng.integers(0, 2, ob)
    return c


def cases(table, d=None):
    """По дням: верно ли на всех длинах (для лексикографического отбора)."""
    days, right, target, mask = G["d6"] if d is None else d
    ok = np.ones(len(days), dtype=bool)
    for w in G["widths"]:
        got, _ = run_days(table, G["design"], days, w, G["t"].CODES)
        ok &= got == right
    return ok


# ---------- способы ----------

def evo(C, rng, pop=16, kids=32, mutate=None, cross=None, data=None, neutral=False, restart=0, stages=None):
    mutate = mutate or (lambda x: flips(rng, x))
    stage = 0
    d = stages[0] if stages else data
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
    S = [C.fit(p, d) for p in P]
    stale, best_seen = 0, -1e9
    while C.left():
        kids_ = []
        for _ in range(kids):
            if cross is not None and rng.random() < 0.5:
                a, b = P[rng.integers(pop)], P[rng.integers(pop)]
                kids_.append(mutate(cross(rng, a, b)))
            else:
                kids_.append(mutate(P[rng.integers(pop)]))
        KS = [C.fit(k, d) for k in kids_]
        allp, alls = P + kids_, S + KS
        keys = [s[0] + (rng.random() * 1e-9 if neutral else 0) for s in alls]
        if neutral:   # при равенстве потомок вперёд родителя
            keys = [k + (1e-12 if i >= pop else 0) for i, k in enumerate(keys)]
        order = np.argsort(keys)[::-1][:pop]
        P, S = [allp[i] for i in order], [alls[i] for i in order]
        ex = S[0][1] if (d is G["d6"] or d is None) else target_exact(P[0])
        C.note(P[0], ex)
        if stages and S[0][1] >= 0.95 and stage < len(stages) - 1:
            stage += 1
            d = stages[stage]
            S = [C.fit(p, d) for p in P]
        if restart:
            if S[0][0] > best_seen + 1e-12:
                best_seen, stale = S[0][0], 0
            else:
                stale += 1
            if stale >= restart:
                P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
                S = [C.fit(p, d) for p in P]
                stale, best_seen = 0, -1e9


def cross_uniform(rng, a, b):
    m = rng.integers(0, 2, a.size).astype(bool)
    return np.where(m, a, b).astype(np.uint8)


def cross_rows(rng, a, b):
    ob = G["design"].out_bits
    m = np.repeat(rng.integers(0, 2, G["design"].rows).astype(bool), ob)
    return np.where(m, a, b).astype(np.uint8)


def anneal(C, rng):
    x = rng.integers(0, 2, G["design"].size, dtype=np.uint8)
    fx = C.fit(x)
    T0, T1 = 0.05, 0.0005
    while C.left():
        T = T0 * (T1 / T0) ** (C.n / C.budget)
        y = flips(rng, x)
        fy = C.fit(y)
        if fy[0] >= fx[0] or rng.random() < math.exp((fy[0] - fx[0]) / T):
            x, fx = y, fy
        C.note(x, fx[1])


def hill(C, rng):
    x = rng.integers(0, 2, G["design"].size, dtype=np.uint8)
    fx = C.fit(x)
    stale = 0
    while C.left():
        y = flips(rng, x, 1, 1)
        fy = C.fit(y)
        if fy[0] >= fx[0]:
            stale = 0 if fy[0] > fx[0] else stale + 1
            x, fx = y, fy
        else:
            stale += 1
        C.note(x, fx[1])
        if stale > 3000:
            x = rng.integers(0, 2, G["design"].size, dtype=np.uint8)
            fx = C.fit(x)
            stale = 0


def lexicase(C, rng, pop=48):
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
    while C.left():
        S = [C.fit(p) for p in P]
        K = [cases(p) for p in P]
        C.n += 0
        for p, s in zip(P, S):
            C.note(p, s[1])
        new = [P[int(np.argmax([s[0] for s in S]))]]
        while len(new) < pop:
            cand = list(range(pop))
            for c in rng.permutation(len(K[0])):
                best = [i for i in cand if K[i][c]]
                if best:
                    cand = best
                if len(cand) == 1:
                    break
            new.append(flips(rng, P[rng.choice(cand)]))
        P = new


def novelty(C, rng, pop=16, kids=32):
    archive = []
    days, right = G["d6"][0], G["d6"][1]
    def beh(x):
        got, _ = run_days(x, G["design"], days, 8, G["t"].CODES)
        return np.minimum(got, 40).astype(float)
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
    S = [C.fit(p) for p in P]
    B = [beh(p) for p in P]
    while C.left():
        kids_ = [flips(rng, P[rng.integers(pop)]) for _ in range(kids)]
        KS = [C.fit(k) for k in kids_]
        KB = [beh(k) for k in kids_]
        allp, alls, allb = P + kids_, S + KS, B + KB
        ref = np.array(archive[-200:] + allb)
        nov = np.array([np.sort(np.abs(ref - b).mean(1))[1:6].mean() for b in allb])
        fit = np.array([s[0] for s in alls])
        rank = np.argsort(np.argsort(fit)) + np.argsort(np.argsort(nov)) * 0.5
        order = np.argsort(rank)[::-1][:pop]
        P, S, B = [allp[i] for i in order], [alls[i] for i in order], [allb[i] for i in order]
        archive.append(allb[int(np.argmax(nov))])
        bi = int(np.argmax([s[0] for s in alls]))
        C.note(allp[bi], alls[bi][1])


def islands(C, rng, n=4, pop=4, kids=8):
    I = [[rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)] for _ in range(n)]
    S = [[C.fit(p) for p in isl] for isl in I]
    gen = 0
    while C.left():
        gen += 1
        for k in range(n):
            kids_ = [flips(rng, I[k][rng.integers(pop)]) for _ in range(kids)]
            KS = [C.fit(x) for x in kids_]
            allp, alls = I[k] + kids_, S[k] + KS
            order = np.argsort([s[0] for s in alls])[::-1][:pop]
            I[k], S[k] = [allp[i] for i in order], [alls[i] for i in order]
            C.note(I[k][0], S[k][0][1])
        if gen % 50 == 0:
            for k in range(n):
                I[(k + 1) % n][-1] = I[k][0].copy()
                S[(k + 1) % n][-1] = S[k][0]


def pbil(C, rng, n=32, lr=0.1):
    p = np.full(G["design"].size, 0.5)
    while C.left():
        X = (rng.random((n, p.size)) < p).astype(np.uint8)
        F = [C.fit(x) for x in X]
        b = int(np.argmax([f[0] for f in F]))
        C.note(X[b], F[b][1])
        p = (1 - lr) * p + lr * X[b]
        m = rng.random(p.size) < 0.02
        p[m] = 0.95 * p[m] + 0.05 * rng.integers(0, 2, m.sum())
        p = np.clip(p, 0.02, 0.98)


def cem(C, rng, n=64, elite=8):
    p = np.full(G["design"].size, 0.5)
    while C.left():
        X = (rng.random((n, p.size)) < p).astype(np.uint8)
        F = [C.fit(x) for x in X]
        order = np.argsort([f[0] for f in F])[::-1]
        C.note(X[order[0]], F[order[0]][1])
        p = np.clip(0.7 * p + 0.3 * X[order[:elite]].mean(0), 0.02, 0.98)


def probes(C, rng):
    tp, dp = G["probe"]
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(16)]
    S = [C.fit(p, dp, tp) for p in P]
    while C.left():
        kids_ = [flips(rng, P[rng.integers(16)]) for _ in range(32)]
        KS = [C.fit(k, dp, tp) for k in kids_]
        allp, alls = P + kids_, S + KS
        order = np.argsort([s[0] for s in alls])[::-1][:16]
        P, S = [allp[i] for i in order], [alls[i] for i in order]
        if S[0][1] >= C.best_exact:
            C.note(P[0], target_exact(P[0]))


def exact_only(C, rng):
    """Отбор только по доле верных дней (без плотного сигнала и без |разницы|)."""
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(16)]
    S = [C.fit(p) for p in P]
    while C.left():
        kids_ = [flips(rng, P[rng.integers(16)]) for _ in range(32)]
        KS = [C.fit(k) for k in kids_]
        allp, alls = P + kids_, S + KS
        order = np.argsort([s[1] + rng.random() * 1e-6 for s in alls])[::-1][:16]
        P, S = [allp[i] for i in order], [alls[i] for i in order]
        C.note(P[0], S[0][1])


def relevant_bits():
    ob = G["design"].out_bits
    return np.array([r * ob + k for r in range(G["design"].rows) if 1 <= (r >> 4) <= 4 for k in range(ob)])


def flips_rel(rng, x, lo=1, hi=4):
    """Переворот только в строках событий, которые встречаются (коды 1..4)."""
    rel = G.setdefault("rel", relevant_bits())
    c = x.copy()
    c[rng.choice(rel, size=rng.integers(lo, hi + 1), replace=False)] ^= 1
    return c


def lexicase_rel(C, rng, pop=48):
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
    while C.left():
        S = [C.fit(p) for p in P]
        K = [cases(p) for p in P]
        for p, s_ in zip(P, S):
            C.note(p, s_[1])
        new = [P[int(np.argmax([x[0] for x in S]))]]
        while len(new) < pop:
            cand = list(range(pop))
            for c in rng.permutation(len(K[0])):
                best = [i for i in cand if K[i][c]]
                if best:
                    cand = best
                if len(cand) == 1:
                    break
            new.append(flips_rel(rng, P[rng.choice(cand)]))
        P = new


def probe_acc(table):
    """Доля верных ответов на вопросы «бит N?» (данные task_probe, ступень «до 6»)."""
    tp, dp = G["probe"]
    days, right, target, mask = dp
    if "pmask" not in G:
        pm = np.zeros_like(mask)
        for i, ev in enumerate(days):
            for j, e in enumerate(ev):
                pm[i, j] = e in (5, 6)
        G["pmask"] = pm
    pm = G["pmask"]
    acc = []
    for w in G["widths"]:
        _, said, _ = run_days(table, G["design"], days, w, tp.CODES, record=True)
        acc.append(((said == target) & pm).sum() / pm.sum())
    return float(min(acc))


def parts(C, rng, pop=16, kids=32):
    """По частям: сначала только «помнить N» (вопросы о N), потом полная задача."""
    P = [rng.integers(0, 2, G["design"].size, dtype=np.uint8) for _ in range(pop)]
    S = [probe_acc(p) for p in P]; C.n += pop
    while C.left() and C.n < C.budget // 2 and max(S) < 1.0:
        kids_ = [flips(rng, P[rng.integers(pop)]) for _ in range(kids)]
        KS = [probe_acc(k) for k in kids_]; C.n += kids
        allp, alls = P + kids_, S + KS
        order = np.argsort(alls)[::-1][:pop]
        P, S = [allp[i] for i in order], [alls[i] for i in order]
    G.setdefault("parts_log", []).append(round(max(S), 3))
    S = [C.fit(p) for p in P]
    while C.left():
        kids_ = [flips(rng, P[rng.integers(pop)]) for _ in range(kids)]
        KS = [C.fit(k) for k in kids_]
        allp, alls = P + kids_, S + KS
        order = np.argsort([x[0] for x in alls])[::-1][:pop]
        P, S = [allp[i] for i in order], [alls[i] for i in order]
        C.note(P[0], S[0][1])


METHODS = {
    "base": lambda C, r: evo(C, r),
    "base_staged": lambda C, r: evo(C, r, stages=[G["d3"], G["d6"]]),
    "big_pop": lambda C, r: evo(C, r, pop=64, kids=128),
    "small_pop": lambda C, r: evo(C, r, pop=4, kids=8),
    "large_mut": lambda C, r: evo(C, r, mutate=lambda x: flips(r, x, 1, 16)),
    "single_mut": lambda C, r: evo(C, r, mutate=lambda x: flips(r, x, 1, 1)),
    "row_mut": lambda C, r: evo(C, r, mutate=lambda x: row_mut(r, x)),
    "neutral": lambda C, r: evo(C, r, neutral=True),
    "restarts": lambda C, r: evo(C, r, restart=150),
    "ga_uniform": lambda C, r: evo(C, r, cross=cross_uniform),
    "ga_rows": lambda C, r: evo(C, r, cross=cross_rows),
    "curriculum_N": lambda C, r: evo(C, r, stages=G["dn"]),
    "anneal": anneal,
    "hill": hill,
    "lexicase": lexicase,
    "novelty": novelty,
    "islands": islands,
    "pbil": pbil,
    "cem": cem,
    "probes": probes,
    "exact_only": exact_only,
    "relevant_bits": lambda C, r: evo(C, r, mutate=lambda x: flips_rel(r, x)),
    "relevant_lexicase": lexicase_rel,
    "parts": parts,
}


def exam(table):
    d10 = E.data(G["t"], G["items"], 10, 1)
    e10 = E.fitness(table, G["design"], G["widths"], G["t"], d10)[1]
    r = random.Random(99)
    days, right, _ = G["t"].make_days([(r.randint(0, 1000), r.randint(0, 1000), r.randint(1, 3)) for _ in range(200)], r)
    got, _ = run_days(table, G["design"], days, 16, G["t"].CODES, max_fetches=int(right.max()) * 2 + 10)
    return round(float(e10), 3), round(float((got == right).mean()), 3)


def job(args):
    name, seed, budget = args
    if not G:
        init()
    rng = np.random.default_rng(seed)
    C = Counter(budget)
    t0 = time.time()
    METHODS[name](C, rng)
    out = {"способ": name, "seed": seed, "точно до 6": round(C.best_exact, 3),
           **({"помнит N после 1-й части": G["parts_log"][-1]} if name == "parts" else {}),
           "найдено на оценке": C.solved_at, "секунд": round(time.time() - t0)}
    if C.solved_at is not None:
        out["экзамен (до 10, до 1000)"] = exam(C.best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=9)
    a = ap.parse_args()
    names = [n for n in METHODS if not a.only or n in a.only.split(",")]
    jobs = [(n, s, a.budget) for n in names for s in range(a.runs)]
    res = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    print("\nитог (способ: находок / запусков, лучшее «точно до 6» по запускам):")
    for n in names:
        rr = [r for r in res if r["способ"] == n]
        print(f"  {n:13s} {sum(r['найдено на оценке'] is not None for r in rr)}/{len(rr)}  "
              f"{sorted(r['точно до 6'] for r in rr)}")
    with open("results_screen.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
