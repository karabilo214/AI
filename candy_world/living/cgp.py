"""Элемент из маленьких частей (2026-11-11): схема из узлов с 2 входами (CGP).

Прежде элемент цепочки — одна таблица на все 7 входов (3 бита кода, 2 бита
памяти, 2 сигнала) = 128 строк × 4 выхода = 512 бит; сумматор в ней не
находится. Здесь элемент — маленькая схема: NODES узлов, у каждого два входа
(любые из 7 входов элемента или более ранних узлов) и своя таблица на 4 бита
(любая из 16 функций двух битов); 4 выхода элемента берутся с любых узлов
или входов. Поиск — декартово генетическое программирование (Miller):
(1 + LAMBDA), потомок меняет гены (вход узла, функцию, выход) с вероятностью
RATE, принимается, если не хуже родителя (нейтральный дрейф). Схема
переводится в ту же таблицу 128 × 4 и считается тем же быстрым движком —
мир, оценка, ступени и экзамены те же, что у evolve.py.

    python cgp.py --task adder [--runs 8] [--budget 40000]
"""
import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np

import evolve as E
from fast_engine import run_days

NODES = 60
LAMBDA = 4
RATE = 0.04
N_IN = 7
N_OUT = 4
ROWS = np.arange(128)
X = [((ROWS >> (6 - i)) & 1).astype(np.uint8) for i in range(N_IN)]   # код(3), c, n, a, b


def random_genome(rng):
    g = []
    for k in range(NODES):
        g += [int(rng.integers(N_IN + k)), int(rng.integers(N_IN + k)), int(rng.integers(16))]
    g += [int(rng.integers(N_IN + NODES)) for _ in range(N_OUT)]
    return np.array(g)


def compile_table(g):
    vals = list(X)
    for k in range(NODES):
        i1, i2, f = g[3 * k], g[3 * k + 1], g[3 * k + 2]
        idx = 2 * vals[i1] + vals[i2]
        vals.append(((f >> idx) & 1).astype(np.uint8))
    outs = [vals[o] for o in g[3 * NODES:]]
    return np.stack(outs, 1).reshape(-1)


def mutate(g, rng):
    c = g.copy()
    while True:
        changed = False
        for pos in range(len(c)):
            if rng.random() < RATE:
                changed = True
                if pos >= 3 * NODES:
                    c[pos] = rng.integers(N_IN + NODES)
                else:
                    k, part = divmod(pos, 3)
                    c[pos] = rng.integers(16) if part == 2 else rng.integers(N_IN + k)
        if changed:
            return c


def job(args):
    task, seed, budget = args
    t, design, widths, items, exams = E.setup(task, "base")
    rng = np.random.default_rng(seed)
    stage = 0
    d = E.data(t, items, E.STAGES[stage], seed)
    parent = random_genome(rng)
    pf = E.fitness(compile_table(parent), design, widths, t, d)
    n, t0, solved = 1, time.time(), None
    history = []
    while n < budget:
        kids = [mutate(parent, rng) for _ in range(LAMBDA)]
        fs = [E.fitness(compile_table(k), design, widths, t, d) for k in kids]
        n += LAMBDA
        b = int(np.argmax([f[0] for f in fs]))
        if fs[b][0] >= pf[0]:
            parent, pf = kids[b], fs[b]
        if n % 2000 < LAMBDA:
            history.append((n, E.STAGES[stage], round(pf[1], 3)))
        if pf[1] >= 0.95 and stage < len(E.STAGES) - 1:
            stage += 1
            d = E.data(t, items, E.STAGES[stage], seed)
            pf = E.fitness(compile_table(parent), design, widths, t, d)
        elif pf[1] >= 1.0 and stage == len(E.STAGES) - 1:
            solved = n
            break
    table = compile_table(parent)
    exam = {}
    r = random.Random(99)
    for name, w, draw in exams:
        days, right, _ = t.make_days([draw(r) for _ in range(300)], r)
        got, _ = run_days(table, design, days, w, t.CODES, max_fetches=int(right.max()) * 2 + 10)
        exam[name] = round(float((got == right).mean()), 3)
    return {"задача": task, "seed": seed, "ступень": E.STAGES[stage], "точно на ступени": round(pf[1], 3),
            "решено на оценке": solved, "секунд": round(time.time() - t0), "экзамен": exam,
            "история": history[-3:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="adder")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(job, [(a.task, s, a.budget) for s in range(a.runs)]):
            print(json.dumps({k: v for k, v in r.items() if k != "история"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
