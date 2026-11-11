"""Насколько широка «яма» вокруг решения «по N каждому» (2026-11-11).

Ручная таблица (task_multiply.hand_table, задача add_steps) портится на k
случайных бит — только в строках событий 1..4 (остальные коды не встречаются).
Отбор тот же, что в evolve.py (популяция 16, 32 потомка, переворот 1-4 бит),
но сразу на ступени «до 10» (все количества гостей), не больше GENS
поколений. Удача: точно 1.0 на обучении и на экзамене «по 1-3, до 1000,
цепочка 16». Печатает для каждого k: сколько из RUNS вернулись и за сколько поколений.

    python basin.py
"""
import random
import time

import numpy as np

import evolve as E
import task_multiply as M
from fast_engine import run_days

KS = (5, 10, 20, 40, 80)
RUNS = 4
GENS = 3000


def used_bits(design):
    per_row = design.out_bits
    bits = []
    for row in range(design.rows):
        if 1 <= (row >> 4) <= 4:
            bits += list(range(row * per_row, (row + 1) * per_row))
    return np.array(bits)


def exam_ok(table, design, t):
    r = random.Random(99)
    days, right, _ = t.make_days([(r.randint(0, 1000), r.randint(0, 1000), r.randint(1, 3)) for _ in range(200)], r)
    got, _ = run_days(table, design, days, 16, t.CODES, max_fetches=int(right.max()) * 2 + 10)
    return float((got == right).mean())


def search(start, design, widths, t, d, rng):
    population = []
    used = used_bits(design)
    for _ in range(E.POPULATION):
        p = start.copy()
        population.append(p)
    scores = [E.fitness(p, design, widths, t, d)[0] for p in population]
    for gen in range(1, GENS + 1):
        children = []
        for _ in range(E.CHILDREN):
            c = population[rng.integers(E.POPULATION)].copy()
            c[rng.choice(design.size, size=rng.integers(1, 5), replace=False)] ^= 1
            children.append(c)
        everyone = population + children
        values = scores + [E.fitness(c, design, widths, t, d)[0] for c in children]
        order = np.argsort(values)[::-1][:E.POPULATION]
        population, scores = [everyone[i] for i in order], [values[i] for i in order]
        if gen % 50 == 0 and E.fitness(population[0], design, widths, t, d)[1] == 1.0:
            return gen, population[0]
    return None, population[0]


def main():
    t, design, widths, items, _ = E.setup("add_steps", "base")
    d = E.data(t, items, 10, 0)
    hand = M.hand_table()
    print("ручная: обучение", E.fitness(hand, design, widths, t, d)[1], "экзамен", exam_ok(hand, design, t), flush=True)
    used = used_bits(design)
    for k in KS:
        res = []
        for run in range(RUNS):
            rng = np.random.default_rng(1000 * k + run)
            start = hand.copy()
            start[rng.choice(used, size=k, replace=False)] ^= 1
            t0 = time.time()
            gen, best = search(start, design, widths, t, d, rng)
            ok = gen is not None and exam_ok(best, design, t) == 1.0
            res.append((gen if ok else None, round(E.fitness(start, design, widths, t, d)[1], 2),
                        round(time.time() - t0)))
        print(f"испорчено {k:3d} бит: вернулись {sum(r[0] is not None for r in res)}/{RUNS}; "
              f"(поколений, точность испорченной, секунд): {res}", flush=True)


if __name__ == "__main__":
    main()
