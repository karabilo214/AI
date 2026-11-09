"""Отбор: таблицу никто не пишет. Меняем случайные биты и оставляем, если не стало хуже.

Оценка как подзатыльник: насколько ошибся в походах, а не только "верно / неверно". Ищем сразу
популяцией из нескольких таблиц, чтобы реже застревать.

Учится только на числах до 10 и цепочке из 8 элементов. На экзамене: числа до 200 и цепочка
длиннее, из тех же одинаковых элементов, без дообучения.

    python evolve.py --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import logic
from logic import TABLE_SIZE, make_days, run_days

WIDTH = 8
GENERATIONS = 3000
POPULATION = 16
CHILDREN = 32


def fitness(table, days, right):
    """Меньше подзатыльников лучше: средняя ошибка в походах, а при равенстве доля точных дней."""
    got = run_days(table, days, WIDTH)
    return -float(np.abs(got - right).mean()) + 0.001 * float((got == right).mean())


def exact(table, days, right):
    return float((run_days(table, days, WIDTH) == right).mean())


def training_set(seed):
    rng = random.Random(seed)
    pairs = [(t, g) for t in range(11) for g in range(11)] * 2
    return make_days(pairs, rng)


def evolve(seed):
    rng = np.random.default_rng(seed)
    days, right = training_set(seed)
    population = [rng.integers(0, 2, TABLE_SIZE, dtype=np.uint8) for _ in range(POPULATION)]
    scores = [fitness(t, days, right) for t in population]
    started = time.perf_counter()
    history = []
    for generation in range(1, GENERATIONS + 1):
        children = []
        for _ in range(CHILDREN):
            parent = population[rng.integers(POPULATION)]
            child = parent.copy()
            child[rng.choice(TABLE_SIZE, size=rng.integers(1, 5), replace=False)] ^= 1
            children.append(child)
        everyone = population + children
        values = scores + [fitness(c, days, right) for c in children]
        order = np.argsort(values)[::-1][:POPULATION]   # остаются лучшие, старые при равенстве тоже
        population = [everyone[i] for i in order]
        scores = [values[i] for i in order]
        best = population[0]
        if generation % 250 == 0:
            rate = exact(best, days, right)
            history.append((generation, round(-scores[0], 3), round(rate, 3)))
            print(f"seed {seed} поколение {generation}: ошибка {-scores[0]:.3f}, точно {rate:.3f}, секунд {time.perf_counter() - started:.0f}", flush=True)
            if rate == 1.0:
                break
    return population[0], exact(population[0], days, right), history


def exam(table, seed=99):
    rng = random.Random(seed)
    rows = {}
    for name, width, low, high in (("знакомые числа до 10, новый порядок", 8, 0, 10),
                                   ("числа 11-20", 8, 11, 20), ("числа до 60", 8, 0, 60),
                                   ("числа до 200, цепочка 16", 16, 0, 200)):
        pairs = [(rng.randint(low, high), rng.randint(low, high)) for _ in range(400)]
        days, right = make_days(pairs, rng)
        need = right > 0
        logic.MAX_FETCHES = int(right.max()) * 2 + 10   # предел походов всегда с запасом выше самого большого ответа
        got = run_days(table, days, width)
        rows[name] = {"точно": round(float((got == right).mean()), 3),
                      "точно, когда надо идти": round(float((got[need] == right[need]).mean()), 3)}
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table, best, history = evolve(args.seed)
    result = {"seed": args.seed, "обучение": best, "история": history, "экзамен": exam(table),
              "таблица": "".join(map(str, table.tolist()))}
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"seed_{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "история"}, ensure_ascii=False), flush=True)
