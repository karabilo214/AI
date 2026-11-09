"""Отбор с плотным сигналом и постепенностью.

После каждого события цепочка предсказывает, что покажет мир: на столе пусто или лежат лишние
предметы. Это видимое последствие, а не число. Ошибка видна на каждом шаге, а не одна за день.
Числа растут ступенями: до 3, до 6, до 10. На следующую ступень переходим при 95% точных дней.

    python evolve_dense.py --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from evolve import exam
from logic import TABLE_SIZE, make_days, run_days, table_is_empty

WIDTH = 8
STAGES = (3, 6, 10)
GENERATIONS = 6000
POPULATION = 16
CHILDREN = 32


def training_set(limit, seed):
    rng = random.Random(seed)
    pairs = [(t, g) for t in range(limit + 1) for g in range(limit + 1)] * 3
    days, right = make_days(pairs, rng)
    empty = table_is_empty(days)
    mask = np.zeros((len(days), max(map(len, days), default=0)), dtype=bool)
    target = np.zeros(mask.shape, dtype=np.uint8)
    for i, row in enumerate(empty):
        mask[i, : len(row)] = True
        target[i, : len(row)] = row
    return days, right, target, mask


def fitness(table, data):
    days, right, target, mask = data
    fetched, said = run_days(table, days, WIDTH, record=True)
    seen_errors = float(((said != target) & mask).sum() / max(mask.sum(), 1))
    trips = float(np.abs(fetched - right).mean())
    exact = float((fetched == right).mean())
    return -(trips + seen_errors) + 0.001 * exact, exact, seen_errors


def evolve(seed):
    rng = np.random.default_rng(seed)
    stage = 0
    data = training_set(STAGES[stage], seed)
    population = [rng.integers(0, 2, TABLE_SIZE, dtype=np.uint8) for _ in range(POPULATION)]
    scores = [fitness(t, data)[0] for t in population]
    started, history = time.perf_counter(), []
    for generation in range(1, GENERATIONS + 1):
        children = []
        for _ in range(CHILDREN):
            child = population[rng.integers(POPULATION)].copy()
            child[rng.choice(TABLE_SIZE, size=rng.integers(1, 5), replace=False)] ^= 1
            children.append(child)
        everyone = population + children
        values = scores + [fitness(c, data)[0] for c in children]
        order = np.argsort(values)[::-1][:POPULATION]
        population, scores = [everyone[i] for i in order], [values[i] for i in order]
        if generation % 100 == 0:
            _, exact, seen = fitness(population[0], data)
            history.append((generation, STAGES[stage], round(exact, 3), round(seen, 3)))
            if generation % 500 == 0:
                print(f"seed {seed} поколение {generation}, числа до {STAGES[stage]}: точно {exact:.3f}, "
                      f"ошибки предсказания {seen:.3f}, секунд {time.perf_counter() - started:.0f}", flush=True)
            if exact >= 0.95:
                if stage == len(STAGES) - 1:
                    if exact == 1.0:
                        break
                else:
                    stage += 1
                    data = training_set(STAGES[stage], seed)
                    scores = [fitness(t, data)[0] for t in population]
                    print(f"seed {seed}: ступень пройдена, теперь числа до {STAGES[stage]}", flush=True)
    return population[0], fitness(population[0], data)[1], STAGES[stage], history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table, exact, reached, history = evolve(args.seed)
    result = {"seed": args.seed, "дошёл до ступени": reached, "точно на ней": exact, "история": history,
              "экзамен": exam(table), "таблица": "".join(map(str, table.tolist()))}
    out = Path(__file__).parent / "results_dense"
    out.mkdir(exist_ok=True)
    (out / f"seed_{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("история", "таблица")}, ensure_ascii=False), flush=True)
