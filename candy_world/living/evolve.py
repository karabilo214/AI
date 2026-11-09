"""Отбор таблиц для любой задачи и любого устройства цепочки.

    python evolve.py --task count --variant types --seed 0
    python evolve.py --task multiply --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from engine import Design
from fast_engine import run_days   # тот же расчёт, переведённый в машинный код (проверка: check_fast_engine.py)

VARIANTS = {
    "base": Design(),
    "types": Design(types=3),
    "bidir": Design(left=1, passes=2),
    "skip": Design(skip=True),
    "all": Design(types=3, left=1, passes=2, skip=True),
}
STAGES = (3, 6, 10)
GENERATIONS = 6000
POPULATION, CHILDREN = 16, 32


def setup(task, variant):
    if task in ("guests", "parity", "double"):
        import task_ops

        class Task:
            CODES = task_ops.CODES

            @staticmethod
            def make_days(pairs, rng):
                return task_ops.make_days(task, pairs, rng)

        design = task_ops.DESIGNS[task]
        widths = (6, 7, 8, 9) if task == "double" else (5, 6, 7, 8)

        def items(limit, rng, exam=False):
            return [(a, b) for a in range(limit + 1) for b in range(limit + 1)] * 3

        exams = (("числа до 10", 9, lambda r: (r.randint(0, 10), r.randint(0, 10))),
                 ("числа до 200, цепочка 16", 16, lambda r: (r.randint(0, 200), r.randint(0, 200))),
                 ("числа до 1000, цепочка 16", 16, lambda r: (r.randint(0, 1000), r.randint(0, 1000))))
        return Task, design, widths, items, exams
    if task == "count":
        import task_count as t
        design = VARIANTS[variant]
        widths = (5, 6, 7, 8)

        def items(limit, rng, exam=False):
            return [(a, b) for a in range(limit + 1) for b in range(limit + 1)] * 3

        exams = (("числа до 10", 8, lambda r: (r.randint(0, 10), r.randint(0, 10))),
                 ("числа до 200, цепочка 16", 16, lambda r: (r.randint(0, 200), r.randint(0, 200))),
                 ("числа до 1000, цепочка 16", 16, lambda r: (r.randint(0, 1000), r.randint(0, 1000))))
    else:
        import task_multiply as t
        design = t.DESIGN
        widths = (7, 8, 9, 10)

        def items(limit, rng, exam=False):
            if task == "add_steps":
                # Ступени по числу гостей: до 3 - не больше 1 гостя (прибавить N один раз),
                # до 6 - не больше 2 (надо сохранить N для второго), до 10 - сколько угодно.
                most = {3: 1, 6: 2}.get(limit, limit)
                return [(a, b, n) for a in range(limit + 1) for b in range(most + 1) for n in (1, 2, 3)] * 2
            return [(a, b, n) for a in range(limit + 1) for b in range(limit + 1) for n in (1, 2, 3)]

        exams = (("по 1-3, числа до 10", 10, lambda r: (r.randint(0, 10), r.randint(0, 10), r.randint(1, 3))),
                 ("по 4-5 (не учился), до 10", 10, lambda r: (r.randint(0, 10), r.randint(0, 10), r.randint(4, 5))),
                 ("по 7 (не учился), до 200, цепочка 16", 16, lambda r: (r.randint(0, 200), r.randint(0, 200), 7)),
                 ("по 1-3, до 1000, цепочка 16", 16, lambda r: (r.randint(0, 1000), r.randint(0, 1000), r.randint(1, 3))))
    return t, design, widths, items, exams


def data(t, items, limit, seed):
    rng = random.Random(seed)
    days, right, empty = t.make_days(items(limit, rng), rng)
    length = max(map(len, days))
    mask = np.zeros((len(days), length), dtype=bool)
    target = np.zeros((len(days), length), dtype=np.uint8)
    for i, row in enumerate(empty):
        for j, v in enumerate(row):
            if v is not None:
                mask[i, j], target[i, j] = True, v
    return days, right, target, mask


def fitness(table, design, widths, t, d):
    days, right, target, mask = d
    total, worst = 0.0, 1.0
    for width in widths:
        fetched, said, _ = run_days(table, design, days, width, t.CODES, record=True)
        total += float(np.abs(fetched - right).mean()) + float(((said != target) & mask).sum() / max(mask.sum(), 1))
        worst = min(worst, float((fetched == right).mean()))
    return -total / len(widths) + 0.001 * worst, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["count", "multiply", "add_steps", "guests", "parity", "double"], default="count")
    parser.add_argument("--variant", choices=list(VARIANTS), default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", action="store_true", help="сохранять лучшую таблицу каждые 100 поколений")
    args = parser.parse_args()
    t, design, widths, items, exams = setup(args.task, args.variant)
    rng = np.random.default_rng(args.seed)
    stage = 0
    d = data(t, items, STAGES[stage], args.seed)
    population = [rng.integers(0, 2, design.size, dtype=np.uint8) for _ in range(POPULATION)]
    scores = [fitness(p, design, widths, t, d)[0] for p in population]
    started, history, solved_at = time.perf_counter(), [], None
    saved_tables = []
    label = f"{args.task}/{args.variant}/{args.seed}"
    for generation in range(1, GENERATIONS + 1):
        children = []
        for _ in range(CHILDREN):
            child = population[rng.integers(POPULATION)].copy()
            child[rng.choice(design.size, size=rng.integers(1, 5), replace=False)] ^= 1
            children.append(child)
        everyone = population + children
        values = scores + [fitness(c, design, widths, t, d)[0] for c in children]
        order = np.argsort(values)[::-1][:POPULATION]
        population, scores = [everyone[i] for i in order], [values[i] for i in order]
        if generation % 100 == 0:
            exact = fitness(population[0], design, widths, t, d)[1]
            history.append((generation, STAGES[stage], round(exact, 3)))
            if args.checkpoints:
                saved_tables.append((generation, STAGES[stage], "".join(map(str, population[0].tolist()))))
            if generation % 500 == 0:
                print(f"{label} поколение {generation}, до {STAGES[stage]}: точно {exact:.3f}, "
                      f"секунд {time.perf_counter() - started:.0f}", flush=True)
            if exact >= 0.95 and stage < len(STAGES) - 1:
                stage += 1
                d = data(t, items, STAGES[stage], args.seed)
                scores = [fitness(p, design, widths, t, d)[0] for p in population]
                print(f"{label}: ступень пройдена, теперь до {STAGES[stage]}", flush=True)
            elif exact == 1.0 and stage == len(STAGES) - 1:
                solved_at = generation
                break
    best = population[0]
    rng_exam = random.Random(99)
    exam = {}
    for name, width, draw in exams:
        days, right, _ = t.make_days([draw(rng_exam) for _ in range(300)], rng_exam)
        limit = int(right.max()) * 2 + 10        # предел походов всегда с запасом выше самого большого ответа
        got, cost = run_days(best, design, days, width, t.CODES, max_fetches=limit)
        need = right > 0
        exam[name] = {"точно": round(float((got == right).mean()), 3),
                      "когда надо идти": round(float((got[need] == right[need]).mean()), 3),
                      "дней упёрлось в предел походов": int((got >= limit).sum()),
                      **{k: round(v, 2) for k, v in cost.items()}}
    result = {"задача": args.task, "вариант": args.variant, "seed": args.seed, "бит в таблицах": design.size,
              "решено на поколении": solved_at, "ступень": STAGES[stage], "секунд": round(time.perf_counter() - started),
              "история": history, "экзамен": exam, "таблица": "".join(map(str, best.tolist())),
              "таблицы по ходу": saved_tables}
    out = Path(__file__).parent / ("results_history" if args.checkpoints else "results")
    out.mkdir(exist_ok=True)
    (out / f"{args.task}__{args.variant}__seed_{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("история", "таблица")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
