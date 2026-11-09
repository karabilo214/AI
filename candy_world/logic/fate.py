"""Решается ли судьба поиска заранее? Продолжаем удачный поиск с точки до скачка, но с другими случайностями.

Для сравнения: продолжение с той же точки, но у поиска, который застрял (с той же ступени и таким же
числом поколений), и старт с нуля. Отбор тот же, что в living/evolve.py (обычная цепочка, счёт).

    python fate.py --run 4 --at 1000 --tries 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LIVING = Path(__file__).resolve().parent.parent / "living"
sys.path.insert(0, str(LIVING))
import evolve  # noqa: E402

BUDGET = 1500


def continue_from(table, seed):
    t, design, widths, items, exams = evolve.setup("count", "base")
    rng = np.random.default_rng(10_000 + seed)
    stage = 0
    d = evolve.data(t, items, evolve.STAGES[stage], seed)
    if table is None:
        population = [rng.integers(0, 2, design.size, dtype=np.uint8) for _ in range(evolve.POPULATION)]
    else:
        population = [table.copy() for _ in range(evolve.POPULATION)]
    scores = [evolve.fitness(p, design, widths, t, d)[0] for p in population]
    for generation in range(1, BUDGET + 1):
        children = []
        for _ in range(evolve.CHILDREN):
            child = population[rng.integers(evolve.POPULATION)].copy()
            child[rng.choice(design.size, size=rng.integers(1, 5), replace=False)] ^= 1
            children.append(child)
        everyone = population + children
        values = scores + [evolve.fitness(c, design, widths, t, d)[0] for c in children]
        order = np.argsort(values)[::-1][: evolve.POPULATION]
        population, scores = [everyone[i] for i in order], [values[i] for i in order]
        if generation % 50 == 0:
            exact = evolve.fitness(population[0], design, widths, t, d)[1]
            if exact >= 0.95:
                return generation      # прошёл ступень "до 3": по разбору это и есть находка правила
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--at", type=int, required=True, help="поколение, с которого продолжаем; 0: старт с нуля")
    parser.add_argument("--tries", type=int, default=8)
    args = parser.parse_args()
    table = None
    if args.at:
        d = json.loads((LIVING / "results_history" / f"count__base__seed_{args.run}.json").read_text(encoding="utf-8"))
        text = next(t for g, s, t in d["таблицы по ходу"] if g == args.at)
        table = np.array([int(c) for c in text], dtype=np.uint8)
    found = [continue_from(table, k) for k in range(args.tries)]
    result = {"поиск": args.run, "с поколения": args.at, "бюджет": BUDGET, "прошли ступень за поколений": found,
              "доля": sum(f is not None for f in found) / len(found)}
    out = Path(__file__).parent / "results_fate"
    out.mkdir(exist_ok=True)
    (out / f"run_{args.run}__at_{args.at}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
