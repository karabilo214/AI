"""Сравнение механизмов (гипотезы, заслуга, миелин по времени, знакомость)
на наборе задач tasks.py быстрой оценкой, на новых запусках.

Запуск из папки human_brain: ../.venv/bin/python compare_mechanisms.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import tasks
import fast

# У каждого варианта полный набор сравниваемых переключателей: контроль явно
# выключает все три (в текущих параметрах они уже включены, и прежний вариант
# «без новых» = {} ничего не выключал — все пять вариантов были одинаковыми).
MECHANISMS = ("hyp_on", "credit_on", "myelin_timing")


def _flags(*on):
    return {m: m in on for m in MECHANISMS}


VARIANTS = {
    "без новых": _flags(),
    "+ гипотезы": _flags("hyp_on"),
    "+ заслуга": _flags("credit_on"),
    "+ миелин по времени": _flags("myelin_timing"),
    "все три": _flags(*MECHANISMS),
}


def check_unique(configs):
    """Все варианты должны отличаться набором параметров."""
    keys = [tuple(sorted(p.items())) for p in configs]
    if len(set(keys)) != len(keys):
        raise ValueError("одинаковые варианты сравнения: сравнение бессмысленно")


def ev(args):
    p, seeds = args
    return fast.evaluate(p, seeds)


def main():
    seeds = list(range(5000, 5050))
    bases = [("текущие", dict(tasks.PARAMS_DEFAULT))]
    best_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolve_results", "best.json")
    if os.path.exists(best_path):
        bases.append(("эволюции", {**tasks.PARAMS_DEFAULT, **json.load(open(best_path))["params"]}))
    jobs = [(name, bname, {**base, **v}) for bname, base in bases for name, v in VARIANTS.items()]
    for bname, _ in bases:
        check_unique([p for _, b, p in jobs if b == bname])
    t = time.time()
    with Pool(9) as pool:
        res = pool.map(ev, [(p, seeds) for _, _, p in jobs])
    for (name, bname, _), (score, d) in zip(jobs, res):
        print(f"{bname:9s} {name:20s}: награда {d['reward']:.2f} (A {d['A']:.2f}, "
              f"same_L {d['same_L']:.2f}, B {d['B']:.2f}), память {d['memory']:.2f}, "
              f"рост {d['growth']:.2f}, верных при обучении {d['hits']:.2f}", flush=True)
    print(f"[{time.time() - t:.0f} с, {len(seeds)} запусков на задачу]")


if __name__ == "__main__":
    main()
