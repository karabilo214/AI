"""Проверка: быстрая версия (fast.py) совпадает с обычной (Brain + tasks.py).

1. Без шума (любопытство 0, без изменения нейронов) на нескольких наборах
   параметров: ответы, доли, силы связей, выросшие связи и проверка памяти
   должны совпасть до бита.
2. С шумом: доли успеха на многих запусках должны быть близки
   (генераторы случайных чисел разные, до бита совпадать не могут).
3. Скорость оценки одного набора.

При расхождении в части 1 код выхода 1. --quick: только часть 1.
Запуск из корня проекта: .venv/bin/python -m human_brain.tests.check_fast
"""
import os as _os
import sys as _sys
import random
import sys
import time

import numpy as np

from human_brain.stages import tasks
from human_brain.core import fast
from human_brain.evolution.evolve import mutate


def python_task(p, name, seed):
    """Как tasks.run_task, но ещё возвращает силы связей в конце."""
    parts = tasks.TASKS[name]
    b = tasks.make_reward_brain(p, seed)
    hits, before = [], None
    for k, (inputs, good) in enumerate(parts):
        if k == 1:
            before = tasks.exam(b, inputs) == good
        hits.append(tasks.learn(b, inputs, good))
    ok = all(tasks.exam(b, inputs) == good for inputs, good in parts)
    return ok, sum(hits) / (60 * len(hits)), before, b.weight[:b.n_conn].copy()


def silent(p):
    q = dict(p)
    q["calm"] = 0.0
    q["curious"] = 0.0
    q["neurons_on"] = False
    return q


def main(n_sets=6, seeds=(0, 1, 2), quick=False):
    rnd = random.Random(1)
    base = dict(tasks.PARAMS_DEFAULT)
    sets = [silent(base)]
    while len(sets) < n_sets:
        sets.append(silent(mutate(base, rnd, rnd.randint(3, 10))))
    sets[-1]["init_weight"] = 14.0   # чтобы выходы горели и без шума
    # переключатели задаются явно (включён в части наборов, выключен в остальных):
    # в текущих параметрах всё включено, и прежнее «= True» ничего не меняло
    switches = {"fam_on": (1, 3, 5), "hyp_on": (2, 3, 4, 5), "credit_on": (1, 3, 5),
                "myelin_timing": (2, 4, 5), "forget_on": (1, 2, 5), "ban_on": (0, 3, 4),
                "ctx_on": (0, 2, 4)}
    for key, on in switches.items():
        for k in range(n_sets):
            sets[k][key] = k in on

    print("1. без шума, совпадение до бита:")
    bad = 0
    total = 0
    for k, p in enumerate(sets):
        tasks.apply(p)
        P = fast.params_array(p)
        for name in tasks.REWARD_TASKS:
            inputs, good = fast.task_arrays(tasks.TASKS[name])
            for s in seeds:
                a = python_task(p, name, s)
                ok, hits, before, w = fast.run_reward_task(P, fast.NC_TASKS, fast.CC_TASKS, s, inputs, good)
                b = (bool(ok), float(hits), None if before < 0 else bool(before), w)
                same = a[:3] == b[:3] and len(a[3]) == len(b[3]) and np.array_equal(a[3], b[3])
                total += 1
                if not same:
                    bad += 1
                    if bad <= 5:
                        print(f"   РАЗНИЦА набор {k} {name} seed {s}: обычная {a[:3]} {len(a[3])} связей, "
                              f"быстрая {b[:3]} {len(b[3])} связей")
        for name in tasks.CONTEXT_TASKS:
            for s in seeds[:2]:
                a = tasks.run_context_task(p, name, s)
                b = fast.run_context_task(P, name, s)
                total += 1
                if tuple(a) != tuple(b):
                    bad += 1
                    if bad <= 5:
                        print(f"   РАЗНИЦА набор {k} {name} seed {s}: обычная {a}, быстрая {b}")
        mp, mf = tasks.memory_test(p), fast.memory_test(P, fast.NC_TASKS, fast.CC_TASKS)
        gp, gf = tasks.growth_test(p, 0), fast.growth_test(P, 0)
        rp, rf = tasks.grass_score(p), fast.grass_score(P)
        total += 3
        bad += (mp != mf) + (gp != gf) + (rp != rf)
        print(f"   набор {k} (знак. {p['fam_on']:d}, гип. {p['hyp_on']:d}, засл. {p['credit_on']:d}, миел. {p['myelin_timing']:d}, забыв. {p['forget_on']:d}, запрет {p['ban_on']:d}): память {mp} / {mf}, рост {gp} / {gf}, трава {rp:.2f} / {rf:.2f}")
    print(f"   итог: расхождений {bad} из {total}")
    if quick:
        return bad

    print("2. с шумом, текущие значения, 40 запусков (обычная / быстрая):")
    p = dict(base)
    tasks.apply(p)
    P = fast.params_array(p)
    for name in tasks.REWARD_TASKS:
        a = [tasks.run_task(p, name, s)[0] for s in range(100, 140)]
        b = [fast.run_task(P, name, s)[0] for s in range(100, 140)]
        print(f"   {name:9s}: {sum(a):2d}/40  {sum(b):2d}/40")

    print("3. скорость оценки одного набора (12 запусков):")
    t = time.time()
    s1, _ = tasks.evaluate(p, list(range(12)))
    t1 = time.time() - t
    fast.evaluate(p, [0])   # прогрев (компиляция)
    t = time.time()
    s2, _ = fast.evaluate(p, list(range(12)))
    t2 = time.time() - t
    print(f"   обычная {t1:.2f} с (оценка {s1:.3f}), быстрая {t2:.3f} с (оценка {s2:.3f}), "
          f"быстрее в {t1 / t2:.0f} раз")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(quick="--quick" in sys.argv) else 0)
