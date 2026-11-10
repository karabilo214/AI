"""Эволюция параметров всего мозга под сложение, не ломая прежние задачи.

Гены: все параметры tasks.SPACE плюс параметры сложения: inhibit_on (обучаемое
торможение вход -> действие), inh_plasticity (его скорость), boredom (скука).
Оценка набора на запусках отбора:
    3 * (сложение: доля верных на свежих днях с числами до 10, обучение 300 дней)
    + оценка прежних задач (fast.evaluate: по награде, память, рост, трава).
Числа больше 10 в отбор не входят: это контроль (сложит ли любые).
В конце: 5 лучших и стартовый набор на запусках проверки (выбор победителя),
затем победитель и стартовый набор один раз на запусках теста со всеми
экзаменами сложения (до 10, 11-20, 21-50, 51-100, 101-1000). Наборы запусков
не пересекаются. Исходные значения (tasks.PARAMS_DEFAULT) не меняются:
победителя принимает владелец.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.evolution.evolve_addition --minutes 40
Результаты: human_brain/data/evolve_addition_results/log.txt, best.json
"""
import argparse
import json
import math
import os
import random
import time
from multiprocessing import Pool

from human_brain.stages import tasks
from human_brain.core import fast
from human_brain.stages import addition
from human_brain.parts import Head

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # human_brain
EXTRA_DEFAULT = {"inhibit_on": False, "inh_plasticity": Head.INH_PLASTICITY,
                 "boredom": Head.BOREDOM}
SPACE = dict(tasks.SPACE)
SPACE.update({"inhibit_on": ("bool",), "inh_plasticity": ("log", 0.1, 20.0),
              "boredom": ("lin", 0.0, 3.0)})
ALL_EXAMS = None   # все наборы


def mutate(p, rnd, n_genes):
    child = dict(p)
    for name in rnd.sample(list(SPACE), n_genes):
        kind, *limits = SPACE[name]
        v = child[name]
        if kind == "bool":
            v = not v
        elif kind == "int":
            lo, hi = limits
            v = min(hi, max(lo, v + rnd.choice([-2, -1, 1, 2])))
        elif kind == "log":
            lo, hi = limits
            v = min(hi, max(lo, v * math.exp(rnd.gauss(0, 0.4))))
        elif kind == "one_minus":
            lo, hi = limits
            v = 1.0 - min(hi, max(lo, (1.0 - v) * math.exp(rnd.gauss(0, 0.4))))
        else:
            lo, hi = limits
            v = min(hi, max(lo, v + rnd.gauss(0, 0.15 * (hi - lo))))
        child[name] = v
    return child


def crossover(a, b, rnd):
    return {k: (a[k] if rnd.random() < 0.5 else b[k]) for k in a}


def evaluate(args):
    """(параметры, запуски сложения, запуски прежних задач, все экзамены?) -> (оценка, подробности)."""
    p, add_seeds, main_seeds, full = args
    try:
        q = dict(p, fam_on=False)
        sets = None if full else ["до 10, свежие дни"]
        res = [addition.run(q, s, sets=sets) for s in add_seeds]
        detail = {k: sum(r[k] for r in res) / len(res) for k in res[0]}
        main_score, main_detail = fast.evaluate(p, main_seeds)
        detail["прежние задачи"] = main_score
        detail.update({f"  {k}": v for k, v in main_detail.items() if k in ("reward", "B", "memory", "growth", "grass")})
        return 3 * detail["до 10, свежие дни"] + main_score, detail
    except Exception as e:
        return -1.0, {"error": f"{type(e).__name__}: {e}"}


def changes(p, base):
    return [f"{k}: {base[k]} -> {round(v, 5) if isinstance(v, float) else v}"
            for k, v in p.items() if v != base.get(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--children", type=int, default=32)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "evolve_addition_results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "w")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    rnd = random.Random(a.seed)
    train_add, train_main = list(range(6)), list(range(12))
    val_add, val_main = list(range(100000, 100020)), list(range(100000, 100030))
    test_add, test_main = list(range(200000, 200020)), list(range(200000, 200050))
    base = {**tasks.PARAMS_DEFAULT, **EXTRA_DEFAULT}
    start = time.time()
    with Pool(a.workers) as pool:
        def score(sets, add_s, main_s, full=False):
            return pool.map(evaluate, [(p, add_s, main_s, full) for p in sets])

        population = [base] + [mutate(base, rnd, rnd.randint(1, 3)) for _ in range(a.population - 1)]
        scored = sorted(zip(score(population, train_add, train_main), population),
                        key=lambda x: -x[0][0])
        say(f"стартовый набор на отборе: {[s for s, p in scored if p is base][0][0]:.3f}")
        gen_start, last = time.time(), 0.0
        for gen in range(1, a.generations + 1):
            if time.time() - gen_start + last > a.minutes * 60:
                say(f"остановлено по времени после {gen - 1} поколений")
                break
            t0 = time.time()
            parents = [p for _, p in scored]
            kids = []
            for _ in range(a.children):
                x = rnd.choice(parents)
                if rnd.random() < 0.3:
                    x = crossover(x, rnd.choice(parents), rnd)
                kids.append(mutate(x, rnd, rnd.randint(1, 3)))
            scored = sorted(scored + list(zip(score(kids, train_add, train_main), kids)),
                            key=lambda x: -x[0][0])[:a.population]
            (s, d), best = scored[0]
            say(f"поколение {gen:3d}: лучший {s:.3f}, сложение до 10 {d.get('до 10, свежие дни', 0):.2f}, "
                f"прежние {d.get('прежние задачи', 0):.2f}  [{time.time() - start:.0f} с]")
            json.dump({"generation": gen, "score": s, "detail": d, "params": best,
                       "changes": changes(best, base)},
                      open(os.path.join(a.out, "checkpoint.json"), "w"), ensure_ascii=False, indent=1)
            last = time.time() - t0

        say("\nпроверка (выбор победителя), запуски, не участвовавшие в отборе:")
        cands = [("стартовый", base)] + [(f"лучший №{i + 1}", p) for i, (_, p) in enumerate(scored[:5])]
        vals = score([p for _, p in cands], val_add, val_main)
        for (name, _), (s, d) in zip(cands, vals):
            say(f"  {name:12s} {s:.3f}  сложение до 10 {d.get('до 10, свежие дни', 0):.2f}, "
                f"прежние {d.get('прежние задачи', 0):.2f}")
        k = max(range(1, len(cands)), key=lambda i: vals[i][0])
        winner = cands[k][1]
        say(f"\nтест (один раз, все экзамены), запуски, не участвовавшие ни в чём:")
        tests = score([base, winner], test_add, test_main, full=True)
        for name, (s, d) in zip(("стартовый", cands[k][0]), tests):
            say(f"  {name}: " + ", ".join(f"{kk} {v:.2f}" for kk, v in d.items()))
        json.dump({"name": cands[k][0], "val": vals[k], "test": tests[1], "test_start": tests[0],
                   "params": winner, "changes": changes(winner, base)},
                  open(os.path.join(a.out, "best.json"), "w"), ensure_ascii=False, indent=1)
        say("изменения победителя:")
        for line in changes(winner, base):
            say("  " + line)
    say(f"всего {time.time() - start:.0f} с")


if __name__ == "__main__":
    main()
