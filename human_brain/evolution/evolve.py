"""Эволюция параметров мозга: перебор с отбором лучших.

Как работает:
    1. Популяция: наборы параметров. Первый = текущие значения (tasks.PARAMS_DEFAULT),
       остальные = его случайные изменения.
    2. Каждое поколение: дети = копия случайного родителя (иногда смесь двух),
       у которой случайно изменены 1..3 параметра в пределах tasks.SPACE.
    3. Все дети оцениваются на запусках для отбора (tasks.evaluate);
       в следующее поколение проходят лучшие из родителей и детей.
    4. В конце 5 лучших и текущие значения проверяются на запусках проверки
       (val), которых эволюция не видела; по ним выбирается победитель.
       Раз этот выбор тоже отбор, победитель и текущие значения затем один
       раз оцениваются на третьем наборе запусков (test), который не
       участвовал ни в чём: это и есть независимая оценка.
Время (--minutes) ограничивает цикл поколений: новое поколение не
начинается, если по длительности прошлого оно не успеет. Начальная оценка
и итоговые проверки идут сверх этого времени.

Оценка по умолчанию быстрая (fast.py, весь цикл в Numba; без шума совпадает
с обычной версией до бита, см. check_fast.py). --slow: обычная версия.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.evolution.evolve --generations 100 --minutes 20
Результаты: human_brain/data/evolve_results/log.txt, history.json, best.json, validation.json
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # human_brain


FIXED = set()   # параметры, которые эволюция не меняет (--fix)


def mutate(p, rnd, n_genes):
    child = dict(p)
    genes = [g for g in tasks.SPACE if g not in FIXED]
    for name in rnd.sample(genes, n_genes):
        kind, *limits = tasks.SPACE[name]
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
        else:  # lin
            lo, hi = limits
            v = min(hi, max(lo, v + rnd.gauss(0, 0.15 * (hi - lo))))
        child[name] = v
    return child


def crossover(a, b, rnd):
    return {k: (a[k] if rnd.random() < 0.5 else b[k]) for k in a}


def _evaluate(args):
    p, seeds, use_fast = args
    try:
        return fast.evaluate(p, seeds) if use_fast else tasks.evaluate(p, seeds)
    except Exception as e:   # набор, ломающий мозг, получает худшую оценку
        return -1.0, {"error": f"{type(e).__name__}: {e}"}


def changes(p):
    """Чем набор отличается от текущих значений (словами)."""
    out = []
    for k, v0 in tasks.PARAMS_DEFAULT.items():
        v = p[k]
        if v != v0:
            out.append(f"{k}: {v0} -> {round(v, 5) if isinstance(v, float) else v}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--children", type=int, default=32)
    ap.add_argument("--train-seeds", type=int, default=12)
    ap.add_argument("--val-seeds", type=int, default=50)
    ap.add_argument("--test-seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--minutes", type=float, default=20.0,
                    help="время на цикл поколений (начальная оценка и итоговые проверки сверх него)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix", default="",
                    help="закрепить параметры, напр. hyp_on=1,credit_on=1 (эволюция их не меняет)")
    ap.add_argument("--start", default="",
                    help="json с набором (поле params), с которого начать вместо текущих значений")
    ap.add_argument("--slow", action="store_true",
                    help="оценивать обычной версией (Brain), а не быстрой (fast.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "evolve_results"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.txt"), "w")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    rnd = random.Random(args.seed)
    train = list(range(args.train_seeds))
    val = list(range(100000, 100000 + args.val_seeds))    # эволюция их не видит
    test = list(range(200000, 200000 + args.test_seeds))  # не участвуют ни в каком выборе
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise SystemExit("запуски отбора, проверки и теста пересекаются")
    start = time.time()
    history = []

    with Pool(args.workers) as pool:
        def score_all(sets, seeds):
            return pool.map(_evaluate, [(p, seeds, not args.slow) for p in sets])

        base = dict(tasks.PARAMS_DEFAULT)
        if args.start:
            base.update(json.load(open(args.start))["params"])
        for item in filter(None, args.fix.split(",")):
            key, value = item.split("=")
            default = tasks.PARAMS_DEFAULT[key]
            base[key] = (value in ("1", "True", "true")) if isinstance(default, bool) else type(default)(value)
            FIXED.add(key)
        if FIXED:
            say("закреплены: " + ", ".join(f"{k}={base[k]}" for k in sorted(FIXED)))
        population = [base] + [mutate(base, rnd, rnd.randint(1, 3))
                               for _ in range(args.population - 1)]
        scored = list(zip(score_all(population, train), population))
        scored.sort(key=lambda x: -x[0][0])
        say(f"стартовый набор на отборе: {scored[[p for _, p in scored].index(base)][0][0]:.3f}")

        gen_start = time.time()
        last_gen = 0.0
        for gen in range(1, args.generations + 1):
            if time.time() - gen_start + last_gen > args.minutes * 60:
                say(f"остановлено по времени после {gen - 1} поколений")
                break
            t_gen = time.time()
            parents = [p for _, p in scored]
            kids = []
            for _ in range(args.children):
                a = rnd.choice(parents)
                if rnd.random() < 0.3:
                    a = crossover(a, rnd.choice(parents), rnd)
                kids.append(mutate(a, rnd, rnd.randint(1, 3)))
            scored = scored + list(zip(score_all(kids, train), kids))
            scored.sort(key=lambda x: -x[0][0])
            scored = scored[:args.population]
            (best_score, best_detail), best = scored[0]
            history.append({"generation": gen, "best": best_score, "detail": best_detail,
                            "mean": sum(s for (s, _), _ in scored) / len(scored)})
            json.dump({"generation": gen, "score": best_score, "detail": best_detail,
                       "params": best, "changes": changes(best)},
                      open(os.path.join(args.out, "checkpoint.json"), "w"),
                      ensure_ascii=False, indent=1)
            say(f"поколение {gen:3d}: лучший {best_score:.3f} "
                + " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in best_detail.items()
                           if k in ("reward", "memory", "growth", "hits", "before"))
                + f"  [{time.time() - start:.0f} с]")
            last_gen = time.time() - t_gen

        json.dump(history, open(os.path.join(args.out, "history.json"), "w"),
                  ensure_ascii=False, indent=1)

        # Проверка на новых запусках: текущие значения и 5 лучших
        say(f"\nпроверка на {len(val)} новых запусках (эволюция их не видела):")
        candidates = [("стартовый набор", base)] + [(f"лучший №{i + 1}", p)
                                                      for i, (_, p) in enumerate(scored[:5])]
        results = score_all([p for _, p in candidates], val)
        report = []
        for (name, p), (score, detail), in zip(candidates, results):
            say(f" {name:16s}: {score:.3f} " + " ".join(
                f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in detail.items()))
            report.append({"name": name, "score": score, "detail": detail, "params": p,
                           "changes": changes(p)})
        json.dump(report, open(os.path.join(args.out, "validation.json"), "w"),
                  ensure_ascii=False, indent=1)
        best_val = max(report[1:], key=lambda r: r["score"])
        # независимая оценка: победитель выбран по val, поэтому его оценка на val
        # смещена вверх; test не участвовал в выборе
        (t_base, d_base), (t_best, d_best) = score_all([base, best_val["params"]], test)
        best_val["val_score"] = best_val["score"]
        best_val["test_score"], best_val["test_detail"] = t_best, d_best
        best_val["test_start_score"], best_val["test_start_detail"] = t_base, d_base
        json.dump(best_val, open(os.path.join(args.out, "best.json"), "w"),
                  ensure_ascii=False, indent=1)
        say(f"\nтест на {len(test)} запусках, не участвовавших в выборе: "
            f"стартовый набор {t_base:.3f}, победитель {t_best:.3f}")
        say(f"\nлучший на проверке: {best_val['name']} ({best_val['score']:.3f}), изменения:")
        for line in best_val["changes"]:
            say("  " + line)
    say(f"всего {time.time() - start:.0f} с")


if __name__ == "__main__":
    main()
