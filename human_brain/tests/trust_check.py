"""Проверка ворот доверия (parts/Trust.py) на ступени деления, один мозг.

Мозг берётся из сохранения после умножения (seed 2000) и учит деление с
выбором устройств по доверию вместо владения вопросами и закрепления.
    --hand : доверие задано руками (заведомо решаемая проверка): на дне с
             вопросом «сколько каждому» верна заготовка «по регистру, регистр
             = гости», иначе накопитель.
Печатает экзамен (прежнее + деление) и доверие устройств в типичных днях.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.tests.trust_check [--hand] [--days 3000] [--seed 2000]
"""
import argparse
import random
import time

from human_brain.stages import tasks, division as dv, subtraction as sb, multiplication as mu
from human_brain.parts.Units import VARIANTS


def right_name(inputs):
    k = [(v, e) for v in VARIANTS for e in inputs].index(("по регистру", "GUEST"))
    return f"U{k + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--days", type=int, default=3000)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--wrong", default=None, help="с --hand: доверять этой заготовке (контроль)")
    ap.add_argument("--const", type=float, default=None, help="абляция: доверие всех устройств = это число")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--every", type=int, default=500)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False, choice="trust", trust_lr=a.lr)
    tasks.apply(p)
    from human_brain.core import brain_store
    b, rng, _ = brain_store.load(brain_store.path_for(f"mul_seed{a.seed}_d2000_2000"))
    if a.const is not None:
        p["trust_hand"] = lambda name, x: a.const
    elif a.hand:
        good = right_name(b.accumulator.inputs + ["ASK_EACH"])

        def hand(name, x):
            if "ASK_EACH" in x:
                return 1.0 if name == (a.wrong or good) else 0.0
            return 1.0 if name == "ACC" else 0.0
        p["trust_hand"] = hand
    dv.add_question(b, p)
    tr = b.units
    print("верная заготовка:", right_name(tr.inputs), flush=True)
    sets = dv.exam_sets(a.seed)
    quick = {k: sets[k] for k in ("сложение до 10", "не хватило: до 10", "осталось: до 10",
                                  "умножение до 10", "деление до 10") if k in sets}
    t0 = time.time()
    for k in range(a.days):
        ev, ans, truth = dv.make_div_day(rng) if k % 2 == 0 else dv.old_day(rng, k // 2)
        b_day = __import__("human_brain.stages.addition", fromlist=["day"]).day
        b_day(b, ev, ans, learn=True, truth=truth)
        if (k + 1) % a.every == 0:
            q = {n: round(sb.exam(b, v) / len(v), 2) for n, v in quick.items()}
            print(f"день {k + 1} ({time.time() - t0:.0f} с): {q}", flush=True)
    print("\nэкзамен:")
    for n, v in sets.items():
        print(f"  {n:28s} {sb.exam(b, v) / len(v):.2f}")
    # доверие в типичных днях: ситуация = что сработало до первого «отвечай»
    print("\nдоверие (накопитель и три лучших заготовки):")
    r = random.Random(1)
    for label, (ev, ans, truth) in (("сложение", sb.make_add_day(r)),
                                    ("не хватило", sb.make_sub_day(r, "ASK")),
                                    ("осталось", sb.make_sub_day(r, "ASK_LEFT")),
                                    ("умножение", mu.make_mul_day(r)),
                                    ("деление", dv.make_div_day(r))):
        import copy
        t = copy.deepcopy(b)
        t.curiosity = None
        from human_brain.stages import addition as ad
        ad.day(t, ev, ans, learn=False, truth=truth)
        x = t.units._x or frozenset()
        g = tr.report(x) if tr.hand is None else {n: t.units.trust(n) for n in tr.gates}
        best = sorted((n for n in g if n != "ACC"), key=lambda n: -g[n])[:3]
        print(f"  {label:11s} ACC {g['ACC']:.2f}  " + "  ".join(f"{n} {g[n]:.2f}" for n in best))


if __name__ == "__main__":
    main()
