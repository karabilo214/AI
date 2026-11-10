"""Умножение после сложения и вычитания (мир candy_world/arith/arith.py), всё прежнее остаётся.

День умножения: N карточек подряд («каждому по N», N от 2 до 5), гости
группами, вопрос ASK «сколько принести?» = N * гостей. Сочетания hidden(N, g)
в обучении не показываются (как arith.py).
Мозг = мозг вычитания (subtraction.py) с регистром в накопителе: карточки
кладутся в регистр, гость может прибавлять его содержимое (Accumulator.py).
Чему учится: накопитель — что класть в регистр, что прибавлять порцией, какое
событие прибавляет регистр (по ошибке после праздника); мозг — правила ответа
(они те же, что для «сколько принести», выучены на прежних ступенях).

Мозг после вычитания сохраняется (human_brain/brains/, brain_store.py) и при
следующем запуске загружается, а не учится заново.
Порядок: сложение (300 дней) -> вычитание вперемешку со сложением (2000) ->
умножение, где каждый второй день — прежняя задача (сложение, «не хватило»,
«осталось» по очереди): выученное прежде не забывать.
Экзамен: всё из subtraction.py плюс умножение: свежие (N 2-5, гостей до 10),
скрытые сочетания, гостей 11-20, 21-50, 101-200, карточка 6-9 (не видел).
--hand: ручная сборка.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.multiplication [--seed 0] [--mul-days 2000] [--hand]
"""
import argparse
import copy
import random

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.core import brain_store

CARD_TOP = 5


def make_mul_day(rng, top=10, low=0, card=(2, CARD_TOP), want_hidden=False):
    for _ in range(10000):
        n, g = rng.randint(*card), rng.randint(max(2, low), top)
        if sb.hidden(n, g) != want_hidden:
            continue
        events = ["CARD"] * n + ad.groups(g, rng, at_least=1) + ["ASK"]
        return events, n * g, n * g
    raise RuntimeError("нет дня")


def old_day(rng, k):
    """Прежняя задача: по очереди сложение, «не хватило», «осталось»."""
    kind = k % 3
    if kind == 0:
        return sb.make_add_day(rng)
    return sb.make_sub_day(rng, "ASK" if kind == 1 else "ASK_LEFT")


def train(b, rng, add_days=300, sub_days=2000, mul_days=2000):
    sb.train(b, rng, add_days, sub_days)
    for k in range(mul_days):
        ev, ans, truth = make_mul_day(rng) if k % 2 == 0 else old_day(rng, k // 2)
        ad.day(b, ev, ans, learn=True, truth=truth)


def exam_sets(seed, n=30):
    sets = sb.exam_sets(seed, n)
    rng = random.Random(2000 + seed)
    sets["умножение до 10"] = [make_mul_day(rng) for _ in range(n)]
    sets["умножение скрытые"] = [make_mul_day(rng, want_hidden=True) for _ in range(n)]
    sets["умножение гостей 11-20"] = [make_mul_day(rng, 20, 11) for _ in range(n)]
    sets["умножение гостей 21-50"] = [make_mul_day(rng, 50, 21) for _ in range(n)]
    sets["умножение гостей 101-200"] = [make_mul_day(rng, 200, 101) for _ in range(5)]
    sets["умножение карточка 6-9"] = [make_mul_day(rng, card=(6, 9)) for _ in range(n)]
    return sets


def make_brain(p, seed, hand=False):
    return sb.make_brain(dict(p, register=True), seed, hand=hand)


def enable_register(b):
    """Мозгу, сохранённому после вычитания, дать регистр (m и r с 0.1, как у нового)."""
    acc = b.accumulator
    if not acc.register:
        acc.register = True
        acc.m = {k: 0.1 for k in acc.inputs}
        acc.r = {k: 0.1 for k in acc.inputs}


def subtraction_brain(p, seed, sub_days=2000):
    """Мозг после сложения и вычитания: из сохранения, а если его нет — обучить и сохранить."""
    path = brain_store.path_for(f"sub_seed{seed}_d{sub_days}")
    if brain_store.exists(path):
        b, rng, _ = brain_store.load(path)
        return b, rng
    tasks.apply(p)
    b = sb.make_brain(p, seed)
    rng = random.Random(seed)
    sb.train(b, rng, 300, sub_days)
    brain_store.save(b, rng, {"ступень": "вычитание", "seed": seed, "add_days": 300,
                              "sub_days": sub_days}, path)
    return b, rng


def run(p, seed, sub_days=2000, mul_days=2000, save=True):
    """Ступень умножения поверх сохранённого мозга вычитания."""
    b, rng = subtraction_brain(p, seed, sub_days)
    enable_register(b)
    for k in range(mul_days):
        ev, ans, truth = make_mul_day(rng) if k % 2 == 0 else old_day(rng, k // 2)
        ad.day(b, ev, ans, learn=True, truth=truth)
    if save:
        brain_store.save(b, rng, {"ступень": "умножение", "seed": seed, "sub_days": sub_days,
                                  "mul_days": mul_days},
                         brain_store.path_for(f"mul_seed{seed}_d{sub_days}_{mul_days}"))
    return {k: sb.exam(b, v) / len(v) for k, v in exam_sets(seed).items()}, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-days", type=int, default=2000)
    ap.add_argument("--mul-days", type=int, default=2000)
    ap.add_argument("--hand", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    if a.hand:
        b = make_brain(p, a.seed, hand=True)
    else:
        _, b = run(p, a.seed, a.sub_days, a.mul_days)
        acc = b.accumulator
        print("накопитель: w", {k: round(v, 2) for k, v in acc.w.items()},
              "m", {k: round(v, 2) for k, v in acc.m.items()},
              "r", {k: round(v, 2) for k, v in acc.r.items()})
    print("ЭКЗАМЕН (мозг / ленивая «всегда 0»):")
    for name, days in exam_sets(a.seed).items():
        lazy = sum(ans == 0 for _, ans, _ in days)
        print(f"  {name:26s} {sb.exam(b, days)}/{len(days)}   ({lazy}/{len(days)})", flush=True)


if __name__ == "__main__":
    main()
