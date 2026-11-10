"""Деление после сложения, вычитания и умножения (мир candy_world/arith/arith.py).

День деления: у меня c конфет (CANDY по одной), пришли g гостей (группами,
g от 2 до 10), вопрос ASK_EACH «сколько каждому?» = c // g (остаток остаётся).
Сочетания hidden(c, g) в обучении не показываются.

Мозг берётся из сохранения после умножения (brain_store, multiplication.py) и
получает новый вопрос: входы ASK_EACH и M_EACH (защёлка), связи к действиям
как у прежних датчиков, место ещё под 12 нейронов-сочетаний. Новых устройств
нет: накопитель и регистр прежние. По расчёту накопитель деление нацело не
выражает (ответ = уровень после событий = суммы и произведения счётчиков), так
что этот опыт — проверка расчёта.

Порядок: каждый второй день деление, остальные — прежние задачи по очереди
(сложение, «не хватило», «осталось», умножение): выученное не забывать.
Экзамен: всё прежнее (multiplication.exam_sets) плюс деление: свежие до 10,
скрытые, конфет 11-20, 21-50 (гостей до 10). Сохранения мозгов не меняются
(учится копия).

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.division [--seed 800] [--days 2000]
"""
import argparse
import random

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.stages import multiplication as mu
from human_brain.core import brain_store
from human_brain.parts.Units import Units
from human_brain.parts.NeuralChoice import NeuralChoice


def make_div_day(rng, top=10, low=0, want_hidden=False):
    for _ in range(10000):
        c, g = rng.randint(low, top), rng.randint(2, 10)
        if sb.hidden(c, g) != want_hidden:
            continue
        events = ["CANDY"] * c + ad.groups(g, rng, at_least=1) + ["ASK_EACH"]
        return events, c // g, c // g
    raise RuntimeError("нет дня")


def add_question(b, p):
    """Новый вопрос «сколько каждому?»: входы, защёлка, связи как у прежних датчиков."""
    w = p.get("sensor_weight", p["action_min"])
    for n in ("ASK_EACH", "M_EACH"):
        b.add_neuron(n, region="входы")
        for o in sb.ACTIONS:
            b.connect(n, o, weight=w, myelin_thickness=1.0, fixed=True)
            b.connect(n, o, weight=0.0, inhibitory=True, myelin_thickness=1.0, plasticity=0.0)
    acc = b.accumulator
    acc.inputs.append("ASK_EACH")
    for d, v in ((acc.w, 0.0), (acc.m, 0.1 if acc.register else 0.0),
                 (acc.r, 0.1 if acc.register else 0.0), (acc.counts, 0)):
        d["ASK_EACH"] = v
    b.latch.inputs.append("ASK_EACH")
    b.latch.outputs.append("M_EACH")
    b.growth.max_neurons += 12
    if p.get("choice", "rule") == "neural":
        # выбор делает сеть: все заготовки сразу, сочетания «выход + вопрос»
        NeuralChoice(b, inputs=acc.inputs, own=["ASK", "ASK_LEFT"])
    elif p.get("units", True):
        # найм заготовок: прежний накопитель владеет прежними вопросами
        Units(b, inputs=acc.inputs, own=["ASK", "ASK_LEFT"], seed=p.get("units_seed", 0))


def old_day(rng, k):
    kind = k % 4
    if kind == 3:
        return mu.make_mul_day(rng)
    return mu.old_day(rng, kind)


def exam_sets(seed, n=30):
    sets = mu.exam_sets(seed, n)
    rng = random.Random(3000 + seed)
    sets["деление до 10"] = [make_div_day(rng) for _ in range(n)]
    sets["деление скрытые"] = [make_div_day(rng, want_hidden=True) for _ in range(n)]
    sets["деление конфет 11-20"] = [make_div_day(rng, 20, 11) for _ in range(n)]
    sets["деление конфет 21-50"] = [make_div_day(rng, 50, 21) for _ in range(n)]
    return sets


def run(p, seed, days=2000, save=False):
    path = brain_store.path_for(f"mul_seed{seed}_d2000_2000")
    if not brain_store.exists(path):
        mu.run(p, seed, 2000, 2000)
    b, rng, _ = brain_store.load(path)
    add_question(b, dict(p, units_seed=seed))
    for k in range(days):
        ev, ans, truth = make_div_day(rng) if k % 2 == 0 else old_day(rng, k // 2)
        ad.day(b, ev, ans, learn=True, truth=truth)
    if save:
        brain_store.save(b, rng, {"ступень": "деление", "seed": seed, "days": days,
                                  "choice": p.get("choice", "rule")},
                         brain_store.path_for(div_name(seed, days, p.get("choice", "rule"))))
    return {k: sb.exam(b, v) / len(v) for k, v in exam_sets(seed).items()}, b


def div_name(seed, days=3000, choice="neural"):
    return f"div_{choice}_seed{seed}_d{days}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=800)
    ap.add_argument("--days", type=int, default=2000)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    res, b = run(p, a.seed, a.days)
    acc = b.accumulator
    print("накопитель: w", {k: round(v, 2) for k, v in acc.w.items()},
          "m", {k: round(v, 2) for k, v in acc.m.items()},
          "r", {k: round(v, 2) for k, v in acc.r.items()})
    for k, v in res.items():
        print(f"  {k:26s} {v:.2f}")


if __name__ == "__main__":
    main()
