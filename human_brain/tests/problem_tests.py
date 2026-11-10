"""Проверки под конкретные проблемы (не входят в оценку эволюции).

1. Ложная память: 5 раз за A идёт B, потом 40 раз B за A лишь в 20%.
   Хорошо: в конце A одна НЕ зажигает B.
2. Заслуга: задача S1 -> L и посторонний нейрон D (S1 -> D), который на
   действие не влияет. Хорошо: связь S1 -> D от награды не крепнет
   (остаётся только рост от обычного правила «сработали друг за другом»).
3. Миелин: Y -> T приходит после срабатывания T (T зажигает X).
   Хорошо: связь Y -> T ускоряется, главная X -> T не замедляется.
4. Задачи вне оценки: три задачи подряд (и зеркально), переучивание
   S1: L -> R за 60 попыток, задача A при 10% перепутанных наград.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.tests.problem_tests evolve_results_new_on/best.json evolve_results_new_off/best.json
"""
import os as _os
import sys as _sys
import copy
import json
import random
import sys

import numpy as np

from human_brain.stages import tasks
from human_brain.core import fast
from human_brain.core.Brain import Brain


def load(path):
    if path == "default":
        return "текущие значения", dict(tasks.PARAMS_DEFAULT)
    return path, {**tasks.PARAMS_DEFAULT, **json.load(open(path))["params"]}


def false_memory(p, runs=10):
    bad = 0
    for seed in range(runs):
        rnd = random.Random(seed)
        b = Brain()
        b.add_neuron("A")
        b.add_neuron("B")
        tasks.attach(b, p, seed, curiosity=False, neurons=False)
        for k in range(45):
            b.stimulate("A", 20.0)
            follow = k < 5 or rnd.random() < 0.2
            for s in range(30):
                if s == 2 and follow:
                    b.stimulate("B", 20.0)
                b.step()
        t = copy.deepcopy(b)
        before = t.neurons["B"].spikes
        t.stimulate("A", 20.0)
        tasks.rest(t, 10)
        bad += t.neurons["B"].spikes > before
    return bad


def distractor(p, runs=10):
    dw, ok = [], 0
    for seed in range(runs):
        b = tasks.make_reward_brain(p, seed)
        b.add_neuron("D")
        cd = b.connect("S1", "D", weight=p["init_weight"], myelin_thickness=1.0,
                       plasticity=p["init_plasticity"])
        w0 = cd.synapse.weight
        tasks.learn(b, ("S1",), "L")
        ok += tasks.exam(b, ("S1",)) == "L"
        dw.append(cd.synapse.weight - w0)
    return ok, float(np.mean(dw))


def late_myelin(p):
    b = Brain()
    for n in "XYT":
        b.add_neuron(n)
    cx = b.connect("X", "T", weight=18.0, myelin_thickness=1.0)
    cy = b.connect("Y", "T", weight=5.0, myelin_thickness=0.5)
    for _ in range(300):
        b.stimulate("X", 20.0)
        b.stimulate("Y", 20.0)
        tasks.rest(b, 20)
    return cy.myelin.delay(), cx.myelin.delay()


def unseen(p, runs=50):
    P = fast.params_array(p)
    out = {}
    for name, parts in {
        "три задачи": [(("S1",), "L"), (("S2",), "R"), (("S1", "S2"), "L")],
        "три зеркально": [(("S1",), "R"), (("S2",), "L"), (("S1", "S2"), "R")],
    }.items():
        inputs, good = fast.task_arrays(parts)
        out[name] = sum(bool(fast.run_reward_task(P, fast.NC_TASKS, fast.CC_TASKS, s, inputs, good)[0])
                        for s in range(3000, 3000 + runs))
    relearned = 0
    for s in range(20):
        b = tasks.make_reward_brain(p, s)
        tasks.learn(b, ("S1",), "L")
        tasks.learn(b, ("S1",), "R")
        relearned += tasks.exam(b, ("S1",)) == "R"
    out["переучился L->R (из 20)"] = relearned
    ok = 0
    for s in range(20):
        rnd = random.Random(s)
        b = tasks.make_reward_brain(p, s)
        for inputs, g in tasks.TASKS["A"]:
            for _ in range(60):
                o = tasks.trial(b, inputs)
                r = +1 if o == g else -1 if o in ("L", "R", "оба") else 0
                if rnd.random() < 0.1:
                    r = -r
                b.reward(r)
                tasks.rest(b)
        ok += all(tasks.exam(b, i) == g for i, g in tasks.TASKS["A"])
    out["A при 10% перепутанных (из 20)"] = ok
    return out


def main(paths):
    for path in paths:
        name, p = load(path)
        tasks.apply(p)
        print(f"=== {name}  (гипотезы {p['hyp_on']}, заслуга {p['credit_on']}, "
              f"миелин по времени {p['myelin_timing']}, знакомость {p['fam_on']})")
        print(f"  1. ложная память A->B: в {false_memory(p)}/10 (меньше = лучше)")
        ok, dw = distractor(p)
        print(f"  2. посторонний D: задача верна {ok}/10, сила S1->D изменилась на {dw:+.2f}")
        dy, dx = late_myelin(p)
        print(f"  3. опаздывающая Y->T: задержка 3 -> {dy}; главная X->T: 1 -> {dx}")
        for k, v in unseen(p).items():
            print(f"  4. {k}: {v}" + ("/50" if "три" in k else ""))


if __name__ == "__main__":
    main(sys.argv[1:] or ["default"])
