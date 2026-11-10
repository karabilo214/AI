"""Проверки областей мозга (исходные значения tasks.PARAMS_DEFAULT, обычная версия).

1. Ответ с задержкой: показали S1 или S2, пауза delay шагов, сигнал «отвечай»
   (GO); верно: S1 -> L, S2 -> R. Стимул заряжает свой нейрон памяти (M1/M2)
   ниже порога; GO добавляет обоим немного; срабатывает тот, кто ещё держит
   заряд; дальше M -> L/R учится по награде. Устройство сети одинаковое,
   отличается только то, в какой области нейроны памяти:
   «память» (заряд почти не стекает) или обычная.
   Заведомо решаемая проверка: обычная сеть с паузой 1 шаг.
   Знакомость выключена в обоих вариантах (иначе стимул приедается).
2. Быстрая память: один показ пары X -> F. Через 20 шагов X сам зажигает F?
   Через 100 и 300? Сравнение с обычной областью (1 и 10 показов).

Запуск из папки human_brain: ../.venv/bin/python tests/region_tests.py
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # модули мозга
import copy

import numpy as np

import tasks
from Brain import Brain
from Regions import add_standard_regions


def params(**kw):
    p = dict(tasks.PARAMS_DEFAULT)
    p.update(fam_on=False, **kw)
    tasks.apply(p)
    return p


# ---------- 1. ответ с задержкой ----------

def make_delay_brain(p, seed, regions):
    """regions: False (единая сеть), True (нейроны памяти в области «память»)
    или имя области для нейронов памяти."""
    b = Brain()
    add_standard_regions(b)
    b.add_region("жёсткая, обычное стекание", learn=0.0)
    mem = regions if isinstance(regions, str) else ("память" if regions else None)
    for n in ("S1", "S2", "GO"):
        b.add_neuron(n)
    for n in ("M1", "M2"):
        b.add_neuron(n, region=mem)
    for n in ("L", "R"):
        b.add_neuron(n, explores=True, region="действия" if regions is True else None)
    b.connect("S1", "M1", weight=6.0, myelin_thickness=1.0)
    b.connect("S2", "M2", weight=6.0, myelin_thickness=1.0)
    b.connect("GO", "M1", weight=4.0, myelin_thickness=1.0)
    b.connect("GO", "M2", weight=4.0, myelin_thickness=1.0)
    for m in ("M1", "M2"):
        for o in ("L", "R"):
            b.connect(m, o, weight=p["init_weight"], myelin_thickness=1.0,
                      plasticity=p["init_plasticity"])
    b.connect("L", "R", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("R", "L", weight=p["inhibit_weight"], inhibitory=True)
    # (стирать память ответом не нужно: сработавший нейрон памяти сбрасывается сам;
    #  торможение медленного нейрона держалось бы десятки шагов и мешало)
    tasks.attach(b, p, seed, curiosity=True, neurons=False)
    return b


def delay_trial(b, stim, delay):
    b.stimulate(stim, 20.0)
    for _ in range(delay):
        b.step()
    b.stimulate("GO", 20.0)
    for _ in range(7):
        f = b.step()
        if "L" in f or "R" in f:
            return "L" if "R" not in f else "R" if "L" not in f else "оба"
    return "никто"


def delay_task(p, seed, regions, delay, n=100):
    b = make_delay_brain(p, seed, regions)
    good = {"S1": "L", "S2": "R"}
    hits = 0
    for k in range(n):
        s = ("S1", "S2")[k % 2]
        out = delay_trial(b, s, delay)
        hits += out == good[s]
        b.reward(+1 if out == good[s] else -1 if out in ("L", "R", "оба") else 0)
        # пауза 300 шагов: медленный нейрон памяти должен успеть «остыть»,
        # иначе остаток заряда от GO заставит его сработать раньше времени
        tasks.rest(b, 300)
    answers = {}
    for s in ("S1", "S2"):
        t = copy.deepcopy(b)
        t.curiosity = None
        answers[s] = delay_trial(t, s, delay)
    ok = all(answers[s] == good[s] for s in good)
    return ok, hits / n, answers


# ---------- 2. быстрая память ----------

def fast_memory(p, region, lessons):
    b = Brain()
    add_standard_regions(b)
    b.add_neuron("X")
    b.add_neuron("F", region=region)
    b.connect("X", "F", weight=2.0, myelin_thickness=1.0, plasticity=p["init_plasticity"])
    for _ in range(lessons):          # урок: X, и сразу за ним F
        b.stimulate("X", 20.0)
        b.step()
        b.stimulate("F", 20.0)
        tasks.rest(b, 29)
    out = {}
    for wait in (20, 100, 300):
        t = copy.deepcopy(b)
        tasks.rest(t, wait)
        before = t.neurons["F"].spikes
        t.stimulate("X", 20.0)
        tasks.rest(t, 5)
        out[wait] = t.neurons["F"].spikes > before
    return out, b.connections[0].synapse.weight


def main(runs=20):
    seeds = range(8000, 8000 + runs)
    print("1. ответ с задержкой (100 попыток, по очереди S1 и S2):")
    rigid = "жёсткая, обычное стекание"
    for regions, delay, label in ((False, 1, "единая сеть, пауза 1"),
                                  (False, 10, "единая сеть, пауза 10"),
                                  (rigid, 1, "жёсткие связи, стекание 0.85, пауза 1"),
                                  (rigid, 10, "жёсткие связи, стекание 0.85, пауза 10"),
                                  (True, 10, "область «память» (0.99), пауза 10"),
                                  (True, 20, "область «память» (0.99), пауза 20")):
        p = params()
        res = [delay_task(p, s, regions, delay) for s in seeds]
        print(f"   {label:42s}: оба ответа верны {sum(r[0] for r in res)}/{runs}, "
              f"верных при обучении {np.mean([r[1] for r in res]):.2f}")
    print("2. быстрая память: X сам зажигает F через 20 / 100 / 300 шагов")
    p = params()
    for region, lessons, label in (("быстрая память", 1, "быстрая память, 1 показ"),
                                   (None, 1, "обычная область, 1 показ"),
                                   (None, 10, "обычная область, 10 показов"),
                                   ("быстрая память", 10, "быстрая память, 10 показов")):
        out, w = fast_memory(p, region, lessons)
        print(f"   {label:28s}: {out[20]} / {out[100]} / {out[300]}  (сила X->F после уроков {w:.1f})")


if __name__ == "__main__":
    main()
