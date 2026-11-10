"""Проверки по отчёту docs/CODE_AUDIT_REPORT.md: каждая — короткий сценарий
воспроизведения дефекта, который после исправления должен давать верный ответ.

Номера совпадают с отчётом. При любом провале код выхода 1.
Запуск из папки human_brain: ../.venv/bin/python tests/check_audit.py
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # модули мозга
import copy
import os
import sys
import tempfile

import numpy as np

import tasks
from Brain import Brain
from Growth import Growth
from Recruit import Recruit
from Hearing import Hearing
from Association import Association
from Episodes import Episodes, Episode

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok ' if ok else 'ПРОВАЛ'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def run(b, schedule, steps):
    """schedule: {шаг: [имена]} — кого толкнуть; возвращает список сработавших по шагам."""
    out = []
    for t in range(steps):
        for n in schedule.get(t, ()):
            b.stimulate(n, 20.0)
        out.append(b.step())
    return out


# ---------- №1: нейрон-сочетание ----------

def pair_brain():
    b = Brain()
    b.add_neuron("X")
    b.add_neuron("Y")
    g = Growth(b)
    g._grow_neuron(("X", "Y"), 2)
    b.growth = None
    return b


def test_pair():
    fired = lambda out: any("X+Y" in f for f in out)
    check("№1 сочетание: только X не зажигает", not fired(run(pair_brain(), {0: ["X"]}, 4)))
    check("№1 сочетание: X и Y вместе зажигают", fired(run(pair_brain(), {0: ["X", "Y"]}, 4)))
    check("№1 сочетание: X на каждом 2-м шаге не копится",
          not fired(run(pair_brain(), {0: ["X"], 2: ["X"], 4: ["X"], 6: ["X"]}, 9)))
    check("№1 сочетание: X, через шаг Y — не вместе",
          not fired(run(pair_brain(), {0: ["X"], 1: ["Y"]}, 5)))
    # входы не учатся: после многих совместных показов сила та же
    b = pair_brain()
    w0 = [c.synapse.weight for c in b.connections if c.target == "X+Y"]
    run(b, {t: ["X", "Y"] for t in range(0, 200, 5)}, 200)
    w1 = [c.synapse.weight for c in b.connections if c.target == "X+Y"]
    check("№1 сочетание: входы закреплены", w0 == w1, f"{w0} -> {w1}")
    # рост не проводит связи в детектор: Z всегда за шаг до X и Y
    b = Brain()
    for n in ("X", "Y", "Z"):
        b.add_neuron(n)
    # explain_weight высокий: срабатывание детектора «не объяснено» (как в
    # subtraction.py, где входы детектора 3.0 + 3.0 < 6.15)
    g = Growth(b, hypotheses=False, explain_weight=20.0)
    g._grow_neuron(("X", "Y"), 2)
    run(b, {t + d: (["Z"] if d == 0 else ["X", "Y"]) for t in range(0, 400, 10) for d in (0, 1)}, 400)
    into = [c.source for c in b.connections if c.target == "X+Y" and not b.fixed[c.index]]
    check("сочетание: рост не проводит связи в детектор", not into, f"выросли входы {into}")


# ---------- №2: правила у каждого мозга ----------

def test_rules():
    p = dict(tasks.PARAMS_DEFAULT)
    tasks.apply({**p, "reward_rate": 1.0})
    b = Brain()
    b.add_neuron("A")
    b.add_neuron("B")
    c = b.connect("A", "B", weight=5.0)
    b.eligibility[c.index] = 1.0
    path = os.path.join(tempfile.mkdtemp(), "b.npz")
    b.save(path)
    tasks.apply({**p, "reward_rate": 7.0})
    d = Brain.load(path)
    d.reward(1.0)
    w_loaded = d.weight[0]
    b.reward(1.0)            # уже созданный мозг тоже не меняется от apply
    w_live = b.weight[0]
    tasks.apply(p)
    check("№2 правила: загруженный мозг учится по своим правилам", w_loaded == 6.0, f"вес {w_loaded}")
    check("№2 правила: apply не меняет созданный мозг", w_live == 6.0, f"вес {w_live}")


# ---------- №3–6: найм памяти ----------

def recruit_brain(n_inputs=1, n_free=2, growth=False):
    b = Brain()
    b.add_region("input")
    b.add_region("mem", learn=0)
    names = [f"X{i}" for i in range(n_inputs)]
    for n in names:
        b.add_neuron(n, region="input")
    for i in range(n_free):
        b.add_neuron(f"W{i}", region="mem", context=True)
    if growth:
        Growth(b)
    r = Recruit(b, inputs=["input"], memory="mem", window=5)
    return b, r, names


def test_recruit():
    b, r, _ = recruit_brain()
    run(b, {0: ["X0"], 2: ["X0"]}, 8)
    check("№3 найм: одно событие — один владелец", len(r.owner) == 1, f"owner {r.owner}")

    b, r, names = recruit_brain(n_inputs=5, n_free=1, growth=True)
    run(b, {0: names}, 8)
    w = next(iter(r.owner))
    run(b, {}, 300)
    live = [c for c in b.connections if c.target == b.names[w] and c.synapse.weight > 0]
    check("№4 найм: закреплённые входы не удаляются", len(live) == 5, f"живых входов {len(live)}")
    out = run(b, {0: names}, 4)
    check("№4 найм: событие узнаётся через 300 шагов", any(b.names[w] in f for f in out))

    # №5: детектор X0+X1 не срабатывает от одного X0, повторённого 3 раза
    b, r, names = recruit_brain(n_inputs=2, n_free=1)
    run(b, {0: names}, 8)
    wname = b.names[next(iter(r.owner))]
    out = run(b, {0: ["X0"], 2: ["X0"], 4: ["X0"]}, 8)
    check("№5 найм: часть набора не зажигает (накопление)", not any(wname in f for f in out))
    out = run(b, {0: names}, 4)
    check("№5 найм: полный набор зажигает", any(wname in f for f in out))
    # 12 входов: 11 из 12 не зажигают
    b, r, names = recruit_brain(n_inputs=12, n_free=1)
    run(b, {0: names}, 8)
    wname = b.names[next(iter(r.owner))]
    out = run(b, {0: names[:11]}, 4)
    check("№5 найм: 11 из 12 входов не зажигают", not any(wname in f for f in out))
    out = run(b, {0: names}, 4)
    check("№5 найм: 12 из 12 зажигают", any(wname in f for f in out))

    # №6: прежние входы с разными задержками выравниваются
    b, r, names = recruit_brain(n_inputs=2, n_free=1)
    b.connect("X0", "W0", weight=1.0, myelin_thickness=0.0)   # задержка 5
    b.connect("X1", "W0", weight=1.0, myelin_thickness=1.0)   # задержка 1
    r._claim(0 + 2, [0, 1])
    out = run(b, {0: names}, 7)
    check("№6 найм: задержки выровнены, полный показ узнаётся", any("W0" in f for f in out))


# ---------- №7: порядок звуков ----------

def hearing_brain():
    b = Brain()
    b.add_region("слух")
    b.add_region("память слуха", learn=0.0, perceptual=True)
    for s in "xy":
        b.add_neuron(s, region="слух")
    h = Hearing(b, sounds="слух", memory="память слуха")
    h.counts[(0, 1, 1)] = 2
    h._grow(0, 1, 1)
    return b


def test_hearing():
    unit = "x→y/1"
    out = run(hearing_brain(), {0: ["y"], 1: ["x"]}, 8)
    check("№7 слух: обратный порядок не зажигает x→y", not any(unit in f for f in out))
    out = run(hearing_brain(), {0: ["x"], 1: ["y"]}, 8)
    check("№7 слух: прямой порядок зажигает x→y", any(unit in f for f in out))


# ---------- №8: доля ассоциации не больше 1 ----------

def test_association():
    b = Brain()
    b.add_neuron("A")
    b.add_neuron("B")
    # scale = 1: связь слабая и сама B не зажигает (иначе эхо меняет сценарий)
    a = Association(b, ["A"], ["B"], window=4, min_count=1, min_lift=0.0, scale=1.0)
    sched = {0: ["A", "B"], 2: ["A"], 4: ["A"], 6: ["A", "B"]}
    out = run(b, sched, 8)
    ws = [c.synapse.weight for c in a.links.values()]
    check("№8 ассоциация: B появился дважды, A один раз", a.episodes == {0: 1, 1: 2},
          f"{a.episodes}, срабатывания {out}")
    check("№8 ассоциация: сила не больше scale", ws and max(ws) <= 1.0 + 1e-9, f"силы {ws}")


# ---------- №9: заслуга по цепочке не зависит от порядка связей ----------

def chain_weight(reverse):
    b = Brain()
    for i in range(6):
        b.add_neuron(str(i), explores=(i == 5))
    edges = [(i, i + 1) for i in range(5)]
    if reverse:
        edges = edges[::-1]
    for s, t in edges:
        b.connect(str(s), str(t), weight=1.0, plasticity=1.0)
    b.input_budget = 0.0
    b.fire_time[5] = b.time
    b.eligibility[:b.n_conn] = 1.0
    b.reward(1.0)
    c = next(c for c in b.connections if (c.source, c.target) == ("0", "1"))
    return c.synapse.weight


def test_credit():
    a, r = chain_weight(False), chain_weight(True)
    check("№9 заслуга: порядок создания связей не важен", a == r, f"{a} / {r}")


# ---------- №10, №11: ответ с первого шага, эпизоды ----------

def test_first_step_and_episodes():
    tasks.apply(dict(tasks.PARAMS_DEFAULT))
    b = Brain()
    b.add_neuron("S1")
    b.add_neuron("L", explores=True)
    b.add_neuron("R", explores=True)
    b.potential[1] = b.threshold[1] + 1.0
    check("№10 tasks.trial видит ответ первого шага", tasks.trial(b, ["S1"]) == "L")

    b = Brain()
    b.add_neuron("X")
    b.add_neuron("L", explores=True)
    e = Episodes(b)
    b.potential[1] = b.threshold[1] + 1.0
    check("№10 Episodes._act видит ответ первого шага", e._act(b, ["X"]) == {"L"})

    b = Brain()
    b.add_neuron("X")
    b.add_neuron("L", explores=True)
    b.add_neuron("R", explores=True)
    b.connect("X", "L", weight=20.0, myelin_thickness=1.0)
    b.connect("X", "R", weight=20.0, myelin_thickness=1.0)
    e = Episodes(b)
    ep = Episode(("X",), "L", -1, 0)
    check("№11 сон: «оба» с наказанным L — не избежал наказания", not e.remembers(ep))


# ---------- №12: варианты сравнения различаются ----------

def test_compare():
    from compare_mechanisms import VARIANTS, check_unique
    configs = [{**tasks.PARAMS_DEFAULT, **v} for v in VARIANTS.values()]
    n = len({tuple(sorted(p.items())) for p in configs})
    check("№12 сравнение: все варианты разные", n == len(VARIANTS), f"{n} из {len(VARIANTS)}")
    check_unique(configs)


# ---------- №13, №14: быстрый движок ----------

def test_fast():
    import fast
    P = fast.params_array(dict(tasks.PARAMS_DEFAULT))
    sim = fast.Sim(4, 1, P)
    sim.add_neuron(False, False)
    sim.add_neuron(False, False)
    try:
        sim.grow_neuron(0, 1)
        ok = False
    except ValueError:
        ok = True
    check("№13 fast: нехватка места — ошибка, а не неполная сеть", ok)

    sim = fast.Sim(8, 16, P)
    for _ in range(3):
        sim.add_neuron(False, True)
    sim.connect(0, 1, 5.0, 1.0, 1.0, False, False, True)
    sim.ip_fixed[1] = True
    sim.ctx_trace[0] = 0.8
    sim.push_in[2] = 4.0
    sim.assoc[0] = True
    sim.perceptual[1] = True
    c = fast.clone(sim)
    diff = []
    for name, _ in fast.spec:
        x, y = getattr(sim, name), getattr(c, name)
        same = np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        if not same:
            diff.append(name)
    check("№14 fast: копия совпадает по всем полям", not diff, ", ".join(diff))


# ---------- №16, №17: границы API ----------

def test_api():
    b = Brain()
    b.add_neuron("X")
    b.add_neuron("Y")
    try:
        b.connect("X", "Y", hypothesis=True)
        ok = False
    except ValueError:
        ok = True
    check("№16 гипотеза без Growth — понятная ошибка", ok)
    try:
        b = Brain(neuron_capacity=0, connection_capacity=0)
        b.add_neuron("X")
        b.add_neuron("Y")
        b.connect("X", "Y")
        b.step()
        ok = True
    except Exception as e:
        ok = False
    check("№17 нулевая ёмкость расширяется", ok)


def main():
    for test in (test_pair, test_rules, test_recruit, test_hearing, test_association,
                 test_credit, test_first_step_and_episodes, test_compare, test_fast, test_api):
        test()
    tasks.apply(dict(tasks.PARAMS_DEFAULT))
    print(f"\nпровалов: {len(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
