"""И, ИЛИ, НЕ: может ли мозг выучить логические операторы тем, что в нём уже есть.

Мозг — свежий, на ядре, без новых устройств: рост связей и нейронов-сочетаний
(Growth), обучаемое торможение вход -> действие, любопытство, награда.
Устроен как мозг вычитания (subtraction.make_brain), только входы другие:
    входы A, B и сигнал «отвечай» (GO, есть в каждой пробе — иначе при «ни A,
    ни B» нечему вызвать ответ, а НЕ без него не проверить);
    действия YES и NO соревнуются (взаимное торможение).
Проба: показаны GO и те из A, B, что «истинны»; ответ — первое действие за 7
шагов; награда +1 за верный ответ, -1 за неверный, 0 за молчание.
Операторы (свой свежий мозг на каждый):
    AND — да, если A и B;   OR — да, если A или B;   NOT — да, если нет A
    (B не показывается);   XOR — да, если ровно один (для сравнения: его
    одними силами связей не решить, нужен нейрон-сочетание и торможение).
Экзамен на копии без шума и обучения: каждое сочетание входов; оператор
выучен, если верны все. Ленивые стратегии («всегда да», «всегда нет») всё
не проходят ни на одном операторе.
Контроль: верный ответ в каждой пробе случайный (учить нечему).
--hand: ручная сборка (заведомо решаемая проверка): И — нейрон-сочетание
A+B -> YES; ИЛИ — A -> YES и B -> YES; НЕ — GO -> YES и тормоз A -> YES;
GO -> NO слабее (побеждает, когда YES не зовут).

Запуск из папки human_brain:
    ../.venv/bin/python logic.py [--op AND] [--seed 0] [--trials 2000] [--control] [--hand]
"""
import argparse
import copy
import random

import tasks
import Head
from Brain import Brain

OPS = {
    "AND": (lambda a, b: a and b, True),
    "OR": (lambda a, b: a or b, True),
    "NOT": (lambda a, b: not a, False),     # False: B не используется
    "XOR": (lambda a, b: a != b, True),
}
INPUTS = ("A", "B", "GO")
ACTIONS = ("YES", "NO")
WAIT = 7
DIRECT_THICKNESS = 0.75   # задержка 2 шага (myelin.delay)
SILENCE = False      # --silence: адресное наказание за молчание (Brain.punish_blockers)
REST = 20


def cases(op):
    _, two = OPS[op]
    if two:
        return [(a, b) for a in (False, True) for b in (False, True)]
    return [(False, False), (True, False)]


def make_brain(p, seed, hand=False, op=None):
    b = Brain()
    b.add_region("входы")
    for n in INPUTS:
        b.add_neuron(n, region="входы")
    if hand:
        b.add_region("действия, жёсткие", learn=0.0)
    b.add_region("действия")
    for n in ACTIONS:
        b.add_neuron(n, explores=True, region="действия, жёсткие" if hand else "действия")
    if hand:
        _hand(b, op)
        b.inh_delayed = True     # тормоз вход -> действие приходит вместе с возбуждением
    else:
        for s in INPUTS:
            for o in ACTIONS:
                # вход -> действие: одна обучаемая связь (с минимумом путей к
                # действиям 1.58: три входа сразу дают 4.7, ниже порога 8).
                # Пробовалось: как у мозга вычитания (неизменная слабая + без
                # обучаемой) — случай «только GO» не отвечался никогда (сочетания
                # из одного сигнала нет); неизменная + обучаемая — три входа
                # сразу перешагивали порог, «нет» срабатывало рефлексом
                # задержка 2 — как у пути через нейрон-сочетание (1 до сочетания
                # + 1 до действия): иначе прямой путь всегда приходил на шаг
                # раньше, и сочетание (A+B для И) не могло выиграть гонку
                b.connect(s, o, weight=p["init_weight"], myelin_thickness=DIRECT_THICKNESS,
                          plasticity=p["init_plasticity"])
                # у отдельных датчиков тормоз не учится (как у мозга вычитания):
                # GO есть в каждой пробе, и его тормоз к «да» рос до предела и
                # запирал «да» навсегда; обучаемый тормоз получают сочетания
                b.connect(s, o, weight=0.0, inhibitory=True, myelin_thickness=DIRECT_THICKNESS,
                          plasticity=p.get("sensor_inh_plasticity", 0.0))
        b.inh_window = Head.INH_WINDOW
        b.inh_delayed = True
    b.connect("YES", "NO", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("NO", "YES", weight=p["inhibit_weight"], inhibitory=True)
    b.inh_short = True
    b.reset_eligibility = True
    b.last_action_only = True
    b.budget_above_floor = True
    b.reward_only_actions = True
    if not hand:
        tasks.attach(b, p, seed)
        b.neuron_change.make_same(("входы", "действия"))
        b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
        b.curiosity.avoid = p.get("avoid", 3.0)
        b.growth.sensory_regions = ["входы"]
        b.growth.max_neurons = b.n_neurons + 12
        b.growth.pair_regions = ["входы"]
        b.growth.new_weight = p["action_min"] + 0.1
        b.growth.inherit_inhibition = p.get("inh_plasticity", Head.INH_PLASTICITY)
        b.budget_window = p.get("budget_window", Head.BUDGET_WINDOW)
    return b


def _hand(b, op):
    th = b.reach(b.neurons["YES"].index)
    for s in INPUTS:
        for o in ACTIONS:
            b.connect(s, o, weight=0.0, myelin_thickness=1.0)
    # GO -> NO: ответ «нет», если «да» не зовут; приходит на шаг позже (толще
    # не значит быстрее: задержка 2), чтобы «да» успело победить
    b.connect("GO", "NO", weight=th + 2.0, myelin_thickness=0.75)
    if op == "AND":
        n = b.add_neuron("A+B")
        b.make_coincidence(n.index)
        for s in ("A", "B"):
            b.connect(s, "A+B", weight=0.6 * b.reach(n.index), myelin_thickness=1.0, fixed=True)
        b.connect("A+B", "YES", weight=th + 4.0, myelin_thickness=1.0)
    elif op == "OR":
        b.connect("A", "YES", weight=th + 4.0, myelin_thickness=1.0)
        b.connect("B", "YES", weight=th + 4.0, myelin_thickness=1.0)
    elif op == "NOT":
        b.connect("GO", "YES", weight=th + 4.0, myelin_thickness=1.0)
        b.connect("A", "YES", weight=th + 10.0, inhibitory=True, myelin_thickness=1.0)
        # и «нет» не должно ждать, если «да» заторможено: NO от GO уже есть
    elif op == "XOR":
        n = b.add_neuron("A+B")
        b.make_coincidence(n.index)
        for s in ("A", "B"):
            b.connect(s, "A+B", weight=0.6 * b.reach(n.index), myelin_thickness=1.0, fixed=True)
            b.connect(s, "YES", weight=th + 4.0, myelin_thickness=0.75)
        b.connect("A+B", "YES", weight=3 * th, inhibitory=True, myelin_thickness=1.0)


def acted(fired):
    a = [n for n in ACTIONS if n in fired]
    return a[0] if len(a) == 1 else ("оба" if a else None)


def trial(b, a, c):
    shown = ["GO"] + (["A"] if a else []) + (["B"] if c else [])
    for s in shown:
        b.stimulate(s, 20.0)
    for _ in range(WAIT):
        out = acted(b.step())
        if out is not None:
            return out
    return None


def rest(b):
    for _ in range(REST):
        b.step()


def exam(b, op):
    t = copy.deepcopy(b)
    t.curiosity = None
    f, _ = OPS[op]
    rows = []
    for a, c in cases(op):
        out = trial(t, a, c)
        rest(t)
        rows.append(((a, c), "YES" if f(a, c) else "NO", out))
    return rows


def train(b, op, trials, rng, control=False):
    f, _ = OPS[op]
    cs = cases(op)
    right = 0
    for _ in range(trials):
        a, c = rng.choice(cs)
        want = rng.choice(ACTIONS) if control else ("YES" if f(a, c) else "NO")
        out = trial(b, a, c)
        right += out == want
        if out is None and SILENCE:
            b.punish_blockers(1.0)    # молчание: наказаны тормоза, помешавшие ответу
        else:
            b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
        rest(b)
    return right / trials


def run(p, op, seed, trials=2000, control=False):
    tasks.apply(p)
    b = make_brain(p, seed)
    train(b, op, trials, random.Random(seed), control)
    rows = exam(b, op)
    return all(w == o for _, w, o in rows), rows, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="AND", choices=list(OPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--silence", action="store_true")
    a = ap.parse_args()
    global SILENCE
    SILENCE = a.silence
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    if a.hand:
        b = make_brain(p, a.seed, hand=True, op=a.op)
        rows = exam(b, a.op)
        ok = all(w == o for _, w, o in rows)
    else:
        ok, rows, b = run(p, a.op, a.seed, a.trials, a.control)
    print(a.op, "выучен" if ok else "не выучен", rows)


if __name__ == "__main__":
    main()
