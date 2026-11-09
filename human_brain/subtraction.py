"""Вычитание после сложения (мир candy_world/arith/arith.py), учим до 10, проверяем любые.

Дни:
    сложение  — как в addition.py: карточка, группы гостей, вопрос ASK
                «сколько принести?»; ответ = число гостей;
    вычитание — карточка, у меня c конфет (CANDY по одной), пришли g гостей
                (группами), вопрос ASK «сколько не хватило?» = max(g - c, 0)
                или ASK_LEFT «сколько останется?» = max(c - g, 0).
После праздника видно, скольким не хватило или сколько осталось: верное число
со знаком g - c (не хватило минус осталось) учит накопитель.

Мозг = мозг сложения (addition.py, те же выключатели) плюс:
    накопитель со знаком: на GO загорается ACC (уровень > 0) или ACC_NEG (< 0),
        «ещё» сдвигает уровень на порцию к 0;
    защёлка (Latch.py): помнит вопрос дня, на GO зажигает M_BRING или M_LEFT.
Мозг должен выучить: «ещё» = (M_BRING и ACC) или (M_LEFT и ACC_NEG), иначе
«готово». Одними силами связей это не решается (нужны нейроны-сочетания,
мозг выращивает их сам из входов: места под рост 12 нейронов). Датчики
(GO, ACC, ACC_NEG, M_*) к действиям — слабые неизменные связи, учатся связи
сочетаний (иначе прямой путь решает раньше сочетания); --direct: как раньше.

Порядок (как arith.py): ступень 1 — только сложение; ступень 2 — вычитание
(оба вопроса), каждый второй день сложение (прежнее не забывать). Учим на
числах до 10, сочетания hidden(c, g) в обучении не показываются.
Экзамен на копии без шума и обучения: сложение (как addition.py) и вычитание
по каждому вопросу: свежие до 10, скрытые сочетания, 11-20, 21-50, 51-100,
101-1000. Рядом ленивая «всегда 0».
--hand: ручная сборка (сочетания заданы руками; заведомо решаемая проверка).

Запуск из папки human_brain:
    ../.venv/bin/python subtraction.py [--seed 0] [--add-days 300] [--sub-days 1000] [--hand]
"""
import argparse
import copy
import random

import tasks
import Head
import addition as ad
from Brain import Brain
from Accumulator import Accumulator
from Latch import Latch

EVENTS = ("CARD", "CANDY", "DOOR", "GUEST", "ASK", "ASK_LEFT")
READ = ("GO", "ACC", "ACC_NEG", "M_BRING", "M_LEFT")
ACTIONS = ad.ACTIONS


# ---------- мир ----------

def hidden(a, b):
    """Сочетания, которых нет в обучении (как arith.py)."""
    return (a * 7 + b * 3) % 6 == 0


def make_sub_day(rng, question, top=10, low=0, want_hidden=False):
    """(события, ответ, верное число со знаком). question: ASK или ASK_LEFT."""
    for _ in range(10000):
        c, g = rng.randint(low, top), rng.randint(max(2, low), top)
        if hidden(c, g) != want_hidden:
            continue
        events = ["CARD"] + ["CANDY"] * c + ad.groups(g, rng, at_least=1) + [question]
        answer = max(g - c, 0) if question == "ASK" else max(c - g, 0)
        return events, answer, g - c
    raise RuntimeError("нет дня")


def make_add_day(rng, top=10, low=0):
    events, g = ad.make_day(rng, top, low)
    return events, g, g


# ---------- мозг ----------

def make_brain(p, seed, hand=False):
    b = Brain()
    b.add_region("входы")
    for n in EVENTS + READ:
        b.add_neuron(n, region="входы")
    if hand:
        b.add_region("действия, жёсткие", learn=0.0)
    b.add_region("действия")
    for n in ACTIONS:
        b.add_neuron(n, explores=True, region="действия, жёсткие" if hand else "действия")
    if hand:
        # сочетания руками: И1 = M_BRING и ACC, И2 = M_LEFT и ACC_NEG -> «ещё»;
        # GO -> «готово» с задержкой 2 (к тому же шагу, что и сигнал сочетаний)
        for name, (x, y) in (("И1", ("M_BRING", "ACC")), ("И2", ("M_LEFT", "ACC_NEG"))):
            nn = b.add_neuron(name)
            b.make_coincidence(nn.index)
            for s in (x, y):
                b.connect(s, name, weight=0.6 * b.reach(nn.index), myelin_thickness=1.0, fixed=True)
            b.connect(name, "MORE", weight=20.0, myelin_thickness=1.0)
        for s in EVENTS + READ:
            for o in ACTIONS:
                w, th = (10.0, 0.75) if (s, o) == ("GO", "DONE") else (1.0, 1.0)
                b.connect(s, o, weight=w, myelin_thickness=th)
    else:
        for s in EVENTS + READ:
            for o in ACTIONS:
                if not p.get("direct_read", False):
                    # датчики и события -> действия: неизменные слабые связи (минимум
                    # путей к действиям; слабый фон помогает поиску: всё вместе 20/24
                    # против 12/24 при силе 0); учатся связи нейронов-сочетаний.
                    # Иначе прямой путь (ACC -> MORE, ASK -> MORE + слабые датчики)
                    # решает на шаг раньше, чем сочетание успевает сказать своё
                    b.connect(s, o, weight=p.get("sensor_weight", p["action_min"]),
                              myelin_thickness=1.0, fixed=True)
                else:
                    b.connect(s, o, weight=p["init_weight"], myelin_thickness=1.0,
                              plasticity=p["init_plasticity"])
    if not hand and p.get("inhibit_on", True):
        # обучаемое торможение вход -> действие (с задержкой и меткой «помешал»)
        # тормоз от отдельного датчика двусмыслен (M_LEFT есть и там, где «ещё»
        # верно, и там, где нет) и, раз выучившись, запирал «ещё» навсегда;
        # поэтому у датчиков тормозящие связи не учатся (пластичность 0), а
        # обучаемые получают нейроны-сочетания (Growth.inherit_inhibition)
        for s in EVENTS + READ:
            for o in ACTIONS:
                b.connect(s, o, weight=0.0, inhibitory=True, myelin_thickness=1.0,
                          plasticity=p.get("sensor_inh_plasticity", 0.0))
        b.inh_window = Head.INH_WINDOW
        b.inh_delayed = True
    b.connect("MORE", "DONE", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("DONE", "MORE", weight=p["inhibit_weight"], inhibitory=True)
    # те же выключатели, что у мозга сложения
    b.inh_short = True
    b.reset_eligibility = True
    b.last_action_only = True
    b.budget_above_floor = True
    b.reward_only_actions = True
    if not hand:
        tasks.attach(b, p, seed)
        b.neuron_change.make_same(("входы", "действия"))
        b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
        b.growth.sensory_regions = ["входы"]
        b.growth.max_neurons = b.n_neurons + 12      # место для нейронов-сочетаний
        # сочетания только из входов (правка головы pairs): иначе за ступень
        # сложения место под рост съедали сочетания из сочетаний и с действиями,
        # и к вычитанию не оставалось места для M_LEFT + ACC_NEG
        b.growth.pair_regions = ["входы"]
        # новые связи чуть выше минимума путей к действиям: иначе (new_weight 1.45 <
        # action_min 1.58) их сразу подтягивает к минимуму, и правило «так нельзя»
        # закрепляет их с рождения (награда меняет их в 20 раз слабее)
        b.growth.new_weight = p["action_min"] + 0.1
        # «проиграл — смени»: наказанное действие ненадолго подавляется
        b.curiosity.avoid = p.get("avoid", 3.0)
        # запас входов только на недавно работавшие входы (правка головы budget):
        # иначе каждая награда в днях сложения срезала молодой путь
        # «ACC_NEG + M_LEFT -> MORE», который там молчит, до минимума, и правило
        # «так нельзя» закрепляло его без единого наказания
        b.budget_window = p.get("budget_window", Head.BUDGET_WINDOW)
        if p.get("inhibit_on", True):
            b.growth.inherit_inhibition = p.get("inh_plasticity", Head.INH_PLASTICITY)
    acc = Accumulator(b, "ACC", EVENTS, decrement=["MORE"], query="GO",
                      signed=True, neg_neuron="ACC_NEG", register=p.get("register", False))
    Latch(b, inputs=["ASK", "ASK_LEFT"], outputs=["M_BRING", "M_LEFT"], query="GO")
    if hand:
        acc.w["GUEST"], acc.w["CANDY"] = 1.0, -1.0
        if acc.register:   # ручная сборка умножения: карточки в регистр, гость прибавляет регистр
            acc.w["GUEST"] = 0.0
            acc.m = {k: 0.0 for k in acc.m}
            acc.r = {k: 0.0 for k in acc.r}
            acc.m["GUEST"], acc.r["CARD"] = 1.0, 1.0
    return b


# ---------- обучение и экзамен ----------

def train(b, rng, add_days, sub_days):
    for _ in range(add_days):
        ev, ans, truth = make_add_day(rng)
        ad.day(b, ev, ans, learn=True, truth=truth)
    for k in range(sub_days):
        if k % 2:
            ev, ans, truth = make_add_day(rng)
        else:
            ev, ans, truth = make_sub_day(rng, rng.choice(("ASK", "ASK_LEFT")))
        ad.day(b, ev, ans, learn=True, truth=truth)


def exam_sets(seed, n=30):
    rng = random.Random(1000 + seed)
    sets = {"сложение до 10": [make_add_day(rng) for _ in range(n)],
            "сложение 11-20": [make_add_day(rng, 20, 11) for _ in range(n)],
            "сложение 21-50": [make_add_day(rng, 50, 21) for _ in range(n)],
            "сложение 101-1000": [make_add_day(rng, 1000, 101) for _ in range(5)]}
    for q, label in (("ASK", "не хватило"), ("ASK_LEFT", "осталось")):
        sets[f"{label}: до 10"] = [make_sub_day(rng, q) for _ in range(n)]
        sets[f"{label}: скрытые"] = [make_sub_day(rng, q, want_hidden=True) for _ in range(n)]
        sets[f"{label}: 11-20"] = [make_sub_day(rng, q, 20, 11) for _ in range(n)]
        sets[f"{label}: 21-50"] = [make_sub_day(rng, q, 50, 21) for _ in range(n)]
        sets[f"{label}: 51-100"] = [make_sub_day(rng, q, 100, 51) for _ in range(10)]
        sets[f"{label}: 101-1000"] = [make_sub_day(rng, q, 1000, 101) for _ in range(5)]
    return sets


def exam(b, days):
    t = copy.deepcopy(b)
    t.curiosity = None
    ok = 0
    for events, answer, _ in days:
        more, done, _ = ad.day(t, events, answer, learn=False)
        ok += more == answer and done
    return ok


def run(p, seed, add_days=300, sub_days=1000):
    tasks.apply(p)
    b = make_brain(p, seed)
    train(b, random.Random(seed), add_days, sub_days)
    return {k: exam(b, v) / len(v) for k, v in exam_sets(seed).items()}, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add-days", type=int, default=300)
    ap.add_argument("--sub-days", type=int, default=1000)
    ap.add_argument("--hand", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    if a.hand:
        b = make_brain(p, a.seed, hand=True)
    else:
        b = make_brain(p, a.seed)
        train(b, random.Random(a.seed), a.add_days, a.sub_days)
    print("ЭКЗАМЕН (мозг / ленивая «всегда 0»):")
    for name, days in exam_sets(a.seed).items():
        lazy = sum(ans == 0 for _, ans, _ in days)
        print(f"  {name:22s} {exam(b, days)}/{len(days)}   ({lazy}/{len(days)})", flush=True)


if __name__ == "__main__":
    main()
