"""Экзамен головы: прежние задачи на одном устройстве (Head.py), без ручной проводки.

Задача только показывает картинки (зрение), проигрывает звуки (слух) и
награждает действия. Картинки: S1 = {0,1}, S2 = {2,3}, C1 = {4,5}, C2 = {6,7};
правильные действия: L = D0, R = D1 (D2, D3 — неверные). Паузы между
попытками те же, что в ручных проверках.

Запуск из корня проекта: .venv/bin/python -m human_brain.tests.head_tests [число запусков]
"""
import os as _os
import sys as _sys
import sys

import numpy as np

import os

from human_brain.stages import tasks
from human_brain.stages import vision_hearing as vh
from human_brain.parts import Head
from human_brain.parts.Head import build_head, see, hear, action, quiet_copy

PIC = {"S1": (0, 1), "S2": (2, 3), "C1": (4, 5), "C2": (6, 7)}
ACT = {"L": "D0", "R": "D1"}

# что давала ручная сборка (прошлые замеры, исходные значения)
HAND = {"по награде (8 задач)": "0.96 (50 запусков)", "контексты: простой": "49/50",
        "контексты: разворот": "36-38/50", "ответ с задержкой": "20/20",
        "зрение и слух": "40/40, 40/40 (без знакомости)", "трава": "1.00"}


def trial(b, items):
    for name in items:
        see(b, PIC[name])
    for _ in range(7):        # с первого шага после показа
        a = action(b.step())
        if a is not None:
            return a
    return None


def reward_for(a, good):
    return +1 if a == good else (-1 if a is not None else 0)


def learn_items(b, items, n=60, rest=20):
    for k in range(n):
        inputs, good = items[k % len(items)]
        a = trial(b, inputs)
        b.reward(reward_for(a, ACT[good]))
        tasks.rest(b, rest)


def exam(b, inputs, good):
    return trial(quiet_copy(b), inputs) == ACT[good]


def reward_task(p, name, seed):
    b = build_head(p, seed)
    parts = tasks.TASKS[name]
    for inputs, good in parts:
        learn_items(b, [(inputs, good)])
    return all(exam(b, i, g) for i, g in parts)


def context_task(p, name, seed):
    b = build_head(p, seed)
    blocks = tasks.CONTEXT_TASKS[name]
    for items in blocks:
        learn_items(b, items)
    return all(exam(b, i, g) for items in blocks for i, g in items)


def delay_task(p, seed, delay=10, n=100, strict=False):
    """strict: ответ во время паузы — ошибка (наказание, попытка окончена)."""
    b = build_head(p, seed)

    def one(brain, s):
        see(brain, PIC[s])
        for _ in range(delay):
            a = action(brain.step())
            if strict and a is not None:
                return "рано"
        hear(brain, "f")
        for _ in range(7):
            a = action(brain.step())
            if a is not None:
                return a
        return None

    for k in range(n):
        s = ("S1", "S2")[k % 2]
        a = one(b, s)
        b.reward(-1 if a == "рано" else reward_for(a, ACT["L" if s == "S1" else "R"]))
        tasks.rest(b, 300)
    return all(one(quiet_copy(b), s) == ACT["L" if s == "S1" else "R"] for s in ("S1", "S2"))


SEQ = {"A": (4, 5), "C": (6, 7), "B": (0, 1)}


def seq_one(b, first, gap=5):
    """first (A или C), пауза gap шагов, затем B; ответ в 6 шагах после B.
    Ответы во время паузы не считаются (как в задаче с задержкой)."""
    see(b, SEQ[first])
    for _ in range(gap):
        b.step()
    see(b, SEQ["B"])
    for _ in range(6):
        a = action(b.step())
        if a is not None:
            return a
    return None


def seq_context(p, seed, n=200):
    """Схема владельца: A, потом B -> D0; C, потом B -> D1. Реакция на B зависит
    от того, что было раньше; попытки вперемешку."""
    import random
    rnd = random.Random(seed)
    good = {"A": "D0", "C": "D1"}
    b = build_head(p, seed)
    for _ in range(n):
        first = rnd.choice("AC")
        a = seq_one(b, first)
        b.reward(reward_for(a, good[first]))
        tasks.rest(b, 30)
    return all(seq_one(quiet_copy(b), f) == good[f] for f in "AC")


def vision_hearing(p, seed):
    import random
    rnd = random.Random(seed)
    b = build_head(p, seed)
    order = [pic for pic in vh.PICTURES for _ in range(20)]
    rnd.shuffle(order)
    for k, pic in enumerate(order):
        vh.show(b, pic, vh.WORDS[pic], picture_first=(k % 2 == 0))
        tasks.rest(b, 40)
    word_ok = order_ok = pic_ok = 0
    vh.USE_HEARING = True
    for pic, word in vh.WORDS.items():
        heard = vh.recall_word(b, pic)
        if set(heard) == {f"A{s}" for s in word}:
            word_ok += 1
            order_ok += heard[f"A{word[0]}"] < heard[f"A{word[1]}"]
        pic_ok += vh.recall_picture(b, pic) == {f"V{i}" for i in vh.PICTURES[pic]}
    return word_ok, order_ok, pic_ok


def grass(p, seed):
    b = build_head(p, seed)
    early = late = 0
    for k in range(200):
        s0 = b.neurons["V0"].spikes
        see(b, PIC["S1"])
        tasks.rest(b, 30)
        noticed = b.neurons["V0"].spikes > s0
        early += k < 10 and noticed
        late += k >= 190 and noticed
    s0 = b.neurons["V2"].spikes
    see(b, PIC["S2"])
    tasks.rest(b, 30)
    new_seen = b.neurons["V2"].spikes > s0
    for _ in range(5):
        see(b, PIC["S1"])
        tasks.rest(b, 3)
        b.reward(1.0)
        tasks.rest(b, 27)
    s0 = b.neurons["V0"].spikes
    see(b, PIC["S1"])
    tasks.rest(b, 30)
    regained = b.neurons["V0"].spikes > s0
    return (float(early >= 8 and late <= 1) + float(new_seen) + float(regained)) / 3


def main(runs=10):
    if "HEAD_FIXES" in os.environ:      # например HEAD_FIXES=budget,pairs или пусто
        Head.FIXES = {x for x in os.environ["HEAD_FIXES"].split(",") if x}
    print("правки:", sorted(Head.FIXES) or "нет", flush=True)
    p = dict(tasks.PARAMS_DEFAULT)
    seeds = range(10000, 10000 + runs)
    res = [np.mean([reward_task(p, n, s) for s in seeds]) for n in tasks.REWARD_TASKS]
    print(f"по награде (8 задач):     голова {np.mean(res):.2f}  ("
          + ", ".join(f"{n} {r:.1f}" for n, r in zip(tasks.REWARD_TASKS, res))
          + f");  ручная сборка {HAND['по награде (8 задач)']}", flush=True)
    for name, label in (("ctx_simple", "контексты: простой"), ("ctx_reverse", "контексты: разворот")):
        ok = sum(context_task(p, name, s) for s in seeds)
        print(f"{label:25s} голова {ok}/{runs};  ручная сборка {HAND[label]}", flush=True)
    ok = sum(delay_task(p, s) for s in seeds)
    print(f"{'ответ с задержкой':25s} голова {ok}/{runs};  ручная сборка {HAND['ответ с задержкой']}", flush=True)
    ok = sum(delay_task(p, s, strict=True) for s in seeds)
    print(f"{'ждать сигнала (строго)':25s} голова {ok}/{runs}", flush=True)
    ok = sum(seq_context(p, s) for s in seeds)
    print(f"{'A..B -> D0, C..B -> D1':25s} голова {ok}/{runs}", flush=True)
    vhr = [vision_hearing(p, s) for s in seeds]
    print(f"{'зрение и слух':25s} голова: картинка->слово {sum(r[0] for r in vhr)}/{4 * runs} "
          f"(порядок {sum(r[1] for r in vhr)}), слово->картинка {sum(r[2] for r in vhr)}/{4 * runs};  "
          f"ручная сборка {HAND['зрение и слух']}", flush=True)
    g = np.mean([grass(p, s) for s in seeds])
    print(f"{'трава':25s} голова {g:.2f};  ручная сборка {HAND['трава']}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
