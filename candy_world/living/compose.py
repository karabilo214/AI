"""Сборка сложного из найденного простого.

Библиотека: таблица счётчика, найденная отбором (candy_world/logic, seed 4). Она закреплена, её проверяет хеш.
У неё три действия, которые она уже умеет: "+1" (событие гость), "-1" (предмет) и "проверить" (поход:
вычесть один и сказать, было ли больше нуля). Можно взять несколько независимых копий: C, R, S.

Отбор ищет не таблицу, а маленькую программу: что делать на каждое событие из вызовов библиотеки.
Инструкция: ("inc", X), ("dec", X) или ("while", X, тело): "пока проверка X говорит да, выполнять тело".
Ответ дня: сколько раз проверка C говорит "да" в фазе походов (как и раньше).
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

LIBRARY = Path(__file__).resolve().parent.parent / "logic" / "results_widths" / "seed_4.json"
TABLE = np.array([int(c) for c in json.loads(LIBRARY.read_text(encoding="utf-8"))["таблица"]], dtype=np.uint8).reshape(32, 3)
TABLE_HASH = hashlib.sha256(TABLE.tobytes()).hexdigest()
CHAINS = ("C", "R", "S")
INC, DEC, CHECK = 1, 2, 3                      # коды событий, которые таблица понимает
TICK, GUEST, THING = "tick", "guest", "thing"   # события мира


class Budget(Exception):
    pass


class Counter:
    """Одна копия библиотечного счётчика: цепочка элементов с найденной таблицей. Таблица не меняется."""

    def __init__(self, width):
        self.bits = [0] * width

    def event(self, code):
        a = b = 1
        for i, bit in enumerate(self.bits):
            new, a, b = TABLE[(code << 3) | (bit << 2) | (a << 1) | b]
            self.bits[i] = int(new)
        return int(b)


class Machine:
    def __init__(self, width, limit):
        self.counters = {name: Counter(width) for name in CHAINS}
        self.steps, self.limit = 0, limit

    def call(self, name, code):
        self.steps += 1
        if self.steps > self.limit:
            raise Budget
        return self.counters[name].event(code)

    def run(self, program):
        for statement in program:
            if statement[0] == "inc":
                self.call(statement[1], INC)
            elif statement[0] == "dec":
                self.call(statement[1], DEC)
            else:
                while self.call(statement[1], CHECK):
                    self.run(statement[2])


def answer(programs, events, width=12, limit=6000):
    machine = Machine(width, limit)
    try:
        for e in events:
            machine.run(programs[e])
        trips = 0
        while machine.call("C", CHECK):
            trips += 1
        return trips
    except Budget:
        return -1


def make_days(task, count, limit, rng, need_range=(1, 3)):
    days = []
    for _ in range(count):
        things, guests = rng.randint(0, limit), rng.randint(0, limit)
        need = {"double": 2, "triple": 3}.get(task) or rng.randint(*need_range)
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        if task == "multiply":
            events = [TICK] * need + events
        days.append((events, max(guests * need - things, 0)))
    return days


def score(programs, days, width=12):
    got = [answer(programs, e, width) for e, _ in days]
    right = [r for _, r in days]
    errors = sum(abs(g - r) if g >= 0 else 100 for g, r in zip(got, right))
    return -errors / len(days), sum(g == r for g, r in zip(got, right)) / len(days)


# Ручные программы: заведомо решаемая проверка, что из библиотеки это собирается.
HAND = {
    "double": {GUEST: [("inc", "C"), ("inc", "C")], THING: [("dec", "C")]},
    "triple": {GUEST: [("inc", "C")] * 3, THING: [("dec", "C")]},
    "multiply": {
        TICK: [("inc", "R")],
        GUEST: [("while", "R", [("inc", "C"), ("inc", "S")]), ("inc", "R"),
                ("while", "S", [("inc", "R")]), ("inc", "S")],
        THING: [("dec", "C")],
    },
}
