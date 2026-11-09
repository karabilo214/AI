"""Разные операции на одном мире. Те же события: гость, предмет, поход.

guests: сходить столько раз, сколько гостей (предметы не важны).
parity: сходить один раз, если гостей нечётное число (кто-то остался без пары).
double: сходить 2 × гостей − предметов раз (каждому по 2). Здесь нужны виды нейронов.
"""
import numpy as np

from engine import Design

GUEST, THING, FETCH = 1, 2, 3
CODES = {"fetch": FETCH}
DESIGNS = {"guests": Design(), "parity": Design(), "double": Design(types=3)}


def balance_step(op, balance, event):
    if op == "guests":
        return balance + (1 if event == GUEST else 0)
    if op == "parity":
        return balance ^ (1 if event == GUEST else 0)
    return balance + (2 if event == GUEST else -1)


def answer(op, things, guests):
    if op == "guests":
        return guests
    if op == "parity":
        return guests % 2
    return max(2 * guests - things, 0)


def seen(op, balance):
    """Что мир показывает после события: всё ли в порядке."""
    return int(balance >= 0) if op != "parity" else 1 - balance


def make_days(op, pairs, rng):
    days, right, empty = [], [], []
    for things, guests in pairs:
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        days.append(events)
        right.append(answer(op, things, guests))
        balance, row = 0, []
        for e in events:
            balance = balance_step(op, balance, e)
            row.append(seen(op, balance))
        empty.append(row)
    return days, np.array(right), empty


def hand_table(op):
    d = DESIGNS[op]
    t = np.zeros((d.types, d.rows, d.out_bits), dtype=np.uint8)
    for kind in range(d.types):
        for code in range(4):
            for bit in (0, 1):
                for a in (0, 1):
                    for b in (0, 1):
                        row = (code << 3) | (bit << 2) | (a << 1) | b
                        new, carry, out_b = bit, 0, 1 - bit
                        if op == "guests":
                            if code == GUEST:
                                new, carry = bit ^ a, bit & a
                            elif code == FETCH:
                                new, carry = bit ^ a, (1 - bit) & a
                            out_b = 1 - new
                        elif op == "parity":
                            # Чётность хранит только первый элемент (у него слева a=1). Остальные передают b дальше.
                            if code in (GUEST, FETCH) and a:
                                new = bit ^ 1
                            out_b = (1 - new if code == GUEST else bit) if a else b
                            if code == THING:
                                out_b = (1 - bit) if a else b
                        else:
                            first = kind == 0 and d.types == 3
                            if code == GUEST:
                                if first:                     # +2: свой бит не трогаем, перенос идёт дальше
                                    new, carry = bit, a
                                else:
                                    new, carry = bit ^ a, bit & a
                            elif code in (THING, FETCH):
                                new, carry = bit ^ a, (1 - bit) & a
                            out_b = 1 - new
                        t[kind, row] = (new, carry, out_b)
    return t.reshape(-1)
