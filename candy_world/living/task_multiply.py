"""Задача "по N каждому": в начале дня карточка штрихами говорит N, каждый гость хочет N, предмет даёт 1.

Коды событий (3 бита): штрих карточки, гость, предмет, поход. У элемента 2 бита памяти.
"""
import random

import numpy as np

from engine import Design

TICK, GUEST, THING, FETCH = 1, 2, 3, 4
CODES = {"fetch": FETCH}
DESIGN = Design(code_bits=3, state_bits=2)


def make_days(items, rng):
    """items: (предметов, гостей, N)."""
    days, right, empty = [], [], []
    for things, guests, need in items:
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        events = [TICK] * need + events
        days.append(events)
        right.append(max(guests * need - things, 0))
        balance, row = 0, []
        for e in events:
            if e == GUEST:
                balance += need
            elif e == THING:
                balance -= 1
            row.append(int(balance >= 0) if e in (GUEST, THING) else None)
        empty.append(row)
    return days, np.array(right), empty


def hand_table():
    """Проверка движка: бит 0 памяти это счётчик, бит 1 хранит N.

    Штрих: N += 1. Гость: счётчик += N (перенос передаётся в перевёрнутом виде, потому что первый
    элемент получает слева сигнал 1). Предмет и поход: счётчик −= 1. Сигнал b: "мой бит счётчика 0".
    """
    d = DESIGN
    t = np.zeros((d.rows, d.out_bits), dtype=np.uint8)
    for code in range(8):
        for c in (0, 1):
            for n in (0, 1):
                for a in (0, 1):
                    for b in (0, 1):
                        row = (code << 4) | (c << 3) | (n << 2) | (a << 1) | b
                        nc, nn, carry = c, n, 0
                        if code == TICK:
                            nn, carry = n ^ a, n & a
                        elif code == GUEST:
                            cin = 1 - a
                            nc = c ^ n ^ cin
                            carry = 1 - ((c & n) | (c & cin) | (n & cin))
                        elif code in (THING, FETCH):
                            nc, carry = c ^ a, (1 - c) & a
                        t[row] = (nc, nn, carry, 1 - nc)
    return t.reshape(-1)
