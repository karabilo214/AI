"""Простая задача: гость +1, предмет −1, поход −1. Ручная таблица для проверки движка."""
import random

import numpy as np

from engine import Design

GUEST, THING, FETCH = 1, 2, 3
CODES = {"fetch": FETCH}


def make_days(pairs, rng):
    days, right, empty = [], [], []
    for things, guests in pairs:
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        days.append(events)
        right.append(max(guests - things, 0))
        balance, row = 0, []
        for e in events:
            balance += 1 if e == GUEST else -1
            row.append(int(balance >= 0))
        empty.append(row)
    return days, np.array(right), empty


def hand_table():
    """Двоичный счётчик для базового устройства: 2 бита кода, 1 бит памяти, 2 правых сигнала."""
    d = Design()
    t = np.zeros((d.rows, d.out_bits), dtype=np.uint8)
    for code in range(4):
        for bit in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    row = (code << 3) | (bit << 2) | (a << 1) | b
                    if code == GUEST:
                        new, carry = bit ^ a, bit & a
                    elif code in (THING, FETCH):
                        new, carry = bit ^ a, (1 - bit) & a
                    else:
                        new, carry = bit, 0
                    t[row] = (new, carry, 1 - new)
    return t.reshape(-1)
