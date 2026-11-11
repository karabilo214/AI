"""Сумматор как отдельная операция (2026-11-11): к счёту прибавить хранимое N один раз.

События (3 бита кода): штрих карточки (N += 1), «+1» (счёт += 1), «прибавь N»
(счёт += N, один раз за день, стоит в случайном месте среди «+1»), поход.
Ответ дня: A + N (A — сколько было «+1»). Счёт к моменту «прибавь N» обычно
не пуст, поэтому скопировать N в счёт нельзя — нужно настоящее сложение с
переносом по цепочке. Устройство как у «по N каждому» (task_multiply.DESIGN:
2 бита памяти, 512 бит таблицы). Сигнала мира по ходу дня нет.
"""
import numpy as np

from task_multiply import DESIGN, hand_table as mul_hand

TICK, ADDN, FETCH, PLUS = 1, 2, 4, 5
CODES = {"fetch": FETCH}


def make_days(items, rng):
    """items: (A, N)."""
    days, right, empty = [], [], []
    for a, n in items:
        body = [PLUS] * a + [ADDN]
        rng.shuffle(body)
        ev = [TICK] * n + body
        days.append(ev)
        right.append(a + n)
        empty.append([None] * len(ev))
    return days, np.array(right), empty


def hand_table():
    """Ручная: «прибавь N» — как гость в task_multiply (сумматор с перевёрнутым
    переносом), «+1» — прибавить 1 к счёту (перенос как у штриха)."""
    d = DESIGN
    t = mul_hand().reshape(d.rows, d.out_bits).copy()
    for c in (0, 1):
        for n in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    row = (PLUS << 4) | (c << 3) | (n << 2) | (a << 1) | b
                    nc = c ^ a
                    t[row] = (nc, n, c & a, 1 - nc)
    return t.reshape(-1)
