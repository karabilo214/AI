"""Сеть из одинаковых логических элементов. Умножений нет: каждый шаг элемента это чтение из таблицы.

Каждый элемент хранит один бит состояния и одинаковую для всех таблицу из 32 строк. Элементы стоят
цепочкой. На каждом событии элемент читает: что за событие, свой бит и два сигнала от соседа слева.
Таблица говорит: новый бит состояния и два сигнала соседу справа. Сигнал последнего элемента это
решение: сходить ещё или "готово".

Таблицу никто не пишет. Её находит отбор: меняем случайные биты и оставляем, если стало не хуже.
"""
import random

import numpy as np

GUEST, THING, ACT = 0, 1, 2
EVENT_CODES = {GUEST: 1, THING: 2, ACT: 3}   # два бита на событие: 01, 10, 11
TABLE_ROWS = 32                              # 2 бита события, свой бит, 2 сигнала слева
TABLE_SIZE = TABLE_ROWS * 3                  # на выходе: новый бит и 2 сигнала вправо
MAX_FETCHES = 300   # хватает для обучения (ответы до 10). Экзамены сами поднимают его выше самого большого ответа.


def run_days(table, days, width, record=False):
    """Сколько раз сходил агент в каждом дне. days: список (события просмотра).

    С record=True ещё и что цепочка говорила после каждого события просмотра (её последний сигнал).
    """
    table = np.asarray(table, dtype=np.uint8).reshape(TABLE_ROWS, 3)
    count = len(days)
    state = np.zeros((count, width), dtype=np.uint8)

    def tick(codes):
        a = np.ones(count, dtype=np.uint8)   # первый элемент получает слева сигналы 1 и 1
        b = np.ones(count, dtype=np.uint8)
        for cell in range(width):
            row = (codes << 3) | (state[:, cell] << 2) | (a << 1) | b
            out = table[row]
            state[:, cell] = out[:, 0]
            a, b = out[:, 1], out[:, 2]
        return b

    length = max(map(len, days), default=0)
    said = np.zeros((count, length), dtype=np.uint8)
    for position in range(length):
        codes = np.array([EVENT_CODES[d[position]] if position < len(d) else 0 for d in days], dtype=np.uint8)
        active = codes > 0
        saved = state.copy()
        said[:, position] = tick(codes)
        state[~active] = saved[~active]       # день без события на этом шаге не меняется
    fetched = np.zeros(count, dtype=np.int64)
    running = np.ones(count, dtype=bool)
    for _ in range(MAX_FETCHES):
        saved = state.copy()
        decision = tick(np.full(count, EVENT_CODES[ACT], dtype=np.uint8)).astype(bool)
        state[~running] = saved[~running]
        go = running & decision
        fetched += go
        running = go
        if not running.any():
            break
    return (fetched, said) if record else fetched


def table_is_empty(days):
    """Что видно после каждого события: на столе пусто (1) или лежат лишние предметы (0)."""
    rows = []
    for events in days:
        balance, row = 0, []
        for event in events:
            balance += 1 if event == GUEST else -1
            row.append(int(balance >= 0))
        rows.append(row)
    return rows


def make_days(pairs, rng):
    days, right = [], []
    for things, guests in pairs:
        events = [GUEST] * guests + [THING] * things
        rng.shuffle(events)
        days.append(events)
        right.append(max(guests - things, 0))
    return days, np.array(right)


def score(table, days, right, width):
    return float((run_days(table, days, width) == right).mean())


def hand_written_table():
    """Заведомо решаемая проверка: таблица, которая точно считает, вписанная вручную.

    Обычный двоичный счётчик со знаком: гость прибавляет 1, предмет и поход вычитают 1.
    Сигнал a это перенос или заём. Сигнал b говорит "мой новый бит 0". У последнего элемента
    это значит "после похода счётчик не отрицательный", то есть до похода он был больше нуля.
    """
    table = np.zeros((TABLE_ROWS, 3), dtype=np.uint8)
    for code in (0, 1, 2, 3):
        for bit in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    row = (code << 3) | (bit << 2) | (a << 1) | b
                    if code == 1:
                        new, carry = bit ^ a, bit & a
                    elif code in (2, 3):
                        new, carry = bit ^ a, (1 - bit) & a
                    else:
                        new, carry = bit, 0
                    table[row] = (new, carry, 1 - new)
    return table.reshape(-1)
