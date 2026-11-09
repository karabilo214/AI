"""Общий движок цепочки логических элементов. Свойства включаются настройками.

Каждый элемент: несколько бит памяти и таблица. Таблица по входам говорит: новые биты памяти и сигналы.
  code_bits  : сколько бит на код события
  state_bits : сколько бит памяти у элемента
  right      : сигналы, идущие вправо (от левого соседа к правому)
  left       : сигналы, идущие влево (от правого соседа к левому)
  skip       : быстрая дальняя связь: первый правый сигнал элемента i приходит ещё и к элементу i+2
  types      : 1 вид или 3 вида (начальный, средний, конечный)
  passes     : сколько проходов на одно событие (нужно, если сигналы идут в обе стороны)
Решение цепочки: второй правый сигнал последнего элемента (или первый, если правый сигнал один).

Заодно считаем, у скольких элементов изменился вход с прошлого раза ("пересчётов на событие").
Это ТОЛЬКО ПОДСЧЁТ: таблица всё равно читается у всех элементов. Число показывает, сколько
элементов пришлось бы считать, если бы неизменившиеся пропускались, то есть верхнюю границу
экономии, а не реальную экономию. Настоящего пропуска здесь нет.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Design:
    code_bits: int = 2
    state_bits: int = 1
    right: int = 2
    left: int = 0
    skip: bool = False
    types: int = 1
    passes: int = 1

    @property
    def in_bits(self):
        return self.code_bits + self.state_bits + self.right + self.left + int(self.skip)

    @property
    def rows(self):
        return 1 << self.in_bits

    @property
    def out_bits(self):
        return self.state_bits + self.right + self.left

    @property
    def size(self):
        """Бит во всех таблицах вместе."""
        return self.types * self.rows * self.out_bits


def type_of(position, width, design):
    if design.types == 1:
        return 0
    return 0 if position == 0 else (2 if position == width - 1 else 1)


def bits_to_int(columns):
    value = np.zeros(columns[0].shape, dtype=np.int64)
    for column in columns:
        value = (value << 1) | column.astype(np.int64)
    return value


class Chain:
    def __init__(self, table_bits, design, width, count):
        self.d = design
        self.width = width
        self.tables = np.asarray(table_bits, dtype=np.uint8).reshape(design.types, design.rows, design.out_bits)
        self.state = np.zeros((count, width, design.state_bits), dtype=np.uint8)
        self.last_input = np.full((count, width, design.passes), -1, dtype=np.int64)
        self.recomputed = 0
        self.lookups = 0

    def event(self, codes, active):
        """Одно событие у всех дней сразу. codes: код события (int). Возвращает решение цепочки."""
        d, count = self.d, len(codes)
        code_cols = [((codes >> (d.code_bits - 1 - k)) & 1).astype(np.uint8) for k in range(d.code_bits)]
        left_prev = np.zeros((count, self.width + 1, max(d.left, 1)), dtype=np.uint8)
        for p in range(d.passes):
            right = np.ones((count, d.right), dtype=np.uint8)
            skip_hist = [np.ones(count, dtype=np.uint8), np.ones(count, dtype=np.uint8)]
            new_state = np.zeros_like(self.state)
            left_now = np.zeros_like(left_prev)
            for i in range(self.width):
                cols = code_cols + [self.state[:, i, k] for k in range(d.state_bits)]
                cols += [right[:, k] for k in range(d.right)]
                cols += [left_prev[:, i + 1, k] for k in range(d.left)]
                if d.skip:
                    cols.append(skip_hist[0])
                row = bits_to_int(cols)
                changed = active & (row != self.last_input[:, i, p])
                self.recomputed += int(changed.sum())
                self.lookups += int(active.sum())
                self.last_input[active, i, p] = row[active]
                out = self.tables[type_of(i, self.width, d)][row]
                new_state[:, i] = out[:, : d.state_bits]
                right = out[:, d.state_bits : d.state_bits + d.right]
                if d.left:
                    left_now[:, i, : d.left] = out[:, d.state_bits + d.right :]
                skip_hist = [skip_hist[1], right[:, 0]]
            left_prev = left_now
        self.state[active] = new_state[active]
        return right[:, 1 if d.right > 1 else 0]


def run_days(table_bits, design, days, width, codes_of, max_fetches=300, record=False):
    """days: список последовательностей кодов событий фазы просмотра. codes_of['fetch']: код похода."""
    count = len(days)
    chain = Chain(table_bits, design, width, count)
    length = max(map(len, days), default=0)
    said = np.zeros((count, length), dtype=np.uint8)
    for position in range(length):
        codes = np.array([d[position] if position < len(d) else 0 for d in days], dtype=np.int64)
        active = np.array([position < len(d) for d in days])
        said[:, position] = chain.event(codes, active)
    fetched = np.zeros(count, dtype=np.int64)
    running = np.ones(count, dtype=bool)
    events = sum(map(len, days))
    for _ in range(max_fetches):
        decision = chain.event(np.full(count, codes_of["fetch"], dtype=np.int64), running).astype(bool)
        events += int(running.sum())
        go = running & decision
        fetched += go
        running = go
        if not running.any():
            break
    cost = {"обращений на событие": chain.lookups / max(events, 1),
            "пересчётов на событие": chain.recomputed / max(events, 1)}
    return (fetched, said, cost) if record else (fetched, cost)
