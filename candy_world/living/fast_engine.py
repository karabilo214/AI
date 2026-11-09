"""Тот же движок цепочки, но переведённый Numba в машинный код. Логика повторяет engine.py до бита.

"Пересчётов на событие" здесь, как и в engine.py, только подсчёт изменившихся входов: таблица
читается у всех элементов, пропуска нет. Это верхняя граница экономии, а не реальная экономия.

Проверка равенства: check_fast_engine.py.
"""
import numpy as np
from numba import njit

from engine import Design


@njit(cache=True)
def _simulate(tables, code_bits, state_bits, n_right, n_left, skip, n_types, passes,
              events, lengths, width, fetch_code, max_fetches, said):
    days = events.shape[0]
    out_bits = state_bits + n_right + n_left
    fetched = np.zeros(days, dtype=np.int64)
    lookups = 0
    recomputed = 0
    total_events = 0
    state = np.zeros((width, state_bits), dtype=np.uint8)
    new_state = np.zeros((width, state_bits), dtype=np.uint8)
    last = np.empty((width, passes), dtype=np.int64)
    left_prev = np.zeros((width + 1, max(n_left, 1)), dtype=np.uint8)
    left_now = np.zeros((width + 1, max(n_left, 1)), dtype=np.uint8)
    right = np.zeros(max(n_right, 1), dtype=np.uint8)
    for day in range(days):
        state[:, :] = 0
        last[:, :] = -1
        step = 0
        fetching = False
        fetches = 0
        while True:
            if not fetching:
                if step >= lengths[day]:
                    fetching = True
                    continue
                code = events[day, step]
            else:
                if fetches >= max_fetches:
                    break
                code = fetch_code
            total_events += 1
            left_prev[:, :] = 0
            decision = 0
            for p in range(passes):
                for k in range(n_right):
                    right[k] = 1
                skip_a = 1
                skip_b = 1
                left_now[:, :] = 0
                for i in range(width):
                    row = code
                    for k in range(state_bits):
                        row = (row << 1) | state[i, k]
                    for k in range(n_right):
                        row = (row << 1) | right[k]
                    for k in range(n_left):
                        row = (row << 1) | left_prev[i + 1, k]
                    if skip:
                        row = (row << 1) | skip_a
                    lookups += 1
                    if row != last[i, p]:
                        recomputed += 1
                        last[i, p] = row
                    kind = 0
                    if n_types == 3:
                        kind = 0 if i == 0 else (2 if i == width - 1 else 1)
                    for k in range(state_bits):
                        new_state[i, k] = tables[kind, row, k]
                    for k in range(n_right):
                        right[k] = tables[kind, row, state_bits + k]
                    for k in range(n_left):
                        left_now[i, k] = tables[kind, row, state_bits + n_right + k]
                    skip_a = skip_b
                    skip_b = right[0]
                left_prev[:, :] = left_now
            for i in range(width):
                for k in range(state_bits):
                    state[i, k] = new_state[i, k]
            decision = right[1] if n_right > 1 else right[0]
            if not fetching:
                said[day, step] = decision
                step += 1
            else:
                if decision == 1:
                    fetched[day] += 1
                    fetches += 1
                else:
                    break
    return fetched, lookups, recomputed, total_events


def run_days(table_bits, design: Design, days, width, codes_of, max_fetches=300, record=False):
    """Тот же интерфейс, что у engine.run_days."""
    tables = np.asarray(table_bits, dtype=np.uint8).reshape(design.types, design.rows, design.out_bits)
    lengths = np.array([len(d) for d in days], dtype=np.int64)
    events = np.zeros((len(days), max(lengths.max(initial=0), 1)), dtype=np.int64)
    for i, d in enumerate(days):
        events[i, : len(d)] = d
    said = np.zeros(events.shape, dtype=np.uint8)
    fetched, lookups, recomputed, total = _simulate(
        tables, design.code_bits, design.state_bits, design.right, design.left, design.skip, design.types,
        design.passes, events, lengths, width, codes_of["fetch"], max_fetches, said)
    said = said[:, : lengths.max(initial=0)]
    cost = {"обращений на событие": lookups / max(total, 1), "пересчётов на событие": recomputed / max(total, 1)}
    return (fetched, said, cost) if record else (fetched, cost)
