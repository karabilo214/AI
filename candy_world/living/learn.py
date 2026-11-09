"""Обучение цепочки наших элементов с памятью удач и неудач.

Агент: цепочка одинаковых элементов (1 бит памяти, общая таблица, сигналы a, b вправо).
Мир (ступень 1, сложение и вычитание): у меня есть конфеты, гости приходят группами минимум по 2
(перед группой "дверь"). После каждого события цепочка говорит "сейчас не хватает?", в конце ходит
за конфетами, пока не скажет "хватит". Верно: max(гостей - конфет, 0).

Как учится (способ "memory"):
  1. живёт пачку свежих дней на цепочках длиной 5-8;
  2. где менять: строки таблицы, которые работали в днях с ошибкой чаще, чем в верных днях;
  3. что пробовать: у каждой строки журнал "что здесь пробовал и чем кончилось" (хорошее и плохое).
     Сначала то, чего ещё не пробовал, потом то, что помогало; то, что вредило, не повторяет;
  4. проверяет правку на тех же днях, записывает исход в журнал в любом случае
     (и для нового варианта, и для старого), оставляет правку, если не хуже.
Для сравнения: "blame" - то же без журнала (вариант строки наугад), "random" - строка и вариант наугад.

    python learn.py --method memory --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from numba import njit

import task_count
from engine import Design

DOOR, GUEST, CANDY, FETCH = 0, 1, 2, 3
DESIGN = Design()                       # 2 бита кода, 1 бит памяти, 2 сигнала вправо
ROWS, OUT = DESIGN.rows, DESIGN.out_bits
WIDTHS = (5, 6, 7, 8)
STAGES = (3, 6, 10)
BATCH = 40
MAX_STEPS = 30000


@njit(cache=True)
def simulate(table, events, lengths, width, max_fetches, said, used):
    """Как fast_engine для базового устройства, но ещё отмечает, какие строки работали в каждом дне."""
    days = events.shape[0]
    fetched = np.zeros(days, dtype=np.int64)
    state = np.zeros(width, dtype=np.uint8)
    for day in range(days):
        state[:] = 0
        step, fetching, fetches = 0, False, 0
        while True:
            if not fetching:
                if step >= lengths[day]:
                    fetching = True
                    continue
                code = events[day, step]
            else:
                if fetches >= max_fetches:
                    break
                code = FETCH
            a, b = 1, 1
            for i in range(width):
                row = (code << 3) | (state[i] << 2) | (a << 1) | b
                used[day, row] = 1
                state[i] = table[row, 0]
                a, b = table[row, 1], table[row, 2]
            if not fetching:
                said[day, step] = b
                step += 1
            elif b == 1:
                fetched[day] += 1
                fetches += 1
            else:
                break
    return fetched


def make_days(limit, count, rng, low=0):
    """Случайные дни: конфет low..limit, гостей max(2, low)..limit."""
    return pair_days([(rng.randint(low, limit), rng.randint(max(2, low), max(limit, 2))) for _ in range(count)], rng)


def stage_days(limit, rng, repeat=3):
    """Полный набор сочетаний ступени (как в старом отборе), каждое 3 раза с разной разбивкой и порядком."""
    return pair_days([(c, g) for c in range(limit + 1) for g in range(2, max(limit, 2) + 1)] * repeat, rng)


def pair_days(pairs, rng):
    """События дня, верный ответ, верные ответы на "сейчас не хватает?" после каждого события."""
    days = []
    for candies, guests in pairs:
        sizes = [2] * rng.randint(1, guests // 2)
        for _ in range(guests - sum(sizes)):
            sizes[rng.randrange(len(sizes))] += 1
        blocks = [[CANDY] for _ in range(candies)] + [[DOOR] + [GUEST] * s for s in sizes]
        rng.shuffle(blocks)
        events = [e for blk in blocks for e in blk]
        balance, dense = 0, []
        for e in events:
            balance += (e == GUEST) - (e == CANDY)
            dense.append(int(balance >= 0))     # как в task_count: 1 = "не хватает или ровно"
        days.append((events, max(guests - candies, 0), dense))
    return days


def pack(days):
    lengths = np.array([len(d[0]) for d in days], dtype=np.int64)
    events = np.zeros((len(days), max(lengths.max(), 1)), dtype=np.int64)
    dense = np.full(events.shape, -1, dtype=np.int64)
    for i, (ev, _, de) in enumerate(days):
        events[i, :len(ev)] = ev
        dense[i, :len(de)] = de
    right = np.array([d[1] for d in days], dtype=np.int64)
    return events, lengths, dense, right


def live(table, packed, widths, max_fetches):
    """Ошибка по каждому дню (на всех длинах) и какие строки работали."""
    events, lengths, dense, right = packed
    errors = np.zeros(len(right))
    exact = np.ones(len(right), dtype=bool)
    used = np.zeros((len(right), ROWS), dtype=np.uint8)
    for w in widths:
        said = np.zeros(events.shape, dtype=np.int64)
        fetched = simulate(table, events, lengths, w, max_fetches, said, used)
        wrong_dense = ((said != dense) & (dense >= 0)).sum(1) / np.maximum(lengths, 1)
        errors += np.abs(fetched - right) + wrong_dense
        exact &= (fetched == right) & ~((said != dense) & (dense >= 0)).any(1)
    return errors / len(widths), exact, used


def exam(table, rng, limit, width, count=300, low=0):
    days = make_days(limit, count, rng, low)
    events, lengths, _, right = pack(days)
    said = np.zeros(events.shape, dtype=np.int64)
    used = np.zeros((count, ROWS), dtype=np.uint8)
    fetched = simulate(table, events, lengths, width, int(right.max()) * 2 + 10, said, used)
    return round(float((fetched == right).mean()), 3), round(float((right == 0).mean()), 3)


def check_engine():
    """Заведомо решаемая проверка: ручная таблица проходит, и расчёт совпадает с fast_engine до бита."""
    from fast_engine import run_days
    hand = task_count.hand_table().reshape(ROWS, OUT)
    rng = random.Random(3)
    days = make_days(10, 200, rng)
    packed = pack(days)
    err, exact, _ = live(hand, packed, WIDTHS, 50)
    fetched_old, said_old, _ = run_days(hand.reshape(-1), DESIGN, [d[0] for d in days], 6,
                                        {"fetch": FETCH}, 50, record=True)
    said = np.zeros(packed[0].shape, dtype=np.int64)
    fetched = simulate(hand, packed[0], packed[1], 6, 50, said, np.zeros((200, ROWS), dtype=np.uint8))
    same = (fetched == fetched_old).all() and all(
        (said[i, :len(d[0])] == said_old[i, :len(d[0])]).all() for i, d in enumerate(days))
    big = exam(hand, random.Random(4), 1000, 16)
    print(f"ручная таблица: точно на обучении {exact.mean():.3f}, ошибка {err.mean():.3f}, "
          f"до 1000 на цепочке 16: {big[0]}; совпадает с fast_engine: {bool(same)}")
    return exact.mean() == 1 and big[0] == 1 and same


TABU = 30   # неудача запрещает вариант, пока не принято 30 других правок: потом "теперь всё иначе"


def train(method, seed, max_steps, log_every=500):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    table = nrng.integers(0, 2, size=(ROWS, OUT), dtype=np.uint8)
    journal = np.zeros((ROWS, 1 << OUT))               # сумма исходов: >0 помогало, <0 вредило
    tried = np.zeros((ROWS, 1 << OUT), dtype=np.int64)
    bad_until = np.zeros((ROWS, 1 << OUT), dtype=np.int64)
    accepted, stage, curve, checks = 0, 0, [], 0
    value = lambda t, r: int(t[r, 0]) << 2 | int(t[r, 1]) << 1 | int(t[r, 2])
    packed = None
    for step in range(1, max_steps + 1):
        limit = STAGES[stage]
        fetch_cap = limit * 2 + 10
        if packed is None:
            packed = pack(stage_days(limit, rng))
            err, exact, used = live(table, packed, WIDTHS, fetch_cap)
            checks += 1
        if step % log_every == 0:
            row = {"шаг": step, "ступень": limit, "проверок": checks, "принято правок": accepted,
                   "ошибка": round(float(err.mean()), 3), "точно": round(float(exact.mean()), 3),
                   "до 10": exam(table, random.Random(9), 10, 8, 200)[0]}
            curve.append(row)
            print(row, flush=True)
        if exact.all():
            confirm = pack(make_days(limit, 200, rng))
            checks += 1
            if live(table, confirm, WIDTHS, fetch_cap)[1].all():
                curve.append({"шаг": step, "ступень пройдена": limit, "проверок": checks})
                print(curve[-1], flush=True)
                if stage == len(STAGES) - 1:
                    break
                stage += 1
            packed = None          # новый набор дней (следующая ступень или новая разбивка)
            continue
        wrong = ~exact
        if method == "random":
            rows, weights = np.arange(ROWS), np.ones(ROWS)
        else:
            blame = used[wrong].mean(0) - (used[~wrong].mean(0) if (~wrong).any() else 0)
            rows = np.nonzero(blame > 0)[0]
            weights = blame[rows]
            if len(rows) == 0:
                rows = np.nonzero(used[wrong].any(0))[0]
                weights = np.ones(len(rows))
        r = int(rng.choices(list(rows), weights=list(weights))[0])
        cur = value(table, r)
        options = [o for o in range(1 << OUT) if o != cur]
        if method == "memory":
            allowed = [o for o in options if bad_until[r, o] <= accepted]       # что вредило недавно, не повторяет
            fresh = [o for o in allowed if tried[r, o] == 0]                   # сначала то, чего не пробовал
            helped = sorted((o for o in allowed if tried[r, o] > 0 and journal[r, o] > 0),
                            key=lambda o: -journal[r, o] / tried[r, o])         # потом то, что помогало
            pool = fresh or helped[:2] or allowed or options
            new = rng.choice(pool)
        else:
            new = rng.choice(options)
        candidate = table.copy()
        candidate[r] = ((new >> 2) & 1, (new >> 1) & 1, new & 1)
        err2, exact2, used2 = live(candidate, packed, WIDTHS, fetch_cap)
        checks += 1
        delta = float(err.mean() - err2.mean())      # >0: стало лучше
        if method == "memory":                       # опыт записывается и при удаче, и при неудаче
            journal[r, new] += delta; tried[r, new] += 1
            journal[r, cur] -= delta; tried[r, cur] += 1
            if delta < 0:
                bad_until[r, new] = accepted + TABU
            journal *= 0.999
        if delta >= 0:
            table, err, exact, used = candidate, err2, exact2, used2
            accepted += 1
    return table, curve, checks, step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("memory", "blame", "random"), default="memory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--out", default="results_learn")
    a = p.parse_args()
    assert check_engine(), "заведомо решаемая проверка не прошла"
    started = time.time()
    table, curve, checks, steps = train(a.method, a.seed, a.steps)
    erng = random.Random(1000 + a.seed)
    res = {"способ": a.method, "seed": a.seed, "шагов": steps, "проверок дней": checks, "кривая": curve,
           "секунд": round(time.time() - started), "таблица": "".join(map(str, table.reshape(-1))),
           "экзамен (агент, всегда 0)": {
               "до 10, цепочка 8": exam(table, erng, 10, 8),
               "11-200, цепочка 16": exam(table, erng, 200, 16, low=11),
               "до 1000, цепочка 16": exam(table, erng, 1000, 16)}}
    print(a.method, a.seed, "шагов", steps, "проверок", checks, res["экзамен (агент, всегда 0)"], flush=True)
    out = Path(__file__).parent / a.out
    out.mkdir(exist_ok=True)
    (out / f"{a.method}_seed_{a.seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
