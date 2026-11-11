"""Способ 1: не искать, а решать — SAT-решатель с добавлением контрпримеров (2026-11-11).

Задача add_steps («по N каждому», устройство task_multiply.DESIGN: 128 строк ×
4 выхода = 512 бит). Работа цепочки на дне записывается логическими условиями
над 512 битами таблицы: для каждого элемента на каждом событии выходы =
строка таблицы, выбранная его входами (код события известен, 4 бита входа —
переменные). Условия: на событиях «гость/предмет» решение = сигнал мира
(«хватает?»), на походах — первые R решений «идти», (R+1)-е — «хватит»
(R — верный ответ дня). Все 4 длины цепочки (7–10), одна таблица.
По контрпримерам: решить на нескольких днях, проверить на всех днях ступени
«до 6» быстрым движком, добавить неверные дни, повторить. Найдено —
экзамен «до 10» и «по 1-3, до 1000, цепочка 16».
Разные решения: случайные начальные дни и случайные «предпочтения» битов.

    python sat_search.py [--solutions 8] [--start 6]
"""
import argparse
import random
import time

import numpy as np
from pysat.solvers import Minisat22

import evolve as E
from fast_engine import run_days


class Enc:
    def __init__(self, design):
        self.d = design
        self.nv = design.rows * design.out_bits
        self.clauses = []
        self.TRUE = self.new()
        self.clauses.append([self.TRUE])

    def new(self):
        self.nv += 1
        return self.nv

    def T(self, row, k):
        return row * self.d.out_bits + k + 1

    def const(self, v):
        return self.TRUE if v else -self.TRUE

    def element(self, code, ins):
        """ins: 4 литерала (c, n, a, b). Возвращает 4 литерала выходов."""
        outs = [self.new() for _ in range(self.d.out_bits)]
        for combo in range(16):
            vals = [(combo >> (3 - k)) & 1 for k in range(4)]
            cond = [(x if v == 0 else -x) for x, v in zip(ins, vals)]
            row = (code << 4) | combo
            for k, o in enumerate(outs):
                t = self.T(row, k)
                self.clauses.append(cond + [-t, o])
                self.clauses.append(cond + [t, -o])
        return outs

    def day(self, events, said_target, answer, width, fetch_code):
        F = -self.TRUE
        state = [[F, F] for _ in range(width)]
        def step(code):
            a, b = self.TRUE, self.TRUE
            for i in range(width):
                c, n = state[i]
                o = self.element(code, [c, n, a, b])
                state[i] = [o[0], o[1]]
                a, b = o[2], o[3]
            return b
        for e, tgt in zip(events, said_target):
            dec = step(e)
            if tgt is not None:
                self.clauses.append([dec if tgt else -dec])
        for j in range(answer + 1):
            dec = step(fetch_code)
            self.clauses.append([dec if j < answer else -dec])


def check(table, design, widths, t, d):
    return E.fitness(table, design, widths, t, d)[1]


def failing_days(table, design, widths, t, d):
    days, right, target, mask = d
    bad = np.zeros(len(days), dtype=bool)
    for w in widths:
        got, said, _ = run_days(table, design, days, w, t.CODES, record=True)
        bad |= got != right
        bad |= ((said != target) & mask).any(1)
    return np.flatnonzero(bad)


EXAMS = None


def exam(table, design, widths, t, items):
    if EXAMS is not None and t.__name__ != "task_multiply":
        out = []
        r = random.Random(99)
        for name, w, draw in EXAMS:
            days, right, _ = t.make_days([draw(r) for _ in range(200)], r)
            got, _ = run_days(table, design, days, w, t.CODES, max_fetches=int(right.max()) * 2 + 10)
            out.append(round(float((got == right).mean()), 3))
        return tuple(out)
    d10 = E.data(t, items, 10, 1)
    e10 = check(table, design, widths, t, d10)
    r = random.Random(99)
    days, right, _ = t.make_days([(r.randint(0, 1000), r.randint(0, 1000), r.randint(1, 3)) for _ in range(200)], r)
    got, _ = run_days(table, design, days, 16, t.CODES, max_fetches=int(right.max()) * 2 + 10)
    return round(float(e10), 3), round(float((got == right).mean()), 3)


VERBOSE = True
ADD = 3          # сколько неверных дней добавлять за раунд
RANDOM_PHASES = False


def solve_once(seed, start, t, design, widths, items, d, time_limit=600, call_limit=120):
    rng = random.Random(seed)
    days, right, target, mask = d
    chosen = set(rng.sample(range(len(days)), start))
    t0 = time.time()
    rounds = 0
    while True:
        rounds += 1
        enc = Enc(design)
        for i in sorted(chosen):
            tg = [int(target[i, j]) if mask[i, j] else None for j in range(len(days[i]))]
            for w in widths:
                enc.day(days[i], tg, int(right[i]), w, t.CODES["fetch"])
        s = Minisat22(bootstrap_with=enc.clauses)
        if RANDOM_PHASES:   # случайные предпочтения битов: сильно тормозят решатель (2026-11-11)
            s.set_phases([v if rng.random() < 0.5 else -v for v in range(1, design.size + 1)])
        import threading
        timer = threading.Timer(call_limit, s.interrupt)
        timer.start()
        tc = time.time()
        ok = s.solve_limited(expect_interrupt=True)
        timer.cancel()
        if VERBOSE:
            print(f"  раунд {rounds}: дней {len(chosen)}, условий {len(enc.clauses)}, решатель {time.time() - tc:.1f} с, "
                  f"ответ {ok}", flush=True)
        if ok is None:
            return {"seed": seed, "итог": f"решатель не успел за {call_limit} с", "дней": len(chosen),
                    "раундов": rounds, "секунд": round(time.time() - t0)}
        if not ok:
            return {"seed": seed, "итог": "нет решения на выбранных днях", "дней": len(chosen),
                    "раундов": rounds, "секунд": round(time.time() - t0)}
        model = set(l for l in s.get_model() if l > 0)
        s.delete()
        table = np.array([1 if (k + 1) in model else 0 for k in range(design.size)], dtype=np.uint8)
        bad = failing_days(table, design, widths, t, d)
        if VERBOSE:
            print(f"    найдена таблица: неверных дней {len(bad)} из {len(days)}", flush=True)
        if len(bad) == 0:
            return {"seed": seed, "итог": "решено", "дней в условиях": len(chosen), "раундов": rounds,
                    "секунд": round(time.time() - t0), "переменных": enc.nv, "условий": len(enc.clauses),
                    "экзамен (до 10, до 1000)": exam(table, design, widths, t, items),
                    "таблица": "".join(map(str, table))}
        if time.time() - t0 > time_limit:
            return {"seed": seed, "итог": "не успел", "дней": len(chosen), "раундов": rounds}
        chosen |= set(rng.sample(list(bad), min(ADD, len(bad))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", type=int, default=8)
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--stage", type=int, default=6)
    ap.add_argument("--task", default="add_steps")
    ap.add_argument("--add", type=int, default=3)
    ap.add_argument("--call", type=int, default=120)
    ap.add_argument("--total", type=int, default=600)
    a = ap.parse_args()
    t, design, widths, items, exams = E.setup(a.task, "base")
    d = E.data(t, items, a.stage, 0)
    hand = t.hand_table()
    global EXAMS
    EXAMS = exams
    print("ручная таблица на данных:", check(hand, design, widths, t, d), "неверных дней", len(failing_days(hand, design, widths, t, d)), flush=True)
    for s in range(a.solutions):
        global ADD
        ADD = a.add
        r = solve_once(s, a.start, t, design, widths, items, d, time_limit=a.total, call_limit=a.call)
        print({k: v for k, v in r.items() if k != "таблица"}, flush=True)


if __name__ == "__main__":
    main()
