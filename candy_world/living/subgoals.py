"""Оценка программ по подцелям: каждый полезный кусочек умения должен давать преимущество сам.

Подцели (не привязаны к одному ручному решению):
  память   - по состоянию счётчиков R и S в конце дня однозначно восстанавливается число с карточки N
             (доля дней, чьё состояние не совпадает с состоянием дня с другим N);
  рост N   - при тех же гостях (1-3, предметов нет) больший N даёт больший ответ (доля верных пар);
  рост гостей - при том же N больше гостей дают больший ответ (доля верных пар);
  точно    - доля дней с верным ответом; ошибка - средняя |ответ - верный|.
Нулевая точка - пустая программа.

    python subgoals.py        # проверка на ручной лесенке к умножению
"""
import copy
import random

import compose as C
import ladder as L

PROBE_N, PROBE_GUESTS = (1, 2, 3), (0, 1, 2, 3)


def value(counter, cap=500):
    c, n = copy.deepcopy(counter), 0
    while n < cap and c.event(C.CHECK):
        n += 1
    return n


def run_day(program, events, words, width=12, limit=6000):
    """Ответ и состояние (R, S) в конце дня; (-1, None) при превышении бюджета."""
    m = L.Machine(width, limit, words)
    try:
        for e in events:
            m.run(program[e])
        state = tuple(value(m.counters[x]) for x in ("R", "S"))
        trips = 0
        while m.call("C", C.CHECK):
            trips += 1
        return trips, state
    except (C.Budget, RecursionError):
        return -1, None


def memory(program, days, words):
    seen = {}
    rows = []
    for events, _ in days:
        need = events.count("tick")
        _, state = run_day(program, events, words)
        rows.append((need, state))
        if state is not None:
            seen.setdefault(state, set()).add(need)
    return sum(s is not None and len(seen[s]) == 1 for _, s in rows) / len(rows)


def growth(program, words):
    ans = {(n, g): run_day(program, ["tick"] * n + ["guest"] * g, words)[0] for n in PROBE_N for g in PROBE_GUESTS}
    by_n = [ans[n, g] < ans[n + 1, g] for g in PROBE_GUESTS[1:] for n in PROBE_N[:-1]]
    by_g = [ans[n, g] < ans[n, g + 1] for n in PROBE_N for g in PROBE_GUESTS[:-1]]
    return sum(by_n) / len(by_n), sum(by_g) / len(by_g)


def parts(program, days, words):
    err, exact = L.score(program, days, words)
    gn, gg = growth(program, words)
    return {"память": memory(program, days, words), "рост N": gn, "рост гостей": gg, "точно": exact, "ошибка": -err}


def total(p):
    """Сумма подцелей; точность весит больше, ошибка разводит равные."""
    return p["память"] + p["рост N"] + p["рост гостей"] + 2 * p["точно"] - 0.01 * p["ошибка"]


T = {"tick": [("inc", "R")], "thing": [("dec", "C")]}
PATH = [
    ("пусто", {"tick": [], "guest": [], "thing": []}),
    ("карточка в R; гость переливает R в C", {**T, "guest": [("while", "R", [("inc", "C")])]}),
    ("+ копия в S", {**T, "guest": [("while", "R", [("inc", "C"), ("inc", "S")])]}),
    ("+ R+1", {**T, "guest": [("while", "R", [("inc", "C"), ("inc", "S")]), ("inc", "R")]}),
    ("+ вернуть S в R", {**T, "guest": [("while", "R", [("inc", "C"), ("inc", "S")]), ("inc", "R"),
                                        ("while", "S", [("inc", "R")])]}),
    ("+ S+1: умножение", {**T, "guest": [("while", "R", [("inc", "C"), ("inc", "S")]), ("inc", "R"),
                                         ("while", "S", [("inc", "R")]), ("inc", "S")]}),
]
TRAPS = [
    ("тупик: гость +1 C", {"tick": [], "guest": [("inc", "C")], "thing": []}),
    ("тупик: гость +1 C, предмет -1 C", {"tick": [], "guest": [("inc", "C")], "thing": [("dec", "C")]}),
    ("тупик: гость +2 C, предмет -1 C", {"tick": [], "guest": [("inc", "C"), ("inc", "C")], "thing": [("dec", "C")]}),
    ("тупик: карточка в C, гость ничего", {"tick": [("inc", "C")], "guest": [], "thing": [("dec", "C")]}),
    ("тупик: карточка в R и S, гость переливает R", {"tick": [("inc", "R"), ("inc", "S")], "thing": [("dec", "C")],
                                                    "guest": [("while", "R", [("inc", "C")])]}),
]


def main():
    days = L.make_days("multiply", 60, 6, random.Random(500))
    fresh = L.make_days("multiply", 60, 6, random.Random(501))
    print(f"{'программа':45} память  рост N  рост гост  точно  ошибка   ИТОГ  (свежие дни: итог)")
    for name, prog in PATH + TRAPS:
        p, q = parts(prog, days, {}), parts(prog, fresh, {})
        print(f"{name:45} {p['память']:6.2f} {p['рост N']:7.2f} {p['рост гостей']:10.2f} {p['точно']:6.2f}"
              f" {p['ошибка']:7.2f} {total(p):6.2f}  ({total(q):.2f})")


if __name__ == "__main__":
    main()
