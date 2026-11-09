"""Проверки восстановления эпизодов (сон) и управляемого забывания.

Всё на сохранённых исходных значениях (tasks.PARAMS_DEFAULT), обычная версия.

1. Сон против забывания: задачи из двух блоков, второй блок длинный
   (150 попыток), чтобы сильнее мешать первому. Со сном: каждые 10 попыток
   второго блока мозг повторяет до 3 забытых эпизодов.
2. Мир изменился: 60 попыток «S1 -> L», затем 100 попыток «S1 -> R».
   Переучился ли (с забыванием и без, со сном и без).
3. Ненадёжная награда: задача A, 10% наград перепутаны. Забывание не должно
   срабатывать на случайных ошибках.
4. Важное не удаляется: у выученной важной связи сила падает ниже порога
   «мёртвой»; удалится ли она.

Запуск из папки human_brain: ../.venv/bin/python memory_tests.py
"""
import random

import numpy as np

import tasks
from Episodes import Episodes


def with_params(**kw):
    p = dict(tasks.PARAMS_DEFAULT)
    p.update(kw)
    tasks.apply(p)
    return p


EXTRA = {   # первый блок: S1 -> L и S2 -> R; второй: сочетание S1+S2 -> L (или R)
    "AB_then_L": [[(("S1",), "L"), (("S2",), "R")], [(("S1", "S2"), "L")]],
    "AB_then_R": [[(("S1",), "L"), (("S2",), "R")], [(("S1", "S2"), "R")]],
}


def two_blocks(p, name, seed, sleep, n2=150):
    """Задача из двух блоков; второй длиннее. Возвращает (все верны, первый блок верен)."""
    if name in EXTRA:
        blocks = EXTRA[name]
        b = tasks.make_reward_brain(p, seed)
    elif name in tasks.CONTEXT_TASKS:
        blocks = tasks.CONTEXT_TASKS[name]
        b = tasks.make_context_brain(p, seed)
    else:
        blocks = [[part] for part in tasks.TASKS[name]]
        b = tasks.make_reward_brain(p, seed)
    E = Episodes(b) if sleep else None
    tasks.learn_block(b, blocks[0], 60)
    for k in range(n2):
        tasks.learn_block(b, [blocks[1][k % len(blocks[1])]], 1)
        if E is not None and k % 10 == 9:
            E.sleep(3)
    first = all(tasks.exam(b, i) == g for i, g in blocks[0])
    every = first and all(tasks.exam(b, i) == g for i, g in blocks[1])
    return every, first


def world_change(p, seed, sleep, n2=100):
    """Возвращает (переучился, попыток до 3 верных подряд или None,
    удалён ли путь S1 -> R, устаревших эпизодов)."""
    b = tasks.make_reward_brain(p, seed)
    E = Episodes(b) if sleep else None
    tasks.learn(b, ("S1",), "L", 60)
    path_gone = not any(c.source == "S1" and c.target == "R" for c in b.connections)
    streak, switched = 0, None
    for k in range(n2):
        hit = tasks.learn(b, ("S1",), "R", 1)
        streak = streak + 1 if hit else 0
        if switched is None and streak == 3:
            switched = k + 1
        if E is not None and k % 10 == 9:
            E.sleep(3)
    outdated = sum(ep.outdated for ep in E.items) if E else None
    return tasks.exam(b, ("S1",)) == "R", switched, path_gone, outdated


def noisy_reward(p, seed):
    rnd = random.Random(seed)
    b = tasks.make_reward_brain(p, seed)
    for inputs, g in tasks.TASKS["A"]:
        for _ in range(60):
            out = tasks.trial(b, inputs)
            r = +1 if out == g else -1 if out in ("L", "R", "оба") else 0
            if rnd.random() < 0.1:
                r = -r
            b.reward(r)
            tasks.rest(b)
    return all(tasks.exam(b, i) == g for i, g in tasks.TASKS["A"])


def important_survives(p, seed):
    b = tasks.make_reward_brain(p, seed)
    tasks.learn(b, ("S1",), "L", 60)
    c = max((c for c in b.connections if c.source == "S1" and c.target == "L"),
            key=lambda c: c.synapse.weight)
    imp = c.synapse.importance
    c.synapse.weight = 0.0          # сила упала ниже порога «мёртвой»
    tasks.rest(b, b.dead_patience + 50)
    return c.alive, imp


def main(runs=20):
    seeds = range(7000, 7000 + runs)
    print("1. сон против забывания (второй блок 150 попыток):")
    for name, extra in (("A", {}), ("B", {}), ("ctx_reverse", {}),
                        ("ctx_reverse", {"ctx_on": False}), ("AB_then_L", {}), ("AB_then_R", {})):
        label = name + (" без контекстных ветвей" if extra else "")
        for sleep in (False, True):
            p = with_params(**extra)
            res = [two_blocks(p, name, s, sleep) for s in seeds]
            if not sleep:
                learned = [two_blocks(p, name, s, False, n2=0)[1] for s in seeds]
                print(f"   {label:34s} первый блок выучен до второго: {sum(learned)}/{runs}")
            print(f"   {label:34s} сон {sleep!s:5s}: все верны {sum(r[0] for r in res)}/{runs}, "
                  f"первый блок помнит {sum(r[1] for r in res)}/{runs}")
    print("2. мир изменился (S1: L -> R, 100 попыток), и сколько попыток до переучивания:")
    for forget in (False, True):
        for sleep in (False, True):
            p = with_params(forget_on=forget)
            res = [world_change(p, s, sleep) for s in seeds]
            sw = [r[1] for r in res if r[1] is not None]
            extra = f", устаревших эпизодов в среднем {np.mean([r[3] for r in res]):.1f}" if sleep else ""
            print(f"   забывание {forget!s:5s} сон {sleep!s:5s}: переучился {sum(r[0] for r in res)}/{runs}, "
                  f"попыток до переучивания в среднем {np.mean(sw):.1f}, "
                  f"путь S1->R удалён ещё в первой задаче {sum(r[2] for r in res)}/{runs}{extra}")
    print("3. ненадёжная награда (A, 10% перепутаны):")
    for forget in (False, True):
        p = with_params(forget_on=forget)
        print(f"   забывание {forget!s:5s}: все верны {sum(noisy_reward(p, s) for s in seeds)}/{runs}")
    print("4. важная связь с силой ниже порога «мёртвой»:")
    for forget in (False, True):
        p = with_params(forget_on=forget)
        res = [important_survives(p, s) for s in seeds]
        print(f"   забывание {forget!s:5s}: осталась в {sum(r[0] for r in res)}/{runs}, "
              f"важность в среднем {np.mean([r[1] for r in res]):.1f}")


if __name__ == "__main__":
    main()
