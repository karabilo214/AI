"""Проверка: обучение с сохранением и загрузкой посередине = обучение подряд (до бита).

Мозг вычитания учится 300 + 200 дней, сохраняется (brain_store), загружается и
учится ещё 200 дней; рядом тот же мозг учится 300 + 400 дней подряд. Сравниваются
файлы сохранения целиком. Код выхода 1 при расхождении.
Запуск из папки human_brain: ../.venv/bin/python check_store.py
"""
import os
import random
import sys
import tempfile

import tasks
import addition as ad
import subtraction as sb
import brain_store
from check_save_head import compare_files


def more_days(b, rng, n):
    for k in range(n):
        ev, ans, tr = sb.make_add_day(rng) if k % 2 else sb.make_sub_day(rng, rng.choice(("ASK", "ASK_LEFT")))
        ad.day(b, ev, ans, learn=True, truth=tr)


def main(seed=3):
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    d = tempfile.mkdtemp()
    tasks.apply(p)
    b = sb.make_brain(p, seed); rng = random.Random(seed)
    sb.train(b, rng, 300, 200)
    brain_store.save(b, rng, {}, os.path.join(d, "mid"))
    more_days(b, rng, 200)
    b.save(os.path.join(d, "straight.npz"))
    c, rng2, _ = brain_store.load(os.path.join(d, "mid"))
    more_days(c, rng2, 200)
    c.save(os.path.join(d, "loaded.npz"))
    diff = compare_files(os.path.join(d, "straight.npz"), os.path.join(d, "loaded.npz"))
    print("подряд и через сохранение совпадают до бита:", not diff, diff[:10])
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
