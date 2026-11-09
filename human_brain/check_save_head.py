"""Проверка: голова после brain.save / Brain.load продолжает учиться ровно так же.

Голову учат зрению и слуху, на середине сохраняют. Дальше одно и то же
получают:
    1. оригинал;
    2. копия, загруженная в этом же процессе;
    3. копия, загруженная в отдельном процессе, где перед загрузкой
       tasks.apply выставил другие правила (награда x7, другой порог, другое
       усиление): загруженный мозг должен учиться по своим сохранённым
       правилам, а не по глобальным.
В конце каждый сохраняется, и сравнивается весь файл: все массивы нейронов и
связей, правила и состояния модулей (рост, память слуха, ассоциации, найм,
ожидание награды, любопытство и т. д.). При расхождении код выхода 1.
Запуск из папки human_brain: ../.venv/bin/python check_save_head.py
"""
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

import tasks
import vision_hearing as vh
from Brain import Brain
from Head import build_head

SEED = 10000


def make_order(seed=SEED):
    rnd = random.Random(seed)
    order = [pic for pic in vh.PICTURES for _ in range(20)]
    rnd.shuffle(order)
    return order


def feed(b, order, start, stop):
    for k in range(start, stop):
        pic = order[k]
        vh.show(b, pic, vh.WORDS[pic], picture_first=(k % 2 == 0))
        tasks.rest(b, 40)


def continue_from(path_in, path_out):
    """Загрузить, доучить вторую половину, сохранить."""
    order = make_order()
    c = Brain.load(path_in)
    np.random.seed(1)
    feed(c, order, 40, len(order))
    c.save(path_out)
    return c


def compare_files(a, b):
    """Список различий между двумя сохранениями (пусто = совпадают до бита)."""
    diff = []
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        keys = sorted(set(x.files) | set(y.files))
        for k in keys:
            if k not in x.files or k not in y.files:
                diff.append(f"{k}: есть только в одном файле")
            elif x[k].shape != y[k].shape or not np.array_equal(x[k], y[k]):
                diff.append(k)
    return diff


def child(path_in, path_out):
    p = dict(tasks.PARAMS_DEFAULT)
    p.update(reward_rate=7.0, threshold=-55.0, strengthen=0.3, max_weight=40.0)
    tasks.apply(p)             # другие глобальные правила в этом процессе
    continue_from(path_in, path_out)


def main():
    d = tempfile.mkdtemp()
    mid = os.path.join(d, "mid.npz")
    order = make_order()
    b = build_head(dict(tasks.PARAMS_DEFAULT), SEED)
    feed(b, order, 0, 40)
    b.save(mid)

    # 2. загрузка в этом же процессе (загружаем до того, как оригинал пойдёт дальше)
    out_same = os.path.join(d, "same.npz")
    c = continue_from(mid, out_same)
    print("после загрузки: память слуха", c.hearing is not None,
          "ассоциации", c.association is not None, "найм", c.recruit is not None)

    # 1. оригинал
    out_orig = os.path.join(d, "orig.npz")
    np.random.seed(1)
    feed(b, order, 40, len(order))
    b.save(out_orig)

    # 3. отдельный процесс с другими глобальными правилами
    out_child = os.path.join(d, "child.npz")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mid, out_child],
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        print("процесс с другими правилами упал")
        return 1

    print(f"нейронов {b.n_neurons} / {c.n_neurons}, связей {b.n_conn} / {c.n_conn}, "
          f"нейронов порядка {len(b.hearing.units)} / {len(c.hearing.units)}, "
          f"ассоциаций {len(b.association.links)} / {len(c.association.links)}")
    bad = 0
    for label, path in (("в этом процессе", out_same), ("в процессе с другими правилами", out_child)):
        diff = compare_files(out_orig, path)
        print(f"загрузка {label}: совпадает до бита: {not diff}"
              + ("" if not diff else " (различаются: " + ", ".join(diff[:10]) + ")"))
        bad += bool(diff)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        sys.exit(0)
    sys.exit(main())
