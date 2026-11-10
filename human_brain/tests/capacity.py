"""Ёмкость памяти «картинка — слово»: сколько пар запоминается.

Сборка как в vision_hearing.py (зрение, слух, рост, «бывает вместе», память
слуха, знакомость выключена, исходные параметры), но размеры и пары задаются:
    зрение  n_vision нейронов, картинка = 3 случайных нейрона (картинки разные,
            но могут делить нейроны);
    слух    n_sounds звуков, слово = 2 разных звука по порядку (слова разные).
Каждая пара показывается repeats раз вперемешку (в половине показов сначала
картинка, в половине сначала слово), между показами 40 шагов.
Вспоминание на копии, по каждой паре:
    картинка -> слово: зазвучали ровно оба звука слова, в правильном порядке;
    слово -> картинка: загорелись ровно 3 нейрона картинки.
Контроль: слово к каждому показу выбирается случайно из всех слов (уровень
угадывания). Ложное вспоминание: 20 картинок, которых не показывали; сколько
из них назвали какое-то слово целиком (зазвучали 2 звука одного из слов).
--no-winner: память слуха без выбора победителя (как было).
Проверка самого замера: --check повторяет vision_hearing (8 зрения, 6 звуков,
исходные 4 пары), должно получиться как там (40/40).

Запуск из корня проекта:
    .venv/bin/python -m human_brain.tests.capacity N [запусков] [--control] [--check]
"""
import os as _os
import sys as _sys
import copy
import random
import sys
from itertools import combinations, permutations

from human_brain.stages import tasks
from human_brain.stages import vision_hearing as vh
from human_brain.core.Brain import Brain
from human_brain.core.Regions import add_standard_regions
from human_brain.parts.Association import Association
from human_brain.parts.Hearing import Hearing

N_VISION = 32
N_SOUNDS = 16


def make_pairs(n_pairs, rnd, n_vision, n_sounds):
    pics = rnd.sample(list(combinations(range(n_vision), 3)), n_pairs)
    words = rnd.sample(list(permutations(range(n_sounds), 2)), n_pairs)
    return list(zip(pics, words))


WINNER = True


def make_brain(p, seed, n_vision, n_sounds):
    b = Brain()
    add_standard_regions(b)
    for i in range(n_vision):
        b.add_neuron(f"V{i}", region="зрение")
    for s in range(n_sounds):
        b.add_neuron(f"A{s}", region="слух")
    tasks.attach(b, p, seed, curiosity=False, neurons=False)
    Hearing(b, winner=WINNER)
    reach = b.rules["Neuron"]["THRESHOLD"] - b.rules["Neuron"]["REST"]
    Association(b, "зрение", "память слуха", scale=reach + 1.0)
    return b


def show(b, pic, word, picture_first):
    events = {}
    if picture_first:
        events = {0: [f"V{i}" for i in pic], 1: [f"A{word[0]}"], 3: [f"A{word[1]}"]}
    else:
        events = {0: [f"A{word[0]}"], 2: [f"A{word[1]}"], 3: [f"V{i}" for i in pic]}
    for t in range(6):
        for n in events.get(t, []):
            b.stimulate(n, 20.0)
        b.step()


def recall_word(b, pic):
    t = copy.deepcopy(b)
    for i in pic:
        t.stimulate(f"V{i}", 20.0)
    first = {}
    for step in range(10):
        for n in t.step():
            if t.region_of[t.neurons[n].index] == "слух" and n not in first:
                first[n] = step
    return {n: s for n, s in first.items() if s <= 6}


def recall_picture(b, word):
    t = copy.deepcopy(b)
    lit = set()
    for step in range(7):
        if step == 0:
            t.stimulate(f"A{word[0]}", 20.0)
        if step == 2:
            t.stimulate(f"A{word[1]}", 20.0)
        lit |= {n for n in t.step() if t.region_of[t.neurons[n].index] == "зрение"}
    return lit


def run(p, seed, n_pairs, control=False, check=False, repeats=20):
    rnd = random.Random(seed)
    if check:
        nv, ns = 8, 6
        idx = {s: k for k, s in enumerate(vh.SOUNDS)}
        pairs = [(vh.PICTURES[k], tuple(idx[c] for c in vh.WORDS[k])) for k in vh.PICTURES]
    else:
        nv, ns = N_VISION, N_SOUNDS
        pairs = make_pairs(n_pairs, rnd, nv, ns)
    b = make_brain(p, seed, nv, ns)
    order = [k for k in range(len(pairs)) for _ in range(repeats)]
    rnd.shuffle(order)
    words = [w for _, w in pairs]
    for j, k in enumerate(order):
        pic, word = pairs[k]
        if control:
            word = rnd.choice(words)
        show(b, pic, word, picture_first=(j % 2 == 0))
        tasks.rest(b, 40)
    word_ok = pic_ok = 0
    for pic, word in pairs:
        heard = recall_word(b, pic)
        if (set(heard) == {f"A{word[0]}", f"A{word[1]}"}
                and heard[f"A{word[0]}"] < heard[f"A{word[1]}"]):
            word_ok += 1
        pic_ok += recall_picture(b, word) == {f"V{i}" for i in pic}
    shown = {pic for pic, _ in pairs}
    novel = [c for c in combinations(range(nv), 3) if c not in shown]
    known_words = [{f"A{w[0]}", f"A{w[1]}"} for _, w in pairs]
    false = 0
    for pic in rnd.sample(novel, 20):
        heard = set(recall_word(b, pic))
        false += any(w <= heard for w in known_words)
    return (word_ok, pic_ok, len(pairs), b.time, len(b.hearing.units), len(b.association.links),
            false)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_pairs = int(args[0]) if args else 4
    runs = int(args[1]) if len(args) > 1 else 5
    control, check = "--control" in sys.argv, "--check" in sys.argv
    global WINNER
    WINNER = "--no-winner" not in sys.argv
    p = dict(tasks.PARAMS_DEFAULT)
    p["fam_on"] = False
    tasks.apply(p)
    res = [run(p, s, n_pairs, control, check) for s in range(9000, 9000 + runs)]
    total = sum(r[2] for r in res)
    label = ("проверка (как vision_hearing)" if check else f"{n_pairs} пар"
             + (", контроль" if control else "")) + ("" if WINNER else ", без выбора победителя")
    print(f"{label}: картинка -> слово {sum(r[0] for r in res)}/{total}, "
          f"слово -> картинка {sum(r[1] for r in res)}/{total}; "
          f"новая картинка назвала слово {sum(r[6] for r in res)}/{20 * runs}; "
          f"шагов {res[0][3]}, нейронов порядка {sum(r[4] for r in res) / runs:.0f}, "
          f"ассоциаций {sum(r[5] for r in res) / runs:.0f} (в среднем)", flush=True)


if __name__ == "__main__":
    main()
