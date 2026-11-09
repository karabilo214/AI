"""Зрение и слух: связать увиденное и услышанное (исходные значения, обычная версия).

Зрение: 8 нейронов, картинка = 3 нейрона сразу. Слух: 6 нейронов-звуков,
слово = 2 звука с промежутком 2 шага. Связей между зрением и слухом в начале
нет: их должен вырастить рост (гипотезы -> проверка -> подтверждение).

Пары (картинки делят нейроны, слова делят звуки, есть слово наоборот):
    P1 = {0,1,2}  - «ab»      P2 = {3,4,5}  - «cd»
    P3 = {5,6,7}  - «de»      P4 = {1,4,7}  - «ba»
Обучение: каждая пара 20 раз вперемешку. Режимы порядка: «вперемешку» (в
половине показов сначала картинка, звуки через 1 и 3 шага; в половине сначала
слово, картинка через 1 шаг после второго звука), «картинка первой»,
«слово первым». Между показами 40 шагов.

Вспоминание (на копии мозга):
    картинка -> слово: зазвучали оба нужных звука, ни одного лишнего,
                        и в правильном порядке;
    слово -> картинка: загорелись все 3 нейрона картинки и ни одного лишнего.
Контроль: те же показы, но слово к картинке подбирается случайно.
Знакомость выключена (за 80 показов картинки приелись бы).

Запуск из папки human_brain: ../.venv/bin/python vision_hearing.py
"""
import copy
import random

import numpy as np

import tasks
from Brain import Brain
from Regions import add_standard_regions
from Association import Association
from Hearing import Hearing
from Neuron import Neuron

USE_ASSOCIATION = True   # правило «бывает вместе» между зрением и слухом
USE_HEARING = True       # память слуха: нейроны порядка звуков; зрение связывается с ними

PICTURES = {"P1": (0, 1, 2), "P2": (3, 4, 5), "P3": (5, 6, 7), "P4": (1, 4, 7)}
WORDS = {"P1": "ab", "P2": "cd", "P3": "de", "P4": "ba"}
SOUNDS = "abcdef"


def make_brain(p, seed):
    b = Brain()
    add_standard_regions(b)
    for i in range(8):
        b.add_neuron(f"V{i}", region="зрение")
    for s in SOUNDS:
        b.add_neuron(f"A{s}", region="слух")
    tasks.attach(b, p, seed, curiosity=False, neurons=False)
    if USE_HEARING:
        Hearing(b)
        Association(b, "зрение", "память слуха", scale=b.rules["Neuron"]["THRESHOLD"] - b.rules["Neuron"]["REST"] + 1.0)
    elif USE_ASSOCIATION:
        Association(b, "зрение", "слух")
    return b


def show(b, picture, word, picture_first):
    """Показ пары. Возвращает после показа (без паузы)."""
    events = {}
    if picture_first:
        events[0] = [f"V{i}" for i in PICTURES[picture]]
        events[1] = [f"A{word[0]}"]
        events[3] = [f"A{word[1]}"]
    else:
        events[0] = [f"A{word[0]}"]
        events[2] = [f"A{word[1]}"]
        events[3] = [f"V{i}" for i in PICTURES[picture]]
    for t in range(6):
        for n in events.get(t, []):
            b.stimulate(n, 20.0)
        b.step()


def recall_word(b, picture):
    """Картинка одна: какие звуки и когда загорелись."""
    t = copy.deepcopy(b)
    first = {}
    for n in (f"V{i}" for i in PICTURES[picture]):
        t.stimulate(n, 20.0)
    for step in range(10):
        for n in t.step():
            # только звуки (по области: имена сочетаний тоже начинаются на «A»)
            if t.region_of[t.neurons[n].index] == "слух" and n not in first:
                first[n] = step
    if USE_HEARING:
        return {n: s for n, s in first.items() if s <= 6}   # проговаривание длится дольше
    # первая волна: звуки, загоревшиеся в первые 3 шага (дальше возвращается эхо)
    return {n: s for n, s in first.items() if s <= 2}


def recall_picture(b, picture):
    """Слово одно: какие нейроны зрения загорелись."""
    t = copy.deepcopy(b)
    word = WORDS[picture]
    lit = set()
    for step in range(7 if USE_HEARING else 5):   # первая волна после второго звука
        if step == 0:
            t.stimulate(f"A{word[0]}", 20.0)
        if step == 2:
            t.stimulate(f"A{word[1]}", 20.0)
        # только нейроны зрения (сочетания «V3+V4» не в области «зрение»)
        lit |= {n for n in t.step() if t.region_of[t.neurons[n].index] == "зрение"}
    return lit


def run(p, seed, shuffled, repeats=20, mode="вперемешку"):
    rnd = random.Random(seed)
    b = make_brain(p, seed)
    order = [pic for pic in PICTURES for _ in range(repeats)]
    rnd.shuffle(order)
    for k, pic in enumerate(order):
        word = WORDS[rnd.choice(list(PICTURES))] if shuffled else WORDS[pic]
        first = {"вперемешку": k % 2 == 0, "картинка первой": True, "слово первым": False}[mode]
        show(b, pic, word, picture_first=first)
        tasks.rest(b, 40)
    res = {"слово верно": 0, "порядок верен": 0, "картинка верна": 0}
    for pic, word in WORDS.items():
        heard = recall_word(b, pic)
        want = {f"A{s}" for s in word}
        if set(heard) == want:
            res["слово верно"] += 1
            if heard[f"A{word[0]}"] < heard[f"A{word[1]}"]:
                res["порядок верен"] += 1
        if recall_picture(b, pic) == {f"V{i}" for i in PICTURES[pic]}:
            res["картинка верна"] += 1
    grown = [e for e in b.growth.log if e["kind"] in ("гипотеза подтверждена", "выросла связь")]
    rejected = [e for e in b.growth.log if e["kind"] == "гипотеза отвергнута"]
    return res, len(grown), len(rejected), b


def main(runs=10):
    global USE_ASSOCIATION, USE_HEARING
    import sys
    USE_ASSOCIATION = "--no-assoc" not in sys.argv
    USE_HEARING = USE_ASSOCIATION and "--no-hearing" not in sys.argv
    print("правило «бывает вместе»:", "есть" if USE_ASSOCIATION else "нет",
          "; память слуха:", "есть" if USE_HEARING else "нет")
    p = dict(tasks.PARAMS_DEFAULT)
    p["fam_on"] = False
    tasks.apply(p)
    for mode, shuffled in (("вперемешку", False), ("вперемешку", True),
                           ("картинка первой", False), ("картинка первой", True),
                           ("слово первым", False), ("слово первым", True)):
        total = {"слово верно": 0, "порядок верен": 0, "картинка верна": 0}
        g = r = 0
        for s in range(9000, 9000 + runs):
            res, ng, nr, b = run(p, s, shuffled, mode=mode)
            for k in total:
                total[k] += res[k]
            g += ng
            r += nr
        label = f"{mode}, " + ("контроль: слова случайные" if shuffled else "пары постоянные")
        print(f"{label:44s}: из {4 * runs} пар: картинка -> слово верно {total['слово верно']}, "
              f"в т.ч. порядок звуков верен {total['порядок верен']}; слово -> картинка верно "
              f"{total['картинка верна']}; подтверждено гипотез в среднем {g / runs:.1f}, "
              f"отвергнуто {r / runs:.1f}")
    # пример одного мозга
    res, ng, nr, b = run(p, 9000, False, mode="картинка первой")
    print("\nпример (запуск 9000, картинка первой):")
    for pic, word in WORDS.items():
        heard = recall_word(b, pic)
        print(f"  {pic} {PICTURES[pic]} -> звуки {sorted(heard, key=heard.get)} (нужно {list('A' + s for s in word)}); "
              f"слово «{word}» -> зрение {sorted(recall_picture(b, pic))}")


if __name__ == "__main__":
    main()
