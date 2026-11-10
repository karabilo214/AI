"""Чтение: мозг, который умеет считать, учится читать целыми словами.

Буквы: LETTERS (8), буква = картинка из 3 нейронов зрения (16 нейронов, наборы
разные, могут делить нейроны). Звуки: по нейрону слуха на букву (S_a, S_b...).
День чтения: слово показывается по буквам (буква каждые 2 шага), потом, после
паузы, звучит целиком (звук каждые 2 шага). Какая буква какой звук, мозгу не
говорят: звучание не в такт с буквами. Наград нет.

Что есть в мозге для этого (новое — органы чувств и место под детекторы):
    детекторы букв: найм памяти (Recruit.py) — каждое новое событие зрения
        (нейроны, загоревшиеся вместе) получает свой детектор совпадения;
    правило «бывает вместе» (Association) между детекторами букв и слухом с окном,
        покрывающим весь день слова: каждая буква «встречается» со всеми
        звуками своего слова, но с правильным звуком — в каждом слове, где она
        есть; сила связи = доля появлений;
    звуки — группа соревнования (выбор победителя, kernel): если звук не
        услышан, а буквы зовут, звучит самый поддержанный большинством.
Зрение и слух — органы чувств: рост в них связей не проводит, сочетаний из
них нет. Порог и стекание у них не подстраиваются.

Порядок: мозг после деления (division.py, выбор нейронами) + зрение и слух;
каждый второй день чтение, остальные — прежняя арифметика по очереди.
Экзамен на копии: слово показывается по буквам без звука; верно, если
прозвучали ровно звуки букв в том же порядке. Новые слова длиной 2-5 (не
было в обучении) и 6-8 (учили на 2-4); вся арифметика. Контроль: звучащее
слово случайное (не то, что написано) — читать не должно получаться.

Запуск из папки human_brain:
    ../.venv/bin/python reading.py [--seed 900] [--days 800] [--control]
"""
import argparse
import random

import tasks
import addition as ad
import subtraction as sb
import multiplication as mu
import division as dv
import brain_store
from Association import Association
from Recruit import Recruit

LETTERS = "abcdefgh"
N_VISION = 16
LETTER_GAP = 2
SOUND_GAP = 2
PAUSE = 4
REST = 40
SOUND_POOL = 7          # номер группы соревнования звуков
N_DETECTORS = 12        # свободных нейронов под детекторы букв


def make_alphabet(seed):
    rnd = random.Random(seed)
    from itertools import combinations
    pics = rnd.sample(list(combinations(range(N_VISION), 3)), len(LETTERS))
    return {ch: pic for ch, pic in zip(LETTERS, pics)}


def make_words(seed, n, lengths, exclude=()):
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        w = "".join(rnd.choice(LETTERS) for _ in range(rnd.choice(lengths)))
        if w not in exclude and w not in out:
            out.append(w)
    return out


def add_senses(b, p):
    """Зрение и слух к мозгу арифметики; правило «бывает вместе»; звуки соревнуются."""
    b.add_region("зрение")
    b.add_region("слух")
    for i in range(N_VISION):
        b.add_neuron(f"V{i}", region="зрение")
    for ch in LETTERS:
        n = b.add_neuron(f"S_{ch}", region="слух")
        b.pool[n.index] = SOUND_POOL
    if b.neuron_change is not None:
        b.neuron_change.make_same(("зрение", "слух"))
    g = b.growth
    g.sensory_regions = sorted(set(g.sensory_regions or []) | {"зрение", "слух"})
    # детекторы букв: найм памяти (Recruit) — новое событие зрения (нейроны,
    # загоревшиеся вместе) получает свой детектор совпадения; иначе буква из
    # нейронов, общих с другими буквами, не имеет своего признака
    b.add_region("буквы", learn=0.0, perceptual=True)
    for k in range(N_DETECTORS):
        b.add_neuron(f"L{k}", region="буквы")
    if b.neuron_change is not None:
        b.neuron_change.make_same(("буквы",))
    g.sensory_regions = sorted(set(g.sensory_regions) | {"буквы"})
    Recruit(b, inputs=["зрение"], memory="буквы")
    span = 2 * (LETTER_GAP * 8 + PAUSE + SOUND_GAP * 8)
    scale = 0.5 * (b.rules["Neuron"]["THRESHOLD"] - b.rules["Neuron"]["REST"])
    Association(b, "буквы", "слух", window=span, scale=p.get("assoc_scale", scale),
                perceptual_counts=True)


def show_word(b, word, alphabet, spoken):
    """Слово по буквам, потом (если spoken не None) звучит spoken. Возвращает звуки по шагам."""
    heard = []
    t = 0
    events = {}
    for k, ch in enumerate(word):
        events.setdefault(k * LETTER_GAP, []).extend(f"V{i}" for i in alphabet[ch])
    end = (len(word) - 1) * LETTER_GAP + PAUSE
    if spoken is not None:
        for k, ch in enumerate(spoken):
            events.setdefault(end + k * SOUND_GAP, []).append(f"S_{ch}")
        end += len(spoken) * SOUND_GAP
    for t in range(end + 6):
        for n in events.get(t, []):
            b.stimulate(n, 20.0)
        fired = b.step()
        heard += sorted(n[2:] for n in fired if n.startswith("S_"))
    return heard


def read_aloud(b, word, alphabet):
    """Экзамен: слово без звука, какие звуки прозвучали (по порядку)."""
    return "".join(show_word(b, word, alphabet, None))


def reading_day(b, rnd, words, alphabet, control):
    w = rnd.choice(words)
    spoken = rnd.choice(words) if control else w
    show_word(b, w, alphabet, spoken)
    tasks.rest(b, REST)


def old_day(rng, k):
    kind = k % 5
    if kind == 4:
        return dv.make_div_day(rng)
    return dv.old_day(rng, kind)


def quiet(b):
    """Копия для экзамена: без шума и без обучения ассоциаций (экзамен мозг не меняет)."""
    import copy
    t = copy.deepcopy(b)
    t.curiosity = None
    t.association.observe = lambda: None
    return t


def exam_reading(b, words, alphabet):
    t = quiet(b)
    ok = 0
    for w in words:
        ok += read_aloud(t, w, alphabet) == w
        tasks.rest(t, REST)
    return ok / len(words)


def run(p, seed, days=800, control=False):
    path = brain_store.path_for(dv.div_name(seed))
    if not brain_store.exists(path):
        dv.run(dict(p, choice="neural"), seed, 3000, save=True)
    b, rng, _ = brain_store.load(path)
    add_senses(b, p)
    alphabet = make_alphabet(seed)
    train = make_words(seed, 40, (2, 3, 4))
    new = make_words(seed + 1, 30, (2, 3, 4, 5), exclude=train)
    long = make_words(seed + 2, 10, (6, 7, 8), exclude=train)
    rnd = random.Random(seed + 7)
    for k in range(days):
        if k % 2 == 0:
            reading_day(b, rnd, train, alphabet, control)
        else:
            ev, ans, truth = old_day(rng, k // 2)
            ad.day(b, ev, ans, learn=True, truth=truth)
    res = {"чтение: учебные слова": exam_reading(b, train, alphabet),
           "чтение: новые слова 2-5": exam_reading(b, new, alphabet),
           "чтение: длинные 6-8": exam_reading(b, long, alphabet)}
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b, alphabet, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--days", type=int, default=800)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    res, b, alphabet, new = run(p, a.seed, a.days, a.control)
    for k, v in res.items():
        print(f"  {k:26s} {v:.2f}")
    t = quiet(b)
    print("примеры (написано -> прочитано):",
          [(w, read_aloud(t, w, alphabet)) for w in new[:6]])


if __name__ == "__main__":
    main()
