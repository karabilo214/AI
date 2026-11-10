"""Правило чтения: мягкость после «e» (вторая ступень чтения, тот же мозг).

Правило (мозгу его не говорят): буква, стоящая после «e», звучит мягко — её
звук плюс признак мягкости (отдельный звуковой нейрон S_soft). Слова не
кончаются на «e» (иначе правилу не к чему применяться).
Обучение в такт: буква показывается каждые 3 шага, её звук (и признак
мягкости, если нужен) слышен через 2 шага после показа — в тот же шаг, когда
мозг сам произносит букву при чтении. Пары «e+g» и «e+h» в обучении не
встречаются никогда (HOLD_OUT).
Экзамен (слово без звука): верно, если для каждой буквы прозвучал её звук и
признак мягкости ровно там, где он нужен, в том же шаге. Наборы: новые слова,
слова с невиданными парами «e+g», «e+h» (проверка правила: таблица пар их не
знает), чтение первой ступени (без правила, reading.py), вся арифметика.

Чем мозг может это выучить: рост связей «X, потом через gap шагов Y»
(Growth) — признак мягкости всегда звучит через одно и то же время после
детектора «e», какая бы буква ни шла следом. Для этого слух больше не
защищён от роста: связи к звукам — это и есть речь. Буквы, вопросы и выходы
заготовок по-прежнему защищены.

Мозг берётся из сохранения после первой ступени чтения (reading.py);
каждый второй день — правило, остальные — прежняя арифметика.
Запуск из папки human_brain:
    ../.venv/bin/python rules.py [--seed 900] [--days 1000]
"""
import argparse
import random

import tasks
import addition as ad
import subtraction as sb
import division as dv
import reading as rd
import brain_store

GAP = 3            # шагов между буквами
SAY = 2            # через сколько шагов после показа буквы звучит её звук
HOLD_OUT = ("eg", "eh")


def soft_tokens(word):
    """Ожидаемое чтение: список (буква, мягкая?)."""
    return [(ch, k > 0 and word[k - 1] == "e") for k, ch in enumerate(word)]


def make_words(seed, n, lengths, exclude=(), need=None, forbid=HOLD_OUT):
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        w = "".join(rnd.choice(rd.LETTERS) for _ in range(rnd.choice(lengths)))
        if w.endswith("e") or w in exclude or w in out:
            continue
        if forbid and any(f in w for f in forbid):
            continue
        if need and not any(f in w for f in need):
            continue
        out.append(w)
    return out


def add_soft(b):
    """Признак мягкости и право роста к звукам (речь)."""
    b.add_neuron("S_soft", region="слух")
    g = b.growth
    g.sensory_regions = [r for r in g.sensory_regions if r != "слух"]
    # к звукам речи связи растут только от детекторов букв и от звуков: от
    # отдельного нейрона зрения (общего для нескольких букв) — неверный звук
    g.source_regions = {"слух": ["буквы", "слух"]}


def say_word(b, word, alphabet, with_sound, soft_at=None):
    """Слово в такт. Возвращает прочитанное: список (буква или None, мягкость) по буквам."""
    events = {}
    for k, ch in enumerate(word):
        events.setdefault(k * GAP, []).extend(f"V{i}" for i in alphabet[ch])
        if with_sound:
            events.setdefault(k * GAP + SAY, []).append(f"S_{ch}")
            soft = (k in soft_at) if soft_at is not None else (k > 0 and word[k - 1] == "e")
            if soft:
                events[k * GAP + SAY].append("S_soft")
    heard = []
    for t in range(len(word) * GAP + 4):
        for n in events.get(t, []):
            b.stimulate(n, 20.0)
        fired = b.step()
        base = sorted(n[2:] for n in fired if n.startswith("S_") and n != "S_soft")
        soft = "S_soft" in fired
        if base or soft:
            heard.append(("".join(base) or None, soft))
    return heard


def exam(b, words, alphabet):
    t = rd.quiet(b)
    ok = 0
    for w in words:
        ok += say_word(t, w, alphabet, False) == soft_tokens(w)
        tasks.rest(t, rd.REST)
    return ok / len(words)


def run(p, seed, days=1000, control=False):
    path = brain_store.path_for(rd.read_name(seed))
    if not brain_store.exists(path):
        rd.run(p, seed, 800, save=True)
    b, rng, _ = brain_store.load(path)
    add_soft(b)
    alphabet = rd.make_alphabet(seed)
    stage1 = rd.make_words(seed + 1, 30, (2, 3, 4, 5), exclude=rd.make_words(seed, 40, (2, 3, 4)))
    train = make_words(seed + 10, 40, (2, 3, 4, 5))
    new = make_words(seed + 11, 30, (2, 3, 4, 5), exclude=train)
    rule = make_words(seed + 12, 20, (2, 3, 4, 5), exclude=train, need=HOLD_OUT, forbid=())
    rnd = random.Random(seed + 13)
    for k in range(days):
        if k % 2 == 0:
            w = rnd.choice(train)
            soft_at = None
            if control:   # контроль: столько же мягких букв, но в случайных местах
                n_soft = sum(s for _, s in soft_tokens(w))
                soft_at = set(rnd.sample(range(len(w)), n_soft))
            say_word(b, w, alphabet, True, soft_at)
            tasks.rest(b, rd.REST)
        else:
            ev, ans, truth = rd.old_day(rng, k // 2)
            ad.day(b, ev, ans, learn=True, truth=truth)
    res = {"правило: учебные слова": exam(b, train, alphabet),
           "правило: новые слова": exam(b, new, alphabet),
           "правило: невиданные пары e+g, e+h": exam(b, rule, alphabet)}
    t = rd.quiet(b)
    s1 = 0
    for w in stage1:      # первая ступень: старый такт; считаются основные звуки букв
        s1 += "".join(x for x in rd.show_word(t, w, alphabet, None) if x != "soft") == w
        tasks.rest(t, rd.REST)
    res["чтение первой ступени (основные звуки)"] = s1 / len(stage1)
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b, alphabet, rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--days", type=int, default=1000)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    res, b, alphabet, rule = run(p, a.seed, a.days)
    for k, v in res.items():
        if not k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление")):
            print(f"  {k:36s} {v:.2f}")
    ar = [k for k in res if k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление"))]
    print(f"  арифметика: наборов 100% {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")
    t = rd.quiet(b)
    print("примеры:", [(w, say_word(t, w, alphabet, False)) for w in rule[:3]])
    soft_in = [(c.source, c.myelin.delay(), round(c.synapse.weight, 1)) for c in b.connections
               if c.target == "S_soft" and not c.inhibitory]
    print("связи к S_soft:", soft_in[:10])


if __name__ == "__main__":
    main()
