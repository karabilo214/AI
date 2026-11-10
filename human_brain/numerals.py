"""Цифры и выражение «a + b»: сможет ли мозг сам соединить чтение и счёт.

Мозг — сохранённый после всей цепочки (арифметика, чтение, правило мягкости,
rules.py). Новое:
    цифры 2-6 — картинки из 3 нейронов зрения (не совпадают с буквами), их
        детекторы нанимаются сами (Recruit), как детекторы букв (добавлено
        место ещё под 12 детекторов);
    мостик к счёту: все зрительные детекторы (буквы и цифры) подключены к
        накопителю как «увиденные» входы с весом 0 — что прибавляет символ,
        накопитель учит сам по ошибке после праздника; символ может только
        прибавлять порции (класть в регистр и прибавлять регистр могут только
        события мира). Это единственная врождённая часть мостика.
Ступень «цифры»: дни сложения, где часть групп гостей записана цифрой (дверь,
картинка «3» вместо дверь, гость, гость, гость); каждый второй день — прежние
задачи (арифметика, чтение, правило мягкости).
Проверка соединения навыков: выражение, которого не было ни разу: картинки
«a», «+» (новый значок), «b» и обычный вопрос «сколько?» — без двери, гостей
и карточки. Контроль: в ступени «цифры» картинка показывается рядом со
случайным числом гостей (её значение ничему не соответствует).

Запуск из папки human_brain:
    ../.venv/bin/python numerals.py [--seed 1000] [--days 1000] [--control]
"""
import argparse
import random
from itertools import combinations

import tasks
import addition as ad
import subtraction as sb
import division as dv
import reading as rd
import rules
import brain_store

DIGITS = (2, 3, 4, 5, 6)
EXTRA_DETECTORS = 12


def glyphs(seed):
    """Картинки цифр и значка «+»: наборы из 3 нейронов, не совпадающие с буквами."""
    used = set(rd.make_alphabet(seed).values())
    rnd = random.Random(seed + 50)
    free = [c for c in combinations(range(rd.N_VISION), 3) if c not in used]
    pics = rnd.sample(free, len(DIGITS) + 1)
    out = {f"D{d}": pic for d, pic in zip(DIGITS, pics)}
    out["PLUS"] = pics[-1]
    return out


def make_show(gl):
    def show(b, e):
        if e in gl:
            for i in gl[e]:
                b.stimulate(f"V{i}", 20.0)
            return True
        return False
    return show


def add_numerals(b):
    """Место под детекторы цифр и мостик «зрительные детекторы -> накопитель»."""
    n0 = len([n for n in b.names if n.startswith("L") and n[1:].isdigit()])
    for k in range(n0, n0 + EXTRA_DETECTORS):
        b.add_neuron(f"L{k}", region="буквы")
    if b.neuron_change is not None:
        b.neuron_change.make_same(("буквы",))
    acc = b.accumulator
    # выученные веса закреплены: учатся в young_plasticity раз медленнее новых
    # (то же отношение, что у молодых и закреплённых связей мозга)
    slow = 1.0 / b.growth.young_plasticity
    for n in acc.inputs:
        acc.plast[n] = slow
    dets = [n for n in b.names if n.startswith("L") and n[1:].isdigit()]
    for n in dets:
        if n not in acc.inputs:
            acc.inputs.append(n)
            acc.w[n] = 0.0
            acc.m[n] = 0.0
            acc.r[n] = 0.0
            acc.counts[n] = 0
            acc.plast[n] = 1.0
    acc.seen = sorted(set(acc.seen) | set(dets))


def digit_day(rng, control=False):
    """Сложение, где часть групп записана цифрой (2-6)."""
    ans = rng.randint(4, 10)
    total = 0
    out = []
    groups = ad.groups(ans, rng)       # дверь + гости, по группам
    k = 0
    while k < len(groups):
        assert groups[k] == "DOOR"
        j = k + 1
        while j < len(groups) and groups[j] == "GUEST":
            j += 1
        size = j - k - 1
        if size in DIGITS and rng.random() < 0.6:
            shown = rng.choice(DIGITS) if control else size
            out += ["DOOR", f"D{shown}"]
        else:
            out += groups[k:j]
        total += size
        k = j
    return ["CARD"] + out + ["ASK"], total, total


def expression_day(a, b_):
    return [f"D{a}", "PLUS", f"D{b_}", "ASK"], a + b_


def exam_expressions(b, gl):
    t = rd.quiet(b)
    show = make_show(gl)
    rows = []
    for a in DIGITS:
        for c in DIGITS:
            ev, ans = expression_day(a, c)
            more, done, _ = ad.day(t, ev, ans, learn=False, show=show)
            rows.append((a, c, ans, more if done else None))
    return rows


def run(p, seed, days=1000, control=False):
    path = brain_store.path_for(rules.rules_name(seed))
    if not brain_store.exists(path):
        rules.run(p, seed, 1000, save=True)
    b, rng, meta = brain_store.load(path)
    add_numerals(b)
    gl = glyphs(seed)
    show = make_show(gl)
    alphabet = rd.make_alphabet(seed)
    train = meta.get("train") or rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    rnd = random.Random(seed + 77)
    for k in range(days):
        if k % 2 == 0:
            ev, ans, truth = digit_day(rnd, control)
            ad.day(b, ev, ans, learn=True, truth=truth, show=show)
        elif k % 4 == 1:
            rules.say_word(b, rnd.choice(train), alphabet, True)
            tasks.rest(b, rd.REST)
        else:
            ev, ans, truth = rd.old_day(rng, k // 4)
            ad.day(b, ev, ans, learn=True, truth=truth)
    rows = exam_expressions(b, gl)
    res = {"выражение a + b": sum(r[2] == r[3] for r in rows) / len(rows)}
    new = rules.make_words(seed + 11, 30, (2, 3, 4, 5), exclude=train)
    res["чтение (правило)"] = rules.exam(b, new, alphabet)
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b, rows, gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    res, b, rows, gl = run(p, a.seed, a.days, a.control)
    acc = b.accumulator
    who = {b.names[w]: next((g for g, pic in gl.items() if sorted(b.neurons[f"V{i}"].index for i in pic) == sorted(ev)), "буква")
           for w, ev in b.recruit.owner.items()}
    print("веса символов в накопителе:", {f"{n}({who.get(n, '?')})": round(acc.w[n], 2) for n in acc.seen if abs(acc.w[n]) > 0.05 or who.get(n, "").startswith(("D", "P"))})
    print("выражения (a, b, верно, мозг):", rows)
    for k, v in res.items():
        if not k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление")):
            print(f"  {k:24s} {v:.2f}")
    ar = [k for k in res if k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление"))]
    print(f"  арифметика наборов 100%: {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")


if __name__ == "__main__":
    main()
