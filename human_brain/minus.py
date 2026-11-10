"""Знак «−», как в школе: знак учится рядом с операцией, которую мозг уже умеет.

Мозг — сохранённый после ступени «цифры» (numerals.py: цифры 2-6, «a + b»).
Новое:
    картинка знака «−» (3 нейрона зрения, не совпадает с буквами, цифрами и
        «+»); детектор для неё мозг нанимает сам (Recruit), как для букв;
    общая способность накопителя «символ задаёт роль следующего» (Accumulator,
        roles): символ, увиденный сразу после другого символа, прибавляет
        свои порции, умноженные на выученный множитель того символа. Она не про
        минус: множитель есть у каждого символа (буквы, цифры, «+», «−»), и
        какой символ какую роль задаёт, мозг учит сам по ошибке после праздника.
Ступень «минус»: дни вычитания (у меня c конфет, пришли g гостей, вопрос
«сколько не хватило?» или «сколько останется?»), где конфеты часто записаны:
картинка «−», затем цифра c, вместо c конфет по одной. Что значит «−», никто
не говорит: после праздника видно только, скольким не хватило или сколько
осталось. Остальные дни — прежние: цифры при сложении, чтение, арифметика.
Проверка: выражение, которого не было ни разу: «a», «−», «b», «сколько не
хватило?» — без карточки, конфет и гостей; ответ max(a - b, 0). Отдельно
пары a > b (ленивое «всегда 0» там 0). Второй вопрос к тому же выражению —
«сколько лишнего?» (ответ max(b - a, 0)): с одним «не хватило» любой уровень
ниже нуля даёт 0, и вычитание 6 при a > b не проверялось бы вовсе. Контроль: после «−» стоит случайная
цифра, не равная числу конфет (знак ничему не соответствует).

Запуск из папки human_brain:
    ../.venv/bin/python minus.py [--seed 1000] [--days 1000] [--control]
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
import numerals as nm
import brain_store

SILENCE = 1.0


def minus_glyph(seed):
    used = set(rd.make_alphabet(seed).values()) | set(nm.glyphs(seed).values())
    free = [c for c in combinations(range(rd.N_VISION), 3) if c not in used]
    return random.Random(seed + 51).choice(free)


def minus_day(rng, control=False):
    """Вычитание, где конфеты часто записаны «−, цифра»; гости — как в днях цифр."""
    c, g = rng.randint(0, 10), rng.randint(2, 10)
    ev = ["CARD"]
    if c in nm.DIGITS and rng.random() < 0.6:
        shown = rng.choice([d for d in nm.DIGITS if d != c]) if control else c
        ev += ["MINUS", f"D{shown}"]
    else:
        ev += ["CANDY"] * c
    groups = ad.groups(g, rng, at_least=1)
    k = 0
    while k < len(groups):
        j = k + 1
        while j < len(groups) and groups[j] == "GUEST":
            j += 1
        size = j - k - 1
        if size in nm.DIGITS and rng.random() < 0.6:
            ev += ["DOOR", f"D{size}"]
        else:
            ev += groups[k:j]
        k = j
    q = rng.choice(("ASK", "ASK_LEFT"))
    ans = max(g - c, 0) if q == "ASK" else max(c - g, 0)
    return ev + [q], ans, g - c


def exam_minus(b, show, q="ASK"):
    """q=ASK: «сколько не хватило?» = max(a - b, 0); ASK_LEFT: «сколько лишнего?» = max(b - a, 0)."""
    t = rd.quiet(b)
    rows = []
    for a in nm.DIGITS:
        for c in nm.DIGITS:
            ans = max(a - c, 0) if q == "ASK" else max(c - a, 0)
            more, done, _ = ad.day(t, [f"D{a}", "MINUS", f"D{c}", q], ans, learn=False, show=show)
            rows.append((a, c, ans, more if done else None))
    return rows


def run(p, seed, days=1000, control=False):
    path = brain_store.path_for(nm.numerals_name(seed))
    if not brain_store.exists(path):
        nm.run(p, seed, 1000, save=True)
    b, rng, meta = brain_store.load(path)
    acc = b.accumulator
    slow = 1.0 / b.growth.young_plasticity
    free = {n for n in acc.seen if b.neurons[n].index not in b.recruit.owner}
    for n in acc.inputs:          # всё выученное раньше закреплено; свободные детекторы — новые
        acc.plast[n] = 1.0 if n in free else slow
    acc.roles = True
    # молчание в ответ на «отвечай» — ошибка (см. addition.day): иначе тормоз к
    # «готово» растёт лавиной и ломает «сколько останется?»
    b.silence_penalty = SILENCE
    gl = dict(nm.glyphs(seed), MINUS=minus_glyph(seed))
    show = nm.make_show(gl)
    alphabet = rd.make_alphabet(seed)
    train = meta.get("train") or rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    rnd = random.Random(seed + 88)
    for k in range(days):
        if k % 4 == 0:
            ev, ans, truth = minus_day(rnd, control)
            ad.day(b, ev, ans, learn=True, truth=truth, show=show)
        elif k % 4 == 2:
            ev, ans, truth = nm.digit_day(rnd)
            ad.day(b, ev, ans, learn=True, truth=truth, show=show)
        elif k % 4 == 1:
            rules.say_word(b, rnd.choice(train), alphabet, True)
            tasks.rest(b, rd.REST)
        else:
            ev, ans, truth = rd.old_day(rng, k // 4)
            ad.day(b, ev, ans, learn=True, truth=truth)
    rows = exam_minus(b, show)
    big = [r for r in rows if r[0] > r[1]]
    res = {"выражение a − b (все 25)": sum(r[2] == r[3] for r in rows) / len(rows),
           "выражение a − b, a > b": sum(r[2] == r[3] for r in big) / len(big)}
    left = exam_minus(b, show, "ASK_LEFT")
    small = [r for r in left if r[0] < r[1]]
    res["выражение a − b, лишнее, a < b"] = sum(r[2] == r[3] for r in small) / len(small)
    both = [x[2] == x[3] and y[2] == y[3] for x, y in zip(rows, left)]
    res["выражение a − b, оба вопроса"] = sum(both) / len(both)
    plus = nm.exam_expressions(b, gl)
    res["выражение a + b"] = sum(r[2] == r[3] for r in plus) / len(plus)
    new = rules.make_words(seed + 11, 30, (2, 3, 4, 5), exclude=train)
    res["чтение (правило)"] = rules.exam(b, new, alphabet)
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b, rows, gl


def owner_name(b, pic):
    ev = sorted(b.neurons[f"V{i}"].index for i in pic)
    for w, e in b.recruit.owner.items():
        if sorted(e) == ev:
            return b.names[w]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    res, b, rows, gl = run(p, a.seed, a.days, a.control)
    acc = b.accumulator
    names = {owner_name(b, pic): g for g, pic in gl.items()}
    print("символы (вес, множитель следующего):",
          {g: (round(acc.w[n], 2), round(acc.g(n), 2)) for n, g in names.items() if n})
    print("выражения (a, b, верно, мозг):", rows)
    for k, v in res.items():
        if not k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление")):
            print(f"  {k:26s} {v:.2f}")
    ar = [k for k in res if k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление"))]
    print(f"  арифметика наборов 100%: {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")


if __name__ == "__main__":
    main()
