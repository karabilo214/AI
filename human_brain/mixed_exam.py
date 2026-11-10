"""Проверка обученного мозга вперемешку: сложение, вычитание, умножение, деление, чтение.

Мозг — сохранённый после всей цепочки (сложение -> вычитание -> умножение ->
деление -> чтение -> правило мягкости, rules.py). По 10 заданий каждого
действия, все вперемешку, подряд на одной копии мозга без шума и без обучения.
Все примеры не встречались в обучении:
    арифметика училась на числах до 10 (умножение — карточки 2-5), поэтому
    здесь числа 11-50 (умножение: гостей 11-50, или карточки 6-9);
    вычитание — 5 «сколько не хватило» и 5 «сколько останется», ответ не 0;
    чтение — слова не из учебных наборов, половина с парами «e+g», «e+h»,
    которых в обучении не было ни разу (мягкость после e обозначена «'»).

Запуск из папки human_brain:
    ../.venv/bin/python mixed_exam.py [--seed 1000] [--order 7]
"""
import argparse
import random

import tasks
import addition as ad
import subtraction as sb
import multiplication as mu
import division as dv
import reading as rd
import rules
import brain_store


def groups_of(events):
    sizes, cur = [], None
    for e in events:
        if e == "DOOR":
            if cur:
                sizes.append(cur)
            cur = 0
        elif e == "GUEST":
            cur += 1
    if cur:
        sizes.append(cur)
    return sizes


def render(tokens):
    return "".join(f"{x or '?'}{chr(39) if s else ''}" for x, s in tokens)


def make_tasks(seed, train_words):
    rng = random.Random(5000 + seed)
    items = []
    for _ in range(10):
        ev, ans = ad.make_day(rng, 50, 11)
        items.append(("сложение", " + ".join(map(str, groups_of(ev))) + " гостей, сколько принести",
                      ev, ans))
    for k in range(10):
        q = "ASK" if k < 5 else "ASK_LEFT"
        while True:       # только с ответом не ноль: «всегда 0» тут не угадывает
            ev, ans, _ = sb.make_sub_day(rng, q, 50, 11)
            if ans > 0:
                break
        c, g = ev.count("CANDY"), ev.count("GUEST")
        what = "не хватило" if q == "ASK" else "останется"
        items.append(("вычитание", f"конфет {c}, гостей {g}: сколько {what}", ev, ans))
    for k in range(10):
        if k < 5:
            ev, ans, _ = mu.make_mul_day(rng, 50, 11)
        else:
            ev, ans, _ = mu.make_mul_day(rng, card=(6, 9))
        items.append(("умножение", f"каждому по {ev.count('CARD')}, гостей {ev.count('GUEST')}",
                      ev, ans))
    for _ in range(10):
        ev, ans, _ = dv.make_div_day(rng, 50, 11)
        items.append(("деление", f"конфет {ev.count('CANDY')}, гостей {ev.count('GUEST')}: "
                                 f"сколько каждому", ev, ans))
    plain = rules.make_words(6000 + seed, 5, (3, 4, 5, 6), exclude=train_words)
    rule = rules.make_words(7000 + seed, 5, (3, 4, 5, 6), exclude=train_words + plain,
                            need=rules.HOLD_OUT, forbid=())
    for w in plain + rule:
        items.append(("чтение", w, None, render(rules.soft_tokens(w))))
    return items


def run_exam(seed, order=7):
    """Одна проверка: (строки таблицы, {действие: [верно, всего]})."""
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    path = brain_store.path_for(rules.rules_name(seed))
    if not brain_store.exists(path):
        rules.run(p, seed, 1000, save=True)
    b, _, _ = brain_store.load(path)
    alphabet = rd.make_alphabet(seed)
    train_words = rd.make_words(seed, 40, (2, 3, 4)) + rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    items = make_tasks(seed, train_words)
    random.Random(order).shuffle(items)
    t = rd.quiet(b)
    rows, score = [], {}
    for n, (kind, text, ev, want) in enumerate(items, 1):
        if kind == "чтение":
            got = render(rules.say_word(t, text, alphabet, False))
            tasks.rest(t, rd.REST)
            shown = f"«{text}»"
        else:
            more, done, _ = ad.day(t, ev, want, learn=False)
            got = str(more) if done else f"{more} (без «готово»)"
            shown = text
        ok = got == str(want)
        score.setdefault(kind, [0, 0])
        score[kind][0] += ok
        score[kind][1] += 1
        rows.append((n, kind, shown, str(want), got, "✓" if ok else "✗"))
    return rows, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--order", type=int, default=7, help="перемешивание заданий")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    path = brain_store.path_for(rules.rules_name(a.seed))
    if not brain_store.exists(path):
        rules.run(p, a.seed, 1000, save=True)
    b, _, meta = brain_store.load(path)
    alphabet = rd.make_alphabet(a.seed)
    train_words = rd.make_words(a.seed, 40, (2, 3, 4)) + rules.make_words(a.seed + 10, 40, (2, 3, 4, 5))
    items = make_tasks(a.seed, train_words)
    random.Random(a.order).shuffle(items)
    t = rd.quiet(b)
    rows, score = [], {}
    for n, (kind, text, ev, want) in enumerate(items, 1):
        if kind == "чтение":
            got = render(rules.say_word(t, text, alphabet, False))
            tasks.rest(t, rd.REST)
            shown = f"«{text}»"
        else:
            more, done, _ = ad.day(t, ev, want, learn=False)
            got = str(more) if done else f"{more} (без «готово»)"
            shown = text
        ok = got == str(want)
        score.setdefault(kind, [0, 0])
        score[kind][0] += ok
        score[kind][1] += 1
        rows.append((n, kind, shown, str(want), got, "✓" if ok else "✗"))
    print("| № | действие | пример | верно | мозг | |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")
    print()
    print("| действие | верно |")
    print("|---|---|")
    for kind in ("сложение", "вычитание", "умножение", "деление", "чтение"):
        print(f"| {kind} | {score[kind][0]}/{score[kind][1]} |")
    total = sum(v[0] for v in score.values())
    print(f"| **всего** | **{total}/{sum(v[1] for v in score.values())}** |")


if __name__ == "__main__":
    main()
