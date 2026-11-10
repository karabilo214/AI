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

Запуск из корня проекта:
    .venv/bin/python -m human_brain.tests.mixed_exam [--seed 1000] [--order 7]
"""
import os as _os
import sys as _sys
import argparse
import random

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.stages import multiplication as mu
from human_brain.stages import division as dv
from human_brain.stages import reading as rd
from human_brain.stages import rules
from human_brain.core import brain_store


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


def write_report(seed, order, path):
    """Отчёт о мозге в файл (Markdown): как учился, проверка вперемешку, широкий экзамен."""
    import datetime
    rows, score = run_exam(seed, order)
    b, _, meta = brain_store.load(brain_store.path_for(rules.rules_name(seed)))
    alphabet = rd.make_alphabet(seed)
    train_words = rd.make_words(seed, 40, (2, 3, 4)) + rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    new = rules.make_words(seed + 11, 30, (2, 3, 4, 5), exclude=train_words)
    rule = rules.make_words(seed + 12, 20, (2, 3, 4, 5), exclude=train_words,
                            need=rules.HOLD_OUT, forbid=())
    wide = {k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()}
    wide["чтение: новые слова (с правилом)"] = rules.exam(b, new, alphabet)
    wide["чтение: невиданные пары e+g, e+h"] = rules.exam(b, rule, alphabet)
    u = b.units
    chosen = {q: f"{u.names[k]} ({u.cands[k].variant}, регистр {u.cands[k].reg_event})"
              for q, k in u.chosen.items()}
    acc = b.accumulator
    grown = sum(1 for e in b.growth.log if e["kind"] == "вырос нейрон")
    L = []
    L.append(f"# Отчёт: мозг {seed}")
    L.append("")
    L.append(f"Дата: {datetime.date.today()}. Мозг обучен с нуля и проверен без обучения "
             f"(копия без шума). Файл мозга: `human_brain/brains/{rules.rules_name(seed)}.npz`.")
    L.append("")
    L.append("## Как учился")
    L.append("")
    L.append("| ступень | дней | чему учился | примеры в обучении |")
    L.append("|---|---|---|---|")
    L.append("| сложение | 300 | «сколько принести» (гости группами) | числа до 10 |")
    L.append("| вычитание | 2000 | «сколько не хватило», «сколько останется»; каждый второй день сложение | числа до 10, часть сочетаний скрыта |")
    L.append("| умножение | 2000 | «каждому по N»; каждый второй день прежнее | N 2-5, гостей до 10 |")
    L.append("| деление | 3000 | «сколько каждому»; заготовку выбирает сеть, закрепляет правило после 10 удач; каждый второй день прежнее | конфет до 10 |")
    L.append("| чтение | 800 | буква -> звук из целых слов (звук не в такт); каждый второй день арифметика | 40 слов из 2-4 букв |")
    L.append("| правило чтения | 1000 | «после e буква мягкая»; каждый второй день арифметика | 40 слов, без пар e+g, e+h |")
    L.append("")
    L.append(f"Врождённое: накопитель со знаком и регистром, защёлка вопроса, 14 заготовок "
             f"(«по одной» / «по регистру»), найм детекторов букв. Выросло само: "
             f"{grown} нейронов-сочетаний (рост), {len(b.recruit.owner)} детекторов букв "
             f"(найм), {len(b.connections)} живых связей. "
             f"Выбор для деления: {chosen or 'не закреплён'}. Накопитель: гость прибавляет регистр "
             f"(m={acc.m['GUEST']:.2f}), карточка кладётся в регистр (r={acc.r['CARD']:.2f}), "
             f"конфета {acc.w['CANDY']:.2f}.")
    L.append("")
    L.append("## Проверка вперемешку: 50 заданий, все вне обучения")
    L.append("")
    L.append("Числа 11-50 (обучение до 10), в умножении и карточки 6-9; вычитание только с ответом "
             "не 0; слова не из учебных, половина с невиданными парами e+g, e+h (мягкость — «'»).")
    L.append("")
    L.append("| № | действие | пример | верно | мозг | |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append("| " + " | ".join(map(str, r)) + " |")
    L.append("")
    L.append("| действие | верно |")
    L.append("|---|---|")
    for kind in ("сложение", "вычитание", "умножение", "деление", "чтение"):
        L.append(f"| {kind} | {score[kind][0]}/{score[kind][1]} |")
    total = sum(v[0] for v in score.values())
    L.append(f"| **всего** | **{total}/{sum(v[1] for v in score.values())}** |")
    L.append("")
    L.append("## Широкий экзамен (те же навыки, больше примеров)")
    L.append("")
    L.append("| набор | доля верных |")
    L.append("|---|---|")
    for k, v in wide.items():
        L.append(f"| {k} | {v:.2f} |")
    L.append("")
    open(path, "w").write("\n".join(L))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--order", type=int, default=7, help="перемешивание заданий")
    ap.add_argument("--report", default="", help="записать отчёт в этот файл (Markdown)")
    a = ap.parse_args()
    if a.report:
        total = write_report(a.seed, a.order, a.report)
        print(f"отчёт: {a.report}, верно {total}/50")
        return
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
