"""Общая проверка вперемешку: всё, что выучено, по 10 заданий в разброс.

Основной мозг — сохранённый после всей цепочки (сложение -> вычитание ->
умножение -> деление -> чтение -> правило мягкости -> цифры -> знак «−»,
minus.py). По 10 заданий каждого навыка, все вперемешку, подряд на одной
копии без шума и без обучения, все вне обучения:
    арифметика — числа 11-50 (обучение до 10), умножение — и карточки 6-9;
    вычитание — 5 «не хватило» и 5 «останется», ответ не 0;
    чтение — слова не из учебных, половина с невиданными парами e+g, e+h;
    «a + b» и «a − b» цифрами-картинками (выражений в обучении не было);
        у «a − b» вопрос «не хватило» или «лишнее», цифра 4 после «−» в
        обучении не встречалась никогда.
Если мозг прошёл ступени памяти и логики (memory.py, logic_stage.py), в
тот же перемешанный список входят: по 10 вопросов «что видел?» и «что
слышал?» (держит оба факта) и по 10 проб И, ИЛИ, НЕ над фактами из памяти.
--separate: ещё и прежние отдельные мозги логики и памяти (для сравнения).

Запуск из корня проекта:
    .venv/bin/python -m human_brain.tests.full_exam --seed 1000          # один мозг, строка JSON
    .venv/bin/python -m human_brain.tests.full_exam --report FILE --seeds 1000-1023,2000
"""
import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import reading as rd
from human_brain.stages import rules
from human_brain.stages import numerals as nm
from human_brain.stages import minus as mn
from human_brain.stages import logic as lg
from human_brain.stages import hold
from human_brain.core import brain_store
from human_brain.tests.mixed_exam import make_tasks, render

MAIN_KINDS = ("сложение", "вычитание", "умножение", "деление", "чтение", "a + b", "a − b")


def main_brain(p, seed):
    """Основной мозг после последней ступени (логика), иначе после «−»."""
    from human_brain.stages import logic_stage as ls
    from human_brain.stages import memory as me
    path = brain_store.path_for(ls.logic_name(seed))
    if brain_store.exists(path):
        b, _, meta = brain_store.load(path)
        return b, meta, "логика"
    path = brain_store.path_for(me.memory_name(seed))
    if brain_store.exists(path):
        b, _, meta = brain_store.load(path)
        ls.me_settings(b)
        return b, meta, "память"
    path = brain_store.path_for(mn.minus_name(seed))
    if not brain_store.exists(path):
        mn.run(p, seed, 2000, save=True)
    b, _, meta = brain_store.load(path)
    return b, meta, ""


def main_exam(p, seed, order):
    b, meta, full = main_brain(p, seed)
    alphabet = rd.make_alphabet(seed)
    train_words = rd.make_words(seed, 40, (2, 3, 4)) + rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    items = make_tasks(seed, train_words)
    rng = random.Random(8000 + seed)
    for _ in range(10):
        a, c = rng.choice(nm.DIGITS), rng.choice(nm.DIGITS)
        items.append(("a + b", f"{a} + {c}", [f"D{a}", "PLUS", f"D{c}", "ASK"], a + c))
    for k in range(10):
        q = "ASK" if k < 5 else "ASK_LEFT"
        a, c = rng.choice(nm.DIGITS), rng.choice(nm.DIGITS)
        ans = max(a - c, 0) if q == "ASK" else max(c - a, 0)
        what = "не хватило" if q == "ASK" else "лишнее"
        items.append(("a − b", f"{a} − {c}, сколько {what}", [f"D{a}", "MINUS", f"D{c}", q], ans))
    gl = dict(nm.glyphs(seed), MINUS=mn.minus_glyph(seed))
    from human_brain.stages import memory as me, logic_stage as ls
    if full:
        # память (и логика, если мозг её прошёл) — в том же мозге
        mrng = random.Random(8500 + seed)
        for q, what in (("Q_SEE", "что видел"), ("Q_HEAR", "что слышал")):
            for _ in range(10):
                while True:
                    shown, qq, want = me.make_memory_task(mrng)
                    if qq == q:
                        break
                txt = ", ".join(("видел " if k == "see" else "слышал ") + str(w) for k, w, _ in shown)
                items.append((f"память: {what}", f"{txt}; {what}?", ("memory", shown, q), want))
        signs = ls.sign_glyphs(seed)
        for op in (ls.OPS if full == "логика" else ()):
            for _ in range(10):
                a, c = mrng.choice(ls.cases(op))
                want = "YES" if ls.OPS[op](a, c) else "NO"
                txt = f"видел {'2' if a else '3'}, слышал {'c' if c else 'd'}"
                items.append((f"логика {ls.NAMES[op]}", f"{txt}; {ls.NAMES[op]}", ("logic", a, c, op), want))
    random.Random(order).shuffle(items)
    show = nm.make_show(gl)
    t = rd.quiet(b)
    rows, score = [], {}
    for kind, text, ev, want in items:
        if isinstance(ev, tuple) and ev[0] == "memory":
            got = str(me.memory_day(t, ev[1], ev[2], gl, False))
            me.quiet(t, 20, False)
        elif isinstance(ev, tuple) and ev[0] == "logic":
            got = str(ls.logic_day(t, ev[1], ev[2], ev[3], gl, signs, random.Random(len(rows)), False))
            me.quiet(t, 20, False)
        elif kind == "чтение":
            got = render(rules.say_word(t, text, alphabet, False))
            tasks.rest(t, rd.REST)
            text = f"«{text}»"
        else:
            more, done, _ = ad.day(t, ev, want, learn=False, show=show)
            got = str(more) if done else f"{more} (без «готово»)"
        ok = got == str(want)
        score.setdefault(kind, [0, 0])
        score[kind][0] += ok
        score[kind][1] += 1
        rows.append((kind, text, str(want), got, ok))
    return rows, score


def logic_exam(p, seed, order):
    by_op, _, b = lg.run_mix(p, seed, 6000, pictures=True)
    gl = lg.make_glyphs(seed)
    rng = random.Random(9000 + seed)
    items = []
    names = {"AND": "И", "OR": "ИЛИ", "NOT": "НЕ"}
    for op in lg.MIX_OPS:
        f, _ = lg.OPS[op]
        for _ in range(10):
            a, c = rng.choice(lg.cases(op))
            items.append((op, a, c, "YES" if f(a, c) else "NO"))
    random.Random(order).shuffle(items)
    import copy
    t = copy.deepcopy(b)
    t.curiosity = None
    rows, score = [], {}
    for op, a, c, want in items:
        out = lg.trial(t, a, c, op, gl)
        lg.rest(t)
        kind = f"логика {names[op]}"
        ok = out == want
        score.setdefault(kind, [0, 0])
        score[kind][0] += ok
        score[kind][1] += 1
        text = f"A={'да' if a else 'нет'}" + (f", B={'да' if c else 'нет'}" if op != "NOT" else "")
        rows.append((kind, text, want, str(out), ok))
    return rows, score


def memory_exam(p, seed, order):
    import copy
    b, facts, qs, acts = hold.multi_train(p, seed, [2, 2], 50, 6000)
    rng = random.Random(9500 + seed)
    items = [(d, k) for d in (0, 1) for k in range(10)]
    random.Random(order).shuffle(items)
    t = copy.deepcopy(b)
    t.curiosity = None
    rows, score = [], {}
    dim = ("цвет", "форма")
    for d, _ in items:
        order2 = [0, 1]
        rng.shuffle(order2)
        shown, state = [], {}
        for e in order2:
            v = rng.randrange(2)
            state[e] = v
            shown.append((facts[e][v], rng.randint(2, 10)))
        out = hold.multi_trial(t, shown, qs[d], 50, acts)
        want = acts[d][state[d]]
        for _ in range(20):
            t.step()
        kind = f"память: {dim[d]}"
        ok = out == want
        score.setdefault(kind, [0, 0])
        score[kind][0] += ok
        score[kind][1] += 1
        text = f"цвет {state[0]}, форма {state[1]}; спрошено: {dim[d]}"
        rows.append((kind, text, want, str(out), ok))
    return rows, score


def one(seed, order=7, separate=False):
    """separate: ещё и прежние отдельные мозги логики и памяти (для сравнения)."""
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    out = {"seed": seed, "логика": {"rows": [], "score": {}}, "память": {"rows": [], "score": {}}}
    parts = (("основной", main_exam),) + ((("логика", logic_exam), ("память", memory_exam)) if separate else ())
    for name, fn in parts:
        p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
        tasks.apply(p)
        rows, score = fn(p, seed, order)
        out[name] = {"rows": rows, "score": score}
    return out


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def report(seeds, path, workers=6):
    def run(s):
        r = subprocess.run([sys.executable, "-m", "human_brain.tests.full_exam", "--seed", str(s)],
                           capture_output=True, text=True)
        return json.loads(r.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, seeds))
    import datetime
    L = [f"# Общая проверка вперемешку", "",
         f"Дата: {datetime.date.today()}. Мозги: {', '.join(map(str, seeds))}. По 10 заданий "
         "каждого выученного навыка в разброс, на копии без шума и без обучения, все вне обучения. "
         "Все навыки — в одном мозге: вся цепочка от сложения до логики, каждая ступень "
         "училась поверх предыдущей.", ""]
    kinds = []
    for r in res:
        for part in ("основной", "логика", "память"):
            for k in r[part]["score"]:
                if k not in kinds:
                    kinds.append(k)
    L.append("## Итог по навыкам (сумма по всем мозгам)")
    L.append("")
    L.append("| навык | верно | мозгов на 10/10 |")
    L.append("|---|---|---|")
    for k in kinds:
        got = tot = full = 0
        for r in res:
            for part in ("основной", "логика", "память"):
                if k in r[part]["score"]:
                    g, n = r[part]["score"][k]
                    got += g
                    tot += n
                    full += g == n
        L.append(f"| {k} | {got}/{tot} | {full}/{len(res)} |")
    L.append("")
    L.append("## По мозгам")
    L.append("")
    L.append("| мозг | " + " | ".join(kinds) + " | всего |")
    L.append("|---|" + "---|" * (len(kinds) + 1))
    for r in res:
        cells, g_all, n_all = [], 0, 0
        for k in kinds:
            for part in ("основной", "логика", "память"):
                if k in r[part]["score"]:
                    g, n = r[part]["score"][k]
                    cells.append(f"{g}/{n}")
                    g_all += g
                    n_all += n
        L.append(f"| {r['seed']} | " + " | ".join(cells) + f" | {g_all}/{n_all} |")
    L.append("")
    L.append("## Ошибки")
    L.append("")
    L.append("| мозг | навык | задание | верно | мозг ответил |")
    L.append("|---|---|---|---|---|")
    for r in res:
        for part in ("основной", "логика", "память"):
            for kind, text, want, got, ok in r[part]["rows"]:
                if not ok:
                    L.append(f"| {r['seed']} | {kind} | {text} | {want} | {got} |")
    L.append("")
    open(path, "w").write("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--order", type=int, default=7)
    ap.add_argument("--report", default="")
    ap.add_argument("--seeds", default="1000-1023,2000")
    a = ap.parse_args()
    if a.report:
        report(parse_seeds(a.seeds), a.report)
        print("отчёт:", a.report)
        return
    print(json.dumps(one(a.seed, a.order), ensure_ascii=False))


if __name__ == "__main__":
    main()
