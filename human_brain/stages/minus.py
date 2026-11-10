"""Знак «−», как в школе: знак учится рядом с операцией, которую мозг уже умеет.

Мозг — сохранённый после ступени «цифры» (numerals.py: цифры 2-6, «a + b»).
Новое:
    картинка знака «−» (3 нейрона зрения, не совпадает с буквами, цифрами и
        «+»); детектор для неё мозг нанимает сам (Recruit), как для букв;
    нейроны ролей (Accumulator, role_of): по нейрону на каждое событие мира,
        которое знает накопитель (карточка, конфета, дверь, гость, вопросы);
        связи к ним от всех символов (буквы, цифры, «+», «−») — синапсы с силой
        0. Общая врождённая способность: цифра, увиденная после сработавшего
        нейрона роли события e, работает как столько-то раз e. Какой символ
        какую роль включает, мозг находит сам: пробует, и после праздника
        использованная связь крепнет, если число сошлось, и слабеет, если нет.
        Про минус в этом ничего нет.
Ступень «минус»: дни вычитания (у меня c конфет, пришли g гостей, вопрос
«сколько не хватило?» или «сколько останется?»), где конфеты часто записаны:
картинка «−», затем цифра c, вместо c конфет по одной. Что значит «−», никто
не говорит: после праздника видно только, скольким не хватило или сколько
осталось. Остальные дни — прежние: цифры при сложении, чтение, арифметика.
Проверка: выражение, которого не было ни разу: «a», «−», «b», «сколько не
хватило?» — без карточки, конфет и гостей; ответ max(a - b, 0). Отдельно
пары a > b (ленивое «всегда 0» там 0). Второй вопрос к тому же выражению —
«сколько лишнего?» (ответ max(b - a, 0)): с одним «не хватило» любой уровень
ниже нуля даёт 0, и вычитание 6 при a > b не проверялось бы вовсе. «−4» в обучении не
встречается никогда (HOLD): «a − 4» проверяет знак, а не выученные пары.
Контроль: знак рядом со случайной операцией — в половине дней с записью
«−, цифра k» означает ещё k гостей (прибавить), в половине — k конфет.
(Прежний контроль — случайная цифра после «−» — для ролей ничего не
проверял: конфеты всё равно уходили, и роль «отнять» оставалась верной.)

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.minus [--seed 1000] [--days 1000] [--control]
"""
import argparse
import random
from itertools import combinations

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.stages import division as dv
from human_brain.stages import reading as rd
from human_brain.stages import rules
from human_brain.stages import numerals as nm
from human_brain.core import brain_store

SILENCE = float(__import__("os").environ.get("HB_SILENCE", "1.0"))
DOUBT_PER_DAY = __import__("os").environ.get("HB_DOUBT", "1") == "1"


def minus_glyph(seed):
    used = set(rd.make_alphabet(seed).values()) | set(nm.glyphs(seed).values())
    free = [c for c in combinations(range(rd.N_VISION), 3) if c not in used]
    return random.Random(seed + 51).choice(free)


HOLD = 4      # «−4» в обучении не встречается никогда (конфет 4 — всегда предметами)


def minus_day(rng, control=False):
    """Вычитание, где конфеты часто записаны «−, цифра»; гости — как в днях цифр.
    Контроль: знак рядом со случайной операцией — в половине дней с записью
    «−, цифра k» означает ещё k гостей (прибавить), а конфеты показаны предметами."""
    c, g = rng.randint(0, 10), rng.randint(2, 10)
    ev = ["CARD"]
    extra = 0
    if control and rng.random() < 0.5 * 0.6:
        extra = rng.choice([d for d in nm.DIGITS if d != HOLD])
        ev += ["CANDY"] * c + ["MINUS", f"D{extra}"]
    elif c in nm.DIGITS and c != HOLD and rng.random() < 0.6:
        ev += ["MINUS", f"D{c}"]
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
    g += extra
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


def add_roles(b, seed):
    """Нейроны ролей и связи к ним от всех символов (сила 0)."""
    acc = b.accumulator
    events = [n for n in acc.inputs if n not in acc.seen]
    b.add_region("роли", learn=0.0)
    for e in events:
        b.add_neuron(f"R_{e}", region="роли")
    if b.neuron_change is not None:
        b.neuron_change.make_same(("роли",))
    g = b.growth
    g.sensory_regions = sorted(set(g.sensory_regions or []) | {"роли"})
    for sym in acc.seen:
        for e in events:
            b.connect(sym, f"R_{e}", weight=0.0, myelin_thickness=1.0, fixed=True)
    acc.role_of = {f"R_{e}": e for e in events}
    acc.role_rng = random.Random(seed + 89)


def minus_name(seed, days=2000):
    return f"minus_seed{seed}_d{days}"


def run(p, seed, days=1000, control=False, save=False):
    path = brain_store.path_for(nm.numerals_name(seed))
    if not brain_store.exists(path):
        nm.run(p, seed, 1000, save=True)
    b, rng, meta = brain_store.load(path)
    acc = b.accumulator
    slow = 1.0 / b.growth.young_plasticity
    free = {n for n in acc.seen if b.neurons[n].index not in b.recruit.owner}
    for n in acc.inputs:          # всё выученное раньше закреплено; свободные детекторы — новые
        acc.plast[n] = 1.0 if n in free else slow
    add_roles(b, seed)
    # молчание в ответ на «отвечай» — ошибка (см. addition.day): иначе тормоз к
    # «готово» растёт лавиной и ломает «сколько останется?»
    b.silence_penalty = SILENCE
    # сомнение в закреплённом пути — не чаще раза за день (см. kernel.reward)
    if DOUBT_PER_DAY:
        b.episode = max(b.episode, 0)
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
    if save and not control:
        brain_store.save(b, rng, dict(meta, ступень="минус", seed=seed, days=days),
                         brain_store.path_for(minus_name(seed, days)))
    rows = exam_minus(b, show)
    big = [r for r in rows if r[0] > r[1]]
    left = exam_minus(b, show, "ASK_LEFT")
    held = [x[2] == x[3] and y[2] == y[3] for x, y in zip(rows, left) if x[1] == HOLD]
    res = {"выражение a − 4 (не было после «−»)": sum(held) / len(held),
           "выражение a − b (все 25)": sum(r[2] == r[3] for r in rows) / len(rows),
           "выражение a − b, a > b": sum(r[2] == r[3] for r in big) / len(big)}
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
    roles = {}
    for c in b.connections:
        if c.target in acc.role_of and c.source in names and c.synapse.weight > 0.01:
            roles.setdefault(names[c.source], {})[acc.role_of[c.target]] = round(c.synapse.weight, 1)
    print("символ -> роль (сила; срабатывает от", round(b.reach(b.neurons["R_CANDY"].index), 1), "):", roles)
    print("выражения (a, b, верно, мозг):", rows)
    for k, v in res.items():
        if not k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление")):
            print(f"  {k:26s} {v:.2f}")
    ar = [k for k in res if k.startswith(("сложение", "не хватило", "осталось", "умножение", "деление"))]
    print(f"  арифметика наборов 100%: {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")


if __name__ == "__main__":
    main()
