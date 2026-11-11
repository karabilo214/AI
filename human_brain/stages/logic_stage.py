"""Логика — ступень основного мозга (поверх памяти, memory.py).

Мозг тот же: считает, читает, знает цифры и «−», держит в памяти, что видел
и что слышал. Новое:
    знаки И, ИЛИ, НЕ — новые картинки (3 нейрона зрения, не совпадают с
        буквами, цифрами и «+», «−»); детекторы для них мозг нанимает сам
        (тот же найм, Recruit). Свободные детекторы стали нейронами-контекстами
        (Synapse.CONTEXTS): пока знак недавно горел, у связей действует своя
        добавка к силе — какой знак что значит, учится наградой;
    вопрос «верно ли?» (Q_LOGIC) — в области вопросов, рядом с «что видел?» и
        «что слышал?»; ответ — те же ДА / НЕТ.
Факты лежат в памяти: A — «видел 2» (истина) или «видел 3» (ложь), B —
«слышал c» (истина) или «слышал d» (ложь); показаны в случайном порядке,
через WAIT шагов — картинка знака, через 2 шага — вопрос. Верный ответ:
И — A и B; ИЛИ — A или B; НЕ — не A. Читается — как в памяти: сочетания
«вопрос + память» (окно совпадения), у памяти нет прямых связей к ответам.
Дни по кругу: логика, память, прежняя арифметика, слово, сложение с цифрами,
вычитание с «−».

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.logic_stage [--seed 1000] [--days 18000] [--save]
"""
import argparse
import copy
import random
from itertools import combinations

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.stages import division as dv
from human_brain.stages import reading as rd
from human_brain.stages import rules
from human_brain.stages import numerals as nm
from human_brain.stages import minus as mn
from human_brain.stages import memory as me
from human_brain.core import brain_store

OPS = {"AND": lambda a, c: a and c, "OR": lambda a, c: a or c, "NOT": lambda a, c: not a}
NAMES = {"AND": "И", "OR": "ИЛИ", "NOT": "НЕ"}
WAIT = 30
SIGN_LEAD = 2


def logic_name(seed, days=18000):
    return f"logic_seed{seed}_d{days}"


def sign_glyphs(seed):
    used = set(rd.make_alphabet(seed).values()) | set(nm.glyphs(seed).values()) | {mn.minus_glyph(seed)}
    free = [c for c in combinations(range(rd.N_VISION), 3) if c not in used]
    pics = random.Random(seed + 600).sample(free, len(OPS))
    return dict(zip(OPS, pics))


def add_logic(b):
    # свободные детекторы картинок — нейроны-контексты: новый знак, нанятый на
    # один из них, станет контекстом
    for i in range(b.n_neurons):
        if b.region_of[i] == "буквы" and i not in b.recruit.owner and b.ctx_slot[i] < 0:
            b.ctx_slot[i] = b.n_contexts
            b.n_contexts += 1
    b.add_neuron("Q_LOGIC", region="вопросы")
    for a in me.ANSWERS:
        b.connect("Q_LOGIC", a, weight=b.rules["Synapse"]["ACTION_MIN_WEIGHT"],
                  myelin_thickness=0.75, fixed=True)
    if b.neuron_change is not None:
        b.neuron_change.make_same(("вопросы",))
    g = b.growth
    mem = [b.names[i] for i in sorted(b.persistent)]
    partners = dict(g.pair_partners)
    partners["память"] = list(me.QUESTIONS) + ["Q_LOGIC"]
    partners["вопросы"] = mem
    g.pair_partners = partners
    g.max_neurons = b.n_neurons + len(mem) + 8
    b.curiosity.gate = list(me.QUESTIONS) + ["Q_LOGIC", "GO"]
    # знак — часть ситуации (иначе при тех же фактах И, ИЛИ и НЕ — одна
    # ситуация, и избегание, выученное при одном знаке, мешало другому)
    b.curiosity.situation_regions = ["вопросы", "память", "буквы"]


def logic_day(b, a, c, op, gl, signs, rng, learn, wait=WAIT):
    shown = [("see", me.A_SEEN if a else me.B_SEEN, rng.randint(2, 10)),
             ("hear", me.YES_SOUND if c else me.NO_SOUND, rng.randint(2, 10))]
    rng.shuffle(shown)
    for kind, what, gap in shown:
        me.show(b, kind, what, gl)
        me.quiet(b, gap, learn)
    me.quiet(b, wait, learn)
    for i in signs[op]:
        b.stimulate(f"V{i}", 20.0)
    me.quiet(b, SIGN_LEAD, learn)
    b.stimulate("Q_LOGIC", 20.0)
    for _ in range(7):
        f = b.step()
        acts = [n for n in me.ALL_ACTIONS if n in f]
        if acts:
            return acts[0] if len(acts) == 1 else "оба"
    return None


def cases(op):
    if op == "NOT":
        return [(False, False), (True, False), (False, True), (True, True)]
    return [(a, c) for a in (False, True) for c in (False, True)]


def exam_logic(b, gl, signs, seed, n=10):
    t = copy.deepcopy(b)
    t.curiosity = None
    rng = random.Random(seed + 777)
    score = {}
    for op in OPS:
        ok = 0
        for _ in range(n):
            a, c = rng.choice(cases(op))
            want = "YES" if OPS[op](a, c) else "NO"
            ok += logic_day(t, a, c, op, gl, signs, rng, False) == want
            me.quiet(t, 20, False)
        score[NAMES[op]] = ok / n
    return score


def run(p, seed, days=18000, save=False):
    path = brain_store.path_for(me.memory_name(seed))
    if not brain_store.exists(path):
        me.run(p, seed, 12000, save=True)
    b, rng, meta = brain_store.load(path)
    me_settings(b)
    add_logic(b)
    gl = dict(nm.glyphs(seed), MINUS=mn.minus_glyph(seed))
    signs = sign_glyphs(seed)
    alphabet = rd.make_alphabet(seed)
    train = meta.get("train") or rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    rnd = random.Random(seed + 66)
    for k in range(days):
        kind = k % 6
        if kind in (0, 3):
            op = rnd.choice(list(OPS))
            a, c = rnd.choice(cases(op))
            want = "YES" if OPS[op](a, c) else "NO"
            out = logic_day(b, a, c, op, gl, signs, rnd, True)
            b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
            me.quiet(b, 20, True)
        elif kind == 1:
            shown, q, want = me.make_memory_task(rnd)
            out = me.memory_day(b, shown, q, gl, True)
            b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
            me.quiet(b, 20, True)
        elif kind == 2:
            ev, ans, truth = rd.old_day(rng, k // 6)
            ad.day(b, ev, ans, learn=True, truth=truth)
        elif kind == 4:
            rules.say_word(b, rnd.choice(train), alphabet, True)
            tasks.rest(b, rd.REST)
        else:
            if (k // 6) % 2:
                ev, ans, truth = nm.digit_day(rnd)
            else:
                ev, ans, truth = mn.minus_day(rnd)
            ad.day(b, ev, ans, learn=True, truth=truth, show=nm.make_show(gl))
    if save:
        brain_store.save(b, rng, dict(meta, ступень="логика", seed=seed, days=days),
                         brain_store.path_for(logic_name(seed, days)))
    res = {f"логика {k}": v for k, v in exam_logic(b, gl, signs, seed).items()}
    res["память"] = me.exam_memory(b, gl, seed)
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b


def me_settings(b):
    """Настройки ступени памяти, которые не хранятся в файле мозга."""
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    b.recall_min = 0.75 * b.association.scale
    b.persistent = {i for i in range(b.n_neurons) if b.region_of[i] == "память"}
    g = b.growth
    g.pair_window = 3
    g.pair_window_regions = ["память"]
    g.inherit_inhibition = False
    b.curiosity.gate = list(me.QUESTIONS) + ["GO"]
    b.curiosity.gate_window = 7
    b.curiosity.situational_avoid = True
    b.curiosity.situation_regions = ["вопросы", "память"]
    b.rules["Synapse"]["NEG_RATE"] = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--days", type=int, default=18000)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    res, b = run(p, a.seed, a.days, a.save)
    for k, v in res.items():
        if k.startswith(("логика", "память")):
            print(f"  {k:14s} {v:.2f}")
    ar = [k for k in res if not k.startswith(("логика", "память"))]
    print(f"  арифметика наборов 100%: {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")


if __name__ == "__main__":
    main()
