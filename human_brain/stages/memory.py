"""Рабочая память — ступень основного мозга (поверх знака «−», minus.py).

Мозг тот же, что учился считать, читать, знает цифры и «−». Новое:
    буфер памяти для каждого восприятия: у каждого нанятого детектора картинок
        (буквы, цифры, знаки) и у каждого звука — по PER залипающих нейронов
        (Brain.persistent). Восприятие возбуждает свой буфер (сильный
        закреплённый вход — как входы детекторов) и тормозит буферы других
        восприятий того же чувства: новая картинка вытесняет прежнюю картинку,
        но не звук. Мир ничего не обнуляет.
    вопросы «что видел?» (Q_SEE) и «что слышал?» (Q_HEAR) — своя область;
    ответы ДА / НЕТ — новые действия (соревнуются между собой).
Устройство чтения — найденное в hold.py (задача с задержкой): ответ только
через сочетания «вопрос + память» (Growth.pair_window = 3, только для области
памяти, один раз на совпадение); у памяти нет прямых связей к ответам;
вопрос -> ответ слабая неизменная; сочетаниям не передаются обучаемые
тормоза; поиск (шум, скука) только после вопроса (вопросы счёта GO тоже
считаются вопросом, поэтому поиск в счёте сохраняется).
Задание дня памяти: показана цифра A_SEEN или B_SEEN и прозвучал звук
YES_SOUND или NO_SOUND (в случайном порядке, через 2-10 шагов), через WAIT
шагов вопрос: «что видел?» — ДА, если была A_SEEN; «что слышал?» — ДА,
если YES_SOUND. (Буквы как факты «что видел» не годятся: мозг читает букву
вслух, и её звук вытесняет из памяти услышанный.) Действие, когда не спрашивали, — ошибка (наказание).
Дни по кругу: память, прежняя арифметика, слово (правило чтения), сложение с
цифрами, вычитание с записью «−», снова память.
Проверка — на копии без шума: память (оба вопроса, оба порядка), и всё
прежнее (тесты full_exam).
Поправки по разбору (25 мозгов, 12000 дней): вспоминание звука только по
надёжной связи (recall_min — цифры перестали «говорить» случайный звук и
затирать слуховую память); «проиграл — смени» по ситуации (Curiosity.
situational_avoid, ситуация = вопрос + что в памяти) и отрицательная память
без замедления — иначе мозг застревал на «ответе по умолчанию». Память на
100%: 14/25 -> 21/25 (средняя 0.91 -> 0.92); у 1010 арифметика 26 -> 20/26.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.memory [--seed 1000] [--days 12000]
"""
import argparse
import copy
import random

from human_brain.stages import tasks
from human_brain.stages import addition as ad
from human_brain.stages import subtraction as sb
from human_brain.stages import division as dv
from human_brain.stages import reading as rd
from human_brain.stages import rules
from human_brain.stages import numerals as nm
from human_brain.stages import minus as mn
from human_brain.parts import Head
from human_brain.core import brain_store

PER = 3
WAIT = 50
# факты «что видел» — цифры 2 и 3: буквы мозг читает вслух, и звук буквы
# вытеснял из памяти услышанный звук (внутренняя речь перебивала слуховую память)
A_SEEN, B_SEEN = 2, 3
YES_SOUND, NO_SOUND = "c", "d"
QUESTIONS = ("Q_SEE", "Q_HEAR")
ANSWERS = ("YES", "NO")
ALL_ACTIONS = ("MORE", "DONE", "YES", "NO")
AVOID = __import__("os").environ.get("HB_MEM_AVOID", "1") == "1"
CRITIC = __import__("os").environ.get("HB_MEM_CRITIC", "0") == "1"


def memory_name(seed, days=12000):
    return f"memory_seed{seed}_d{days}"


def detector_of(b, pic):
    ev = sorted(b.neurons[f"V{i}"].index for i in pic)
    for w, e in b.recruit.owner.items():
        if sorted(e) == ev:
            return b.names[w]
    return None


def add_memory(b, p):
    """Буферы памяти, вопросы, ответы ДА/НЕТ и устройство чтения."""
    th = b.rules["Neuron"]["THRESHOLD"] - b.rules["Neuron"]["REST"]
    seen = [b.names[w] for w in sorted(b.recruit.owner)]            # детекторы картинок
    heard = [n for n in b.names if n.startswith("S_")]              # звуки
    b.add_region("память")
    buf = {}
    for src in seen + heard:
        buf[src] = []
        for k in range(PER):
            n = f"MEM_{src}_{k}"
            b.add_neuron(n, region="память")
            buf[src].append(n)
    for group in (seen, heard):
        for src in group:
            for other in group:
                for m in buf[other]:
                    if other == src:
                        b.connect(src, m, weight=2 * th, myelin_thickness=1.0, fixed=True)
                    else:
                        b.connect(src, m, weight=2 * th, inhibitory=True, myelin_thickness=1.0)
    b.add_region("вопросы")
    for q in QUESTIONS:
        b.add_neuron(q, region="вопросы")
    for a in ANSWERS:
        b.add_neuron(a, explores=True, region="действия")
    for q in QUESTIONS:
        for a in ANSWERS:
            b.connect(q, a, weight=p["action_min"], myelin_thickness=0.75, fixed=True)
    b.connect("YES", "NO", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("NO", "YES", weight=p["inhibit_weight"], inhibitory=True)
    mem = [m for ms in buf.values() for m in ms]
    b.persistent = {b.neurons[m].index for m in mem}
    if b.neuron_change is not None:
        b.neuron_change.make_same(("память", "вопросы", "действия"))
    g = b.growth
    g.sensory_regions = sorted(set(g.sensory_regions or []) | {"память", "вопросы"})
    g.pair_regions = sorted(set(g.pair_regions or []) | {"память", "вопросы"})
    partners = dict(g.pair_partners or {})
    partners["память"] = list(QUESTIONS)
    partners["вопросы"] = mem
    g.pair_partners = partners
    g.pair_window = 3
    g.pair_window_regions = ["память"]
    g.inherit_inhibition = False
    g.max_neurons = b.n_neurons + 2 * len(mem) + 8
    # вспоминать звук только по надёжной связи «бывает вместе» (не меньше 3/4
    # полной силы): у цифр и «−» выросли слабые случайные связи со звуками
    # (окно 72 шага захватывало слова соседних дней), и любая картинка
    # заставляла мозг «сказать» какой-то звук — он вытеснял из памяти услышанный
    b.recall_min = 0.75 * b.association.scale
    b.curiosity.gate = list(QUESTIONS) + ["GO"]
    b.curiosity.gate_window = 7
    if AVOID:
        # «проиграл — смени» по ситуации (вопрос + что держится в памяти)
        b.curiosity.situational_avoid = True
        b.curiosity.situation_regions = ["вопросы", "память"]
    # отрицательная память без замедления: связь «вопрос + память -> ответ»,
    # однажды опущенная на минимум, иначе поднималась наградой в 20 раз
    # медленнее, и мозг оставался на «ответе по умолчанию» (правила мозга
    # хранятся в его файле — задаём ему самому)
    b.rules["Synapse"]["NEG_RATE"] = float(__import__("os").environ.get("HB_NEG_RATE", "1.0"))
    if CRITIC and b.critic is None:
        # ожидание награды по тому, что недавно горело (входы, вопросы, память,
        # детекторы, звуки): ложная связь «вопрос + память другого чувства»
        # получала награду и наказание поровну и держалась; с ожиданием
        # неожиданное наказание сильнее ожидаемой награды
        from human_brain.core.RewardPrediction import RewardPrediction
        RewardPrediction(b, regions=["входы", "вопросы", "память", "буквы", "слух"])
    return buf


def show(b, kind, what, glyphs):
    if kind == "see":
        for i in glyphs[f"D{what}"]:
            b.stimulate(f"V{i}", 20.0)
    else:
        b.stimulate(f"S_{what}", 20.0)


def quiet(b, n, learn):
    for _ in range(n):
        if set(ALL_ACTIONS) & b.step() and learn:
            b.reward(-1.0)


def memory_day(b, shown, q, glyphs, learn, wait=WAIT):
    """shown: [(«see»/«hear», что, шагов до следующего)]; потом wait и вопрос q."""
    for kind, what, gap in shown:
        show(b, kind, what, glyphs)
        quiet(b, gap, learn)
    quiet(b, wait, learn)
    b.stimulate(q, 20.0)
    out = None
    for _ in range(7):
        f = b.step()
        a = [n for n in ALL_ACTIONS if n in f]
        if a:
            out = a[0] if len(a) == 1 else "оба"
            break
    return out


def make_memory_task(rng):
    letter = rng.choice((A_SEEN, B_SEEN))
    sound = rng.choice((YES_SOUND, NO_SOUND))
    shown = [("see", letter, rng.randint(2, 10)), ("hear", sound, rng.randint(2, 10))]
    rng.shuffle(shown)
    q = rng.choice(QUESTIONS)
    if q == "Q_SEE":
        want = "YES" if letter == A_SEEN else "NO"
    else:
        want = "YES" if sound == YES_SOUND else "NO"
    return shown, q, want


def exam_memory(b, glyphs, seed, n=40):
    t = copy.deepcopy(b)
    t.curiosity = None
    rng = random.Random(seed + 4242)
    ok = 0
    for _ in range(n):
        shown, q, want = make_memory_task(rng)
        ok += memory_day(t, shown, q, glyphs, False) == want
        quiet(t, 20, False)
    return ok / n


def run(p, seed, days=12000, save=False):
    path = brain_store.path_for(mn.minus_name(seed))
    if not brain_store.exists(path):
        mn.run(p, seed, 2000, save=True)
    b, rng, meta = brain_store.load(path)
    add_memory(b, p)
    alphabet = rd.make_alphabet(seed)
    gl = dict(nm.glyphs(seed), MINUS=mn.minus_glyph(seed))
    show_gl = nm.make_show(gl)
    train = meta.get("train") or rules.make_words(seed + 10, 40, (2, 3, 4, 5))
    rnd = random.Random(seed + 55)
    for k in range(days):
        kind = k % 6
        if kind in (0, 3):
            shown, q, want = make_memory_task(rnd)
            out = memory_day(b, shown, q, gl, True)
            b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
            quiet(b, 20, True)
        elif kind == 1:
            ev, ans, truth = rd.old_day(rng, k // 6)
            ad.day(b, ev, ans, learn=True, truth=truth)
        elif kind == 2:
            rules.say_word(b, rnd.choice(train), alphabet, True)
            tasks.rest(b, rd.REST)
        elif kind == 4:
            ev, ans, truth = nm.digit_day(rnd)
            ad.day(b, ev, ans, learn=True, truth=truth, show=show_gl)
        else:
            ev, ans, truth = mn.minus_day(rnd)
            ad.day(b, ev, ans, learn=True, truth=truth, show=show_gl)
    if save:
        brain_store.save(b, rng, dict(meta, ступень="память", seed=seed, days=days),
                         brain_store.path_for(memory_name(seed, days)))
    res = {"память (оба вопроса)": exam_memory(b, gl, seed)}
    res.update({k: sb.exam(b, v) / len(v) for k, v in dv.exam_sets(seed).items()})
    return res, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--days", type=int, default=12000)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    if __import__("os").environ.get("HB_NEG_RATE"):
        p["neg_rate"] = float(__import__("os").environ["HB_NEG_RATE"])
    tasks.apply(p)
    res, b = run(p, a.seed, a.days, a.save)
    print(f"  память (оба вопроса)       {res['память (оба вопроса)']:.2f}")
    ar = [k for k in res if not k.startswith("память")]
    print(f"  арифметика наборов 100%: {sum(res[k] == 1.0 for k in ar)}/{len(ar)}")


if __name__ == "__main__":
    main()
