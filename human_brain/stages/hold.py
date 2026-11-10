"""Удержание: может ли активность нейронов сама держаться между событиями.

Сейчас всё, что живёт дольше нескольких шагов (число дня, вопрос), держат
устройства, которые каждое утро обнуляет код мира. Здесь проверяется, может
ли это делать сама сеть: кольцо нейронов, раз загоревшись, горит само, пока
его не погасит торможение.

Проверка 1 (заведомо решаемая, --ring): кольцо из RING нейронов, связи по
кругу сильнее порога; толчок CUE зажигает первый, STOP тормозит все. Сколько
шагов кольцо горит само и гаснет ли по STOP. Два варианта: голое ядро и мозг
со всеми механизмами (рост, любопытство, изменение нейронов, запас входов,
обучение связей) — они могут кольцо разрушить.

Проверка 2 (--random): никаких колец заранее. Область «память» из N_MEM
нейронов, у каждого OUT случайных связей к другим со случайной задержкой
1..4 шага (важна длина петли во времени, а не число нейронов в ней), сила —
доля g от «до порога» (разброс 0.5..1.5). Факт A зажигает свои CUE_SIZE
нейронов, факт B — другие. Мерится: горит ли что-то через HOLD шагов, не
взрыв ли (горит почти всё), различаются ли следы A и B (доля общих нейронов).
До обучения и после показов фактов (рост связей и обучение могут сами
выделить петли).

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.hold --ring [--steps 200] [--stop 150]
    .venv/bin/python -m human_brain.stages.hold --random [--seed 0]
    .venv/bin/python -m human_brain.stages.hold --plateau
    .venv/bin/python -m human_brain.stages.hold --delay [--seed 0] [--wait 50] [--no-persist] [--control]

Проверка 3 (--delay): задача с задержкой и наградой. Память — N_DMEM
залипающих нейронов без связей внутри. Факт A или B ведёт в свои DMEM_IN
случайных нейронов памяти (сильный вход) и тормозит остальные (новый факт
гасит старый; мир ничего не обнуляет). Через wait шагов — «отвечай» (GO),
ответ YES (было A) / NO (было B), награда. Связи память -> ответ учатся
(как в мозге логики: ожидание награды, скука по ситуации). Экзамен на копии
без шума: A и B, и смена факта (A, через 20 шагов B, потом ждать wait —
верно NO). Сравнение: без залипания (--no-persist), контроль — ответ
случайный (--control).
Что для этого понадобилось (каждое — по разбору, по умолчанию так):
    ответ читается только через сочетания «GO + память» с окном совпадения
        (Growth.pair_window = 3): залипший нейрон горит раз в 3 шага;
    у памяти нет прямых связей к ответам (DIRECT=0): постоянная память с
        минимумом путей к действиям копила заряд и сама поджигала ответы;
    GO -> ответ слабая неизменная (GO_FIXED): GO есть в каждой пробе и,
        обучаемая, забирала ответ себе (24.2); от неё сочетания наследуют
        направление к ответам;
    сочетания без наследованных обучаемых тормозов (INHERIT=0): тормоза
        жадно забивали выход;
    поиск (шум, скука) только после вопроса (Curiosity.gate, 7 шагов):
        иначе за 50 шагов ожидания поиск поджигал действия, наказания душили ответ;
    скука по ситуации выключена (лавина от наказаний за действия без вопроса);
    вход факта закреплён, у памяти нет подстройки порога.
Результат (2000 проб): выучено 11/12, свежие 23/24; без залипания 0/12;
случайные ответы 2/12.
"""
import argparse
import copy
import random

from human_brain.stages import tasks
from human_brain.core.Brain import Brain

RING = 3
N_MEM = 40
OUT = 4
CUE_SIZE = 5
HOLD = 50
N_INH = 0      # тормозных нейронов в области (0 = без торможения)
INH_IN = 8     # от скольких нейронов памяти каждый тормозной получает возбуждение
INH_OUT = 10   # скольких нейронов памяти каждый тормозной гасит
INH_W = 1.0    # сила торможения, доля «до порога»
CUE_HOLD = 1   # сколько шагов подряд показывается факт (ансамбли: несколько)
PERSIST = False  # нейроны памяти умеют залипать (Brain.persistent)


def make_ring_brain(p, seed, full):
    b = Brain()
    b.add_region("входы")
    for n in ("CUE", "STOP"):
        b.add_neuron(n, region="входы")
    b.add_region("кольцо")
    names = [f"M{k}" for k in range(RING)]
    for n in names:
        b.add_neuron(n, region="кольцо")
    th = b.reach(b.neurons["M0"].index)
    b.connect("CUE", "M0", weight=th + 4.0, myelin_thickness=1.0)
    for k in range(RING):
        b.connect(names[k], names[(k + 1) % RING], weight=th + 4.0, myelin_thickness=1.0)
        b.connect("STOP", names[k], weight=3 * th, inhibitory=True, myelin_thickness=1.0)
    if full:
        tasks.attach(b, p, seed)
        b.growth.sensory_regions = ["входы"]
    return b


N_DMEM = 10
GO_FIXED = __import__("os").environ.get("HB_GO_FIXED", "1") == "1"
GO_DIRECT = __import__("os").environ.get("HB_GO_DIRECT", "1") == "1"   # прямые связи GO -> ответ
WEAK_GO = __import__("os").environ.get("HB_WEAK_GO", "0") == "1"   # GO -> ответ с минимума
DIRECT = __import__("os").environ.get("HB_DIRECT", "0") == "1"   # прямые связи память -> ответ
INHERIT = __import__("os").environ.get("HB_INHERIT", "0") == "1"
GATE_WINDOW = int(__import__("os").environ.get("HB_GATE_WINDOW", "7"))
GATE = __import__("os").environ.get("HB_GATE", "1") == "1"
SITUATIONAL = __import__("os").environ.get("HB_SITUATIONAL", "0") == "1"
PAIR_WINDOW = int(__import__("os").environ.get("HB_PAIR_WINDOW", "3"))
DMEM_IN = 3


def make_delay_brain(p, seed, persist=True):
    from human_brain.parts import Head
    from human_brain.core.RewardPrediction import RewardPrediction
    rnd = random.Random(seed)
    b = Brain()
    b.add_region("входы")
    for n in ("CUE_A", "CUE_B", "GO"):
        b.add_neuron(n, region="входы")
    b.add_region("память")
    mem = [f"M{k}" for k in range(N_DMEM)]
    for n in mem:
        b.add_neuron(n, region="память")
    b.add_region("действия")
    for n in ("YES", "NO"):
        b.add_neuron(n, explores=True, region="действия")
    th = b.reach(b.neurons["M0"].index)
    for cue in ("CUE_A", "CUE_B"):
        own = set(rnd.sample(mem, DMEM_IN))
        for m in mem:
            if m in own:
                # вход восприятия закреплён (как у детекторов букв): иначе правило
                # ослабления разъедало его, и факт переставал доходить до памяти
                b.connect(cue, m, weight=2 * th, myelin_thickness=1.0, fixed=True)
            else:
                b.connect(cue, m, weight=2 * th, inhibitory=True, myelin_thickness=1.0)
    for s in (mem if DIRECT else []) + (["GO"] if GO_DIRECT else []):
        for o in ("YES", "NO"):
            # память горит постоянно: её прямые связи к ответам стартуют с
            # минимума путей к действиям, а не с обычной начальной силы (14) —
            # иначе каждая вспышка памяти сама поджигала и «да», и «нет»
            w0 = p["action_min"] if (s != "GO" or WEAK_GO) else p["init_weight"]
            if s == "GO" and GO_FIXED:
                # GO -> ответ: слабая неизменная (как датчики в мозге вычитания):
                # GO есть в каждой пробе и ничего не говорит о факте; от неё
                # сочетания «GO + память» наследуют направление к ответам
                b.connect(s, o, weight=p["action_min"], myelin_thickness=0.75, fixed=True)
            else:
                b.connect(s, o, weight=w0, myelin_thickness=1.0 if s != "GO" else 0.75,
                          plasticity=p["init_plasticity"])
            # тормоз от постоянно горящей памяти и от GO не учится (как у датчиков
            # в мозге вычитания): иначе память, чтобы не отвечать без вопроса,
            # держала действия глубоко под порогом и при вопросе; учатся тормоза
            # сочетаний «память + GO»
            b.connect(s, o, weight=0.0, inhibitory=True, myelin_thickness=1.0, plasticity=0.0)
    b.inh_window = Head.INH_WINDOW
    b.inh_delayed = True
    b.connect("YES", "NO", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("NO", "YES", weight=p["inhibit_weight"], inhibitory=True)
    b.inh_short = True
    b.reset_eligibility = True
    b.last_action_only = True
    b.budget_above_floor = True
    b.reward_only_actions = True
    tasks.attach(b, p, seed)
    # и у памяти без подстройки порога: залипший нейрон горит часто, порог
    # рос, и сильный вход факта переставал включать залипание
    b.neuron_change.make_same(("входы", "действия", "память"))
    b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
    b.curiosity.avoid = p.get("avoid", 3.0)
    # скука по ситуации здесь выключена: наказание за действие без вопроса
    # прибавляло к скуке ситуации «горит память», скука толкала действия, и
    # те снова срабатывали без вопроса (лавина, ~14 лишних действий на пробу)
    b.curiosity.situational = SITUATIONAL
    if GATE:
        b.curiosity.gate = "GO"      # поиск (шум, скука) только после вопроса
        b.curiosity.gate_window = GATE_WINDOW
    b.growth.sensory_regions = ["входы", "память"]
    # ответ должен зависеть от «вопрос И память»: сочетания нейронов памяти с GO
    # (память горит всю задержку, сама по себе она не повод отвечать); между
    # собой залипшие нейроны не сочетаются
    b.growth.pair_regions = ["входы", "память"]
    b.growth.pair_partners = {"память": ["GO"]}
    # окно совпадения: залипший нейрон горит раз в 3 шага, «отвечай» — один
    # толчок; на одном шаге они совпадали лишь в 1 из 3 случаев
    b.growth.pair_window = PAIR_WINDOW
    b.growth.max_neurons = b.n_neurons + N_DMEM + 4
    if INHERIT:
        b.growth.inherit_inhibition = p.get("inh_plasticity", Head.INH_PLASTICITY)
    b.growth.new_weight = p["action_min"] + 0.1
    b.budget_window = p.get("budget_window", Head.BUDGET_WINDOW)
    RewardPrediction(b, regions=["входы", "память"])
    if persist:
        b.persistent = {b.neurons[m].index for m in mem}
    return b


def quiet_steps(b, n, learn):
    """n шагов, когда не спрашивали: действие — ошибка (при обучении наказание),
    как в днях счёта (addition.day)."""
    for _ in range(n):
        if {"YES", "NO"} & b.step() and learn:
            b.reward(-1.0)


def delay_trial(b, cues, wait, learn=False):
    """cues: [(факт, через сколько шагов следующий)]; потом wait шагов и GO."""
    for cue, gap in cues:
        b.stimulate(cue, 20.0)
        quiet_steps(b, gap, learn)
    quiet_steps(b, wait, learn)
    b.stimulate("GO", 20.0)
    for _ in range(7):
        f = b.step()
        a = [n for n in ("YES", "NO") if n in f]
        if a:
            return a[0] if len(a) == 1 else "оба"
    return None


def delay_run(p, seed, wait=50, trials=2000, persist=True, control=False):
    b = make_delay_brain(p, seed, persist)
    rnd = random.Random(seed + 3)
    for _ in range(trials):
        cue = rnd.choice(("CUE_A", "CUE_B"))
        want = rnd.choice(("YES", "NO")) if control else ("YES" if cue == "CUE_A" else "NO")
        out = delay_trial(b, [(cue, 0)], wait, learn=True)
        b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
        quiet_steps(b, 20, True)
    t = copy.deepcopy(b)
    t.curiosity = None
    rows = []
    for cues, want in (([("CUE_A", 0)], "YES"), ([("CUE_B", 0)], "NO"), ([("CUE_A", 0)], "YES"),
                       ([("CUE_A", 20), ("CUE_B", 0)], "NO"), ([("CUE_B", 20), ("CUE_A", 0)], "YES")):
        out = delay_trial(t, cues, wait)
        rows.append((" ".join(c[-1] for c, _ in cues), want, out))
        for _ in range(20):
            t.step()
    return rows


def make_multi_brain(p, seed, dims, per=3, spare=4):
    # Результаты (12 мозгов, 2000 проб, задержка 50, доля верных на экзамене):
    #   один факт из 2: 12/12 = 1.0; из 3: 12/12; из 4: 11/12 (0.94);
    #   из 2 с задержкой 100: 12/12; с задержкой 200: 9/12 (0.89);
    #   два факта сразу (2,2): 0/12 (0.64) — не решено. Перебрано 9 вариантов
    #   (вопросы в своей области и ответы только своего измерения — оставлены;
    #   6000 проб, без отрицательной памяти, без ожидания награды, без запаса
    #   входов): лучший — без ожидания награды, 3-4/12 (0.77), и он портится
    #   при долгом обучении. Посторонняя память горит вместе с нужной, награда
    #   делится между ними, нужные пути застревают на минимуме.
    """Обобщение: dims — число значений у каждого измерения фактов (например
    [2, 2] — цвет и форма). Значение v измерения d: факт F{d}_{v} возбуждает
    свою группу из per нейронов памяти (группы не пересекаются) и тормозит
    группы других значений того же измерения (новый цвет гасит старый цвет,
    но не форму). Вопрос Q{d} на каждое измерение; ответы — действия
    R{d}_{v}, все соревнуются. Устройство чтения — как в make_delay_brain."""
    from human_brain.parts import Head
    from human_brain.core.RewardPrediction import RewardPrediction
    rnd = random.Random(seed)
    b = Brain()
    b.add_region("входы")
    facts = [[f"F{d}_{v}" for v in range(n)] for d, n in enumerate(dims)]
    qs = [f"Q{d}" for d in range(len(dims))]
    for n in [f for fs in facts for f in fs]:
        b.add_neuron(n, region="входы")
    # вопросы — своя область: сочетания растут только «вопрос + память»
    # (иначе два факта, показанные с промежутком 2-10 шагов, попадали в окно
    # совпадения, и сочетания «факт + факт» занимали место под рост)
    b.add_region("вопросы")
    for n in qs:
        b.add_neuron(n, region="вопросы")
    b.add_region("память")
    n_mem = per * sum(dims) + spare
    mem = [f"M{k}" for k in range(n_mem)]
    for n in mem:
        b.add_neuron(n, region="память")
    b.add_region("действия")
    acts = [[f"R{d}_{v}" for v in range(n)] for d, n in enumerate(dims)]
    all_acts = [a for a_s in acts for a in a_s]
    for n in all_acts:
        b.add_neuron(n, explores=True, region="действия")
    th = b.reach(b.neurons["M0"].index)
    pool = rnd.sample(mem, per * sum(dims))
    group = {}
    k = 0
    for fs in facts:
        for f in fs:
            group[f] = pool[k:k + per]
            k += per
    for fs in facts:
        for f in fs:
            for g in fs:
                for m in group[g]:
                    if g == f:
                        b.connect(f, m, weight=2 * th, myelin_thickness=1.0, fixed=True)
                    else:
                        b.connect(f, m, weight=2 * th, inhibitory=True, myelin_thickness=1.0)
    # вопрос слабо связан только с ответами своего измерения (на вопрос о
    # цвете отвечают цветом); от него сочетания «вопрос + память» наследуют
    # выходы — тоже только к своим ответам
    for d, q in enumerate(qs):
        for o in acts[d]:
            b.connect(q, o, weight=p["action_min"], myelin_thickness=0.75, fixed=True)
    for i, x in enumerate(all_acts):
        for y in all_acts[i + 1:]:
            b.connect(x, y, weight=p["inhibit_weight"], inhibitory=True)
            b.connect(y, x, weight=p["inhibit_weight"], inhibitory=True)
    b.inh_window = Head.INH_WINDOW
    b.inh_delayed = True
    b.inh_short = True
    b.reset_eligibility = True
    b.last_action_only = True
    b.budget_above_floor = True
    b.reward_only_actions = True
    tasks.attach(b, p, seed)
    b.neuron_change.make_same(("входы", "вопросы", "действия", "память"))
    b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
    b.curiosity.avoid = p.get("avoid", 3.0)
    b.curiosity.gate = qs            # поиск только после любого вопроса
    b.curiosity.gate_window = GATE_WINDOW
    b.growth.sensory_regions = ["входы", "вопросы", "память"]
    b.growth.pair_regions = ["вопросы", "память"]
    b.growth.pair_partners = {"память": qs, "вопросы": mem}
    b.growth.max_neurons = b.n_neurons + n_mem * len(qs) + 4
    b.growth.new_weight = p["action_min"] + 0.1
    b.growth.pair_window = PAIR_WINDOW
    b.budget_window = p.get("budget_window", Head.BUDGET_WINDOW)
    RewardPrediction(b, regions=["входы", "вопросы", "память"])
    b.persistent = {b.neurons[m].index for m in mem}
    return b, facts, qs, acts


def multi_trial(b, shown, q, wait, acts, learn=False):
    """shown: [(факт, шагов до следующего)]; потом wait шагов и вопрос q."""
    names = [a for a_s in acts for a in a_s]
    def quiet(n):
        for _ in range(n):
            if set(names) & b.step() and learn:
                b.reward(-1.0)
    for f, gap in shown:
        b.stimulate(f, 20.0)
        quiet(gap)
    quiet(wait)
    b.stimulate(q, 20.0)
    for _ in range(7):
        f = b.step()
        a = [n for n in names if n in f]
        if a:
            out = a[0] if len(a) == 1 else "оба"
            return out
    return None


def multi_run(p, seed, dims, wait=50, trials=2000, persist=True):
    b, facts, qs, acts = make_multi_brain(p, seed, dims)
    if not persist:
        b.persistent = set()
    rnd = random.Random(seed + 3)
    state = [None] * len(dims)
    def episode(learn, b_, rnd_):
        # показать значения всех (или части) измерений в случайном порядке
        order = list(range(len(dims)))
        rnd_.shuffle(order)
        shown = []
        for d in order:
            v = rnd_.randrange(dims[d])
            state[d] = v
            shown.append((facts[d][v], rnd_.randint(2, 10)))
        d = rnd_.randrange(len(dims))
        out = multi_trial(b_, shown, qs[d], wait, acts, learn)
        return out, acts[d][state[d]]
    for _ in range(trials):
        out, want = episode(True, b, rnd)
        b.reward(1.0 if out == want else (-1.0 if out is not None else 0.0))
        for _ in range(20):
            if {a for a_s in acts for a in a_s} & b.step():
                b.reward(-1.0)
    t = copy.deepcopy(b)
    t.curiosity = None
    ex = random.Random(seed + 99)
    ok = 0
    n = 40
    for _ in range(n):
        out, want = episode(False, t, ex)
        ok += out == want
        for _ in range(20):
            t.step()
    return ok / n


def plateau_test(p, seed, steps=1200, stop=1000):
    """Один залипающий нейрон P: держит ли факт сам и гаснет ли по STOP."""
    b = Brain()
    b.add_region("входы")
    for n in ("CUE", "STOP"):
        b.add_neuron(n, region="входы")
    b.add_region("память")
    b.add_neuron("P", region="память")
    th = b.reach(b.neurons["P"].index)
    # факт — сильный вход (вдвое выше порога): залипание включает только он
    b.connect("CUE", "P", weight=2 * th, myelin_thickness=1.0)
    b.connect("STOP", "P", weight=2 * th, inhibitory=True, myelin_thickness=1.0)
    tasks.attach(b, p, seed)
    b.growth.sensory_regions = ["входы"]
    b.persistent = {b.neurons["P"].index}
    b.stimulate("CUE", 20.0)
    fires = []
    for t in range(steps):
        if t == stop:
            b.stimulate("STOP", 20.0)
        fires.append("P" in b.step())
    held = max((t for t in range(stop) if fires[t]), default=-1)
    return held, sum(fires[:stop]), any(fires[stop + 2:])


def ring_test(p, seed, full, steps, stop):
    b = make_ring_brain(p, seed, full)
    b.stimulate("CUE", 20.0)
    alive = []
    for t in range(steps):
        if t == stop:
            b.stimulate("STOP", 20.0)
        fired = b.step()
        alive.append(any(n.startswith("M") for n in fired))
    held = next((t for t in range(1, stop) if not any(alive[t:t + RING + 1])), stop)
    after = any(alive[stop + RING + 2:])
    w = [round(c.synapse.weight, 1) for c in b.connections
         if c.source.startswith("M") and c.target.startswith("M")]
    return held, after, w


def make_random_brain(p, seed, g, full):
    rnd = random.Random(seed)
    b = Brain()
    b.add_region("входы")
    for n in ("CUE_A", "CUE_B"):
        b.add_neuron(n, region="входы")
    b.add_region("память")
    mem = [f"M{k}" for k in range(N_MEM)]
    for n in mem:
        b.add_neuron(n, region="память")
    th = b.reach(b.neurons["M0"].index)
    for k, n in enumerate(mem):
        for m in rnd.sample([x for x in mem if x != n], OUT):
            d = rnd.randint(1, 4)
            b.connect(n, m, weight=g * th * rnd.uniform(0.5, 1.5),
                      myelin_thickness=(5.0 - d) / 4.0)
    if N_INH:
        # тормозные нейроны (общий механизм равновесия): их зовут нейроны
        # памяти, а они гасят случайных соседей — побеждает одна группа
        b.add_region("тормоз")
        for k in range(N_INH):
            iname = f"I{k}"
            b.add_neuron(iname, region="тормоз")
            for m in rnd.sample(mem, INH_IN):
                b.connect(m, iname, weight=0.5 * th, myelin_thickness=1.0)
            for m in rnd.sample(mem, INH_OUT):
                b.connect(iname, m, weight=INH_W * th, inhibitory=True, myelin_thickness=1.0)
    if PERSIST:
        b.persistent = {b.neurons[m].index for m in mem}
    order = rnd.sample(mem, 2 * CUE_SIZE)
    for cue, group in (("CUE_A", order[:CUE_SIZE]), ("CUE_B", order[CUE_SIZE:])):
        for m in group:
            b.connect(cue, m, weight=(2 * th if PERSIST else th + 4.0), myelin_thickness=1.0)
    if full:
        tasks.attach(b, p, seed)
        b.growth.sensory_regions = ["входы", "тормоз"]
        b.growth.pair_regions = []          # без нейронов-сочетаний: смотрим саму область
    return b


def trace(b, cue, steps=HOLD + 10):
    """Показать факт на копии; доля горящих по шагам и кто горел около шага HOLD."""
    t = copy.deepcopy(b)
    t.curiosity = None
    t.plateau = set()          # проверка одного факта с чистого состояния
    for _ in range(CUE_HOLD - 1):
        t.stimulate(cue, 20.0)
        t.step()
    t.stimulate(cue, 20.0)
    frac, late = [], set()
    for k in range(steps):
        f = [n for n in t.step() if n.startswith("M")]
        frac.append(len(f) / N_MEM)
        if HOLD - 5 <= k <= HOLD + 5:
            late |= set(f)
    return frac, late


def switch_test(b):
    """A, через 30 шагов B: что горит около шага HOLD после B — доля следа B
    и доля следа A среди горящих (следы — при одиночном показе)."""
    _, la = trace(b, "CUE_A")
    _, lb = trace(b, "CUE_B")
    t = copy.deepcopy(b)
    t.curiosity = None
    t.plateau = set()
    t.stimulate("CUE_A", 20.0)
    for _ in range(30):
        t.step()
    _, late = trace_from(t, "CUE_B")
    if not late:
        return 0.0, 0.0
    return round(len(late & lb) / len(late), 2), round(len(late & (la - lb)) / len(late), 2)


def trace_from(t, cue, steps=HOLD + 10):
    t.stimulate(cue, 20.0)
    frac, late = [], set()
    for k in range(steps):
        f = [n for n in t.step() if n.startswith("M")]
        frac.append(len(f) / N_MEM)
        if HOLD - 5 <= k <= HOLD + 5:
            late |= set(f)
    return frac, late


def random_test(p, seed, g, full, shows=300):
    """(когда, доля горящих около шага HOLD, наибольшая, горит у A, у B, общих) до и после показов."""
    b = make_random_brain(p, seed, g, full)
    out = []
    for stage in ("до", "после"):
        if stage == "после":
            rnd = random.Random(seed + 1)
            for _ in range(shows):
                cue = rnd.choice(("CUE_A", "CUE_B"))
                for _ in range(CUE_HOLD - 1):    # факт держится на входе несколько шагов
                    b.stimulate(cue, 20.0)
                    b.step()
                b.stimulate(cue, 20.0)
                for _ in range(HOLD + 30):
                    b.step()
        fa, la = trace(b, "CUE_A")
        fb, lb = trace(b, "CUE_B")
        both = len(la & lb) / max(1, len(la | lb))
        out.append((stage, round(max(fa[HOLD - 5:HOLD + 5]), 2), round(max(fa), 2), len(la), len(lb), round(both, 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--plateau", action="store_true")
    ap.add_argument("--delay", action="store_true")
    ap.add_argument("--multi", default="", help="число значений по измерениям, например 4 или 2,2")
    ap.add_argument("--wait", type=int, default=50)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--stop", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    if a.multi:
        dims = [int(x) for x in a.multi.split(",")]
        print("измерения", dims, "задержка", a.wait, "верно на экзамене",
              multi_run(p, a.seed, dims, a.wait, a.trials, not a.no_persist))
        return
    if a.delay:
        rows = delay_run(p, a.seed, a.wait, a.trials, not a.no_persist, a.control)
        ok = all(w == o for _, w, o in rows)
        print("задержка", a.wait, "выучено" if ok else "нет", rows)
        return
    if a.plateau:
        held, n, after = plateau_test(p, a.seed)
        print(f"залипающий нейрон: последний раз горел до STOP на шаге {held} (STOP на 1000), "
              f"срабатываний {n}; после STOP горит: {after}")
        return
    if a.random:
        print("g, механизмы: (когда, доля горящих около шага 50, наибольшая доля, горит у A, у B, общих)")
        for full in (False, True):
            for g in (0.5, 0.8, 1.0, 1.3, 1.7):
                print(g, "все" if full else "голое", random_test(p, a.seed, g, full))
        return
    for full in (False, True):
        held, after, w = ring_test(p, a.seed, full, a.steps, a.stop)
        print(f"{'все механизмы' if full else 'голое ядро':14s} горело само до шага {held} "
              f"(STOP на {a.stop}); после STOP горит: {after}; связи кольца {w}")


if __name__ == "__main__":
    main()
