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
    for s in mem + ["GO"]:
        for o in ("YES", "NO"):
            b.connect(s, o, weight=p["init_weight"], myelin_thickness=1.0 if s != "GO" else 0.75,
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
    b.neuron_change.make_same(("входы", "действия"))
    b.curiosity.boredom = p.get("boredom", Head.BOREDOM)
    b.curiosity.avoid = p.get("avoid", 3.0)
    b.curiosity.situational = True
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


def delay_run(p, seed, wait=50, trials=600, persist=True, control=False):
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
    ap.add_argument("--wait", type=int, default=50)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--stop", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
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
