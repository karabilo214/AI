"""Параметры мозга и набор проверок для их подбора (см. evolve.py).

Параметры: все числа, заданные вручную: порог, стекание и отдых нейрона,
пределы и скорости связей, миелин, начальные силы связей и торможения
в задачах, любопытство, рост, запас входов, изменение нейронов.

PARAMS_DEFAULT   выбранные значения всех параметров (эволюция 2026-11-09, набор
                 с гипотезами, точной заслугой и миелином по времени, плюс
                 контекстные ветви; см. params_chosen.json; затем эволюция под
                 сложение evolve_addition.py, 16 параметров изменены, см.
                 evolve_addition_results/best.json, решение владельца;
                 гипотезы hyp_on оставлены включёнными: без них ложная
                 память 10/10). Они же записаны
                 как исходные в Neuron, Synapse, Myelin, Growth, Curiosity,
                 NeuronChange, Familiarity и Brain. Повторно подбирать без
                 нужды не надо.
SPACE            для каждого параметра: вид и пределы перебора.
apply(p)         выставляет правила (константы классов) из набора p.
evaluate(p, seeds) оценка набора: сумма долей пройденных проверок.

Проверки (каждая даёт число от 0 до 1):
    reward   средняя доля по 8 задачам по награде (в конце верны все части):
             одна задача (на L и на R), A и A зеркальная, «одинаковый ответ»
             (на L и на R), B и B зеркальная. Набор сбалансирован: заученная
             стратегия вроде «на новое отвечай другим выходом» проваливает
             половину задач (первая версия набора была несбалансированной,
             и эволюция нашла такую стратегию);
    memory   A, через 1..4 шага B: к 10-му повтору A одна зажигает B,
             и после паузы 2000 шагов это сохраняется;
    growth   на шуме не растут ложные связи, настоящая пара находится;
    grass    «трава»: знакомое без последствий перестают замечать, новое
             замечают, знакомое снова замечают, если за ним пришла награда.
Переучивание (S1 -> L, затем S1 -> R) в оценку не входит: медленное
переучивание задумано (закреплённое меняется трудно).
"""
import copy
import random

from human_brain.core.Brain import Brain
from human_brain.core.Neuron import Neuron
from human_brain.core.Synapse import Synapse
from human_brain.core.Myelin import Myelin
from human_brain.parts.Curiosity import Curiosity
from human_brain.parts.Growth import Growth
from human_brain.core.NeuronChange import NeuronChange
from human_brain.parts.Familiarity import Familiarity

PARAMS_DEFAULT = {
    # нейрон
    "threshold": -62.0, "decay": 0.85, "refractory": 1,
    # связь
    "max_weight": 24.303764, "consolidate_weight": 17.655948, "default_thickness": 0.309776,
    # начальное устройство задач по награде
    "init_weight": 14.0, "init_plasticity": 43.465607, "inhibit_weight": 28.905862,
    # синапс
    "strengthen": 0.022685, "weaken": 0.185019, "trace_decay": 0.658229,
    "reward_rate": 1.0, "eligibility_decay": 0.99,
    # миелин
    "myelin_grow": 0.024373, "myelin_shrink": 0.006042, "myelin_fade": 9.9e-05,
    "suppress_steps": 3, "protect_inhibited": True,
    # любопытство
    "calm": 0.141609, "curious": 1.434077, "curiosity_up": 0.05, "curiosity_down": 1.0,
    "curiosity_decay": 0.998041,
    # рост
    "connection_threshold": 3, "min_reliability": 0.443501, "min_lift": 3.328922,
    "explain_weight": 6.153709, "young_plasticity": 100.0, "new_weight": 1.452021,
    "pair_weight": 11.127045, "neuron_threshold": 2, "fast_new_outputs": True,
    "dead_weight": 2.0, "dead_patience": 92,
    # запас силы входов
    "budget_on": True, "input_budget": 40.811013, "protect_consolidated": True,
    # изменение нейронов
    "neurons_on": True, "nc_rate": 0.030059, "nc_target": 0.043085, "nc_threshold_spread": 3.468095,
    # знакомость
    "fam_on": True, "fam_habituation": 0.05, "fam_max_ignore": 0.717818,
    "fam_trace_decay": 0.756039, "fam_dishabituation": 1.0, "fam_explain": 11.899689,
    # гипотезы, заслуга, миелин по времени
    "hyp_on": True, "hyp_needed": 5, "hyp_min_rel": 0.818103, "hyp_lift": 2.186701,
    "credit_on": True, "credit_horizon": 15, "myelin_timing": True, "timing_window": 4,
    "myelin_idle": 501,
    # контекстные ветви
    "ctx_on": True, "ctx_horizon": 13, "fam_recovery": 0.999859,
    # управляемое забывание (выбрано вручную, без эволюции)
    "forget_on": True, "imp_decay": 0.9999, "important": 3.0, "doubt_needed": 3,
    "relearn_plast": 20.0, "consolidated_rate": 1.0,
    # защита путей к действиям (выбрано вручную)
    "action_min": 1.579161, "curiosity_idle": 0.2, "neg_rate": 0.05, "neg_needed": 0,
    "necessity_on": False,
    # изменчивый запрет (на глаз)
    "ban_on": False, "ban_up": 0.5, "ban_down": 0.465469, "ban_power": 7.591039,
}

# вид: log (множитель), lin (сдвиг), int, bool, one_minus (множитель для 1 - x)
SPACE = {
    "threshold": ("lin", -62.0, -50.0), "decay": ("lin", 0.85, 0.99),
    "refractory": ("int", 1, 4),
    "max_weight": ("lin", 15.0, 60.0), "consolidate_weight": ("lin", 8.0, 30.0),
    "default_thickness": ("lin", 0.0, 1.0),
    "init_weight": ("lin", 2.0, 14.0), "init_plasticity": ("log", 1.0, 45.0),
    "inhibit_weight": ("lin", 5.0, 40.0),
    "strengthen": ("log", 0.005, 0.5), "weaken": ("log", 0.005, 0.5),
    "trace_decay": ("lin", 0.3, 0.95), "reward_rate": ("log", 0.01, 1.0),
    "eligibility_decay": ("lin", 0.5, 0.99),
    "myelin_grow": ("log", 0.001, 0.1), "myelin_shrink": ("log", 0.0005, 0.05),
    "myelin_fade": ("log", 0.00001, 0.001), "suppress_steps": ("int", 1, 6),
    "protect_inhibited": ("bool",),
    "calm": ("log", 0.05, 2.0), "curious": ("log", 0.3, 5.0),
    "curiosity_up": ("lin", 0.05, 1.0), "curiosity_down": ("lin", 0.05, 1.0),
    "curiosity_decay": ("one_minus", 0.0001, 0.01),
    "connection_threshold": ("int", 2, 10), "min_reliability": ("lin", 0.3, 0.95),
    "min_lift": ("log", 1.5, 10.0), "explain_weight": ("lin", 5.0, 30.0),
    "young_plasticity": ("log", 5.0, 100.0), "new_weight": ("lin", 1.0, 12.0),
    # pair_weight не перебирается: входы нейрона-сочетания теперь задаются от
    # его порога (Growth.PAIR_SHARE), параметр ни на что не влияет
    "neuron_threshold": ("int", 2, 10),
    "fast_new_outputs": ("bool",), "dead_weight": ("log", 0.1, 2.0),
    "dead_patience": ("int", 20, 500),
    "budget_on": ("bool",), "input_budget": ("lin", 15.0, 60.0),
    "protect_consolidated": ("bool",),
    "neurons_on": ("bool",), "nc_rate": ("log", 0.005, 0.5),
    "nc_target": ("log", 0.005, 0.1), "nc_threshold_spread": ("lin", 0.0, 4.0),
    "fam_on": ("bool",), "fam_habituation": ("log", 0.005, 0.3),
    "fam_max_ignore": ("lin", 0.1, 1.0), "fam_trace_decay": ("lin", 0.3, 0.97),
    "fam_dishabituation": ("log", 0.1, 3.0), "fam_explain": ("lin", 5.0, 30.0),
    "hyp_on": ("bool",), "hyp_needed": ("int", 2, 12), "hyp_min_rel": ("lin", 0.3, 0.95),
    "hyp_lift": ("log", 1.0, 10.0), "credit_on": ("bool",), "credit_horizon": ("int", 2, 20),
    "myelin_timing": ("bool",), "timing_window": ("int", 1, 6),
    "myelin_idle": ("int", 50, 5000),
    "ctx_on": ("bool",), "ctx_horizon": ("int", 3, 20),
    "fam_recovery": ("one_minus", 0.00001, 0.01),
    "forget_on": ("bool",), "imp_decay": ("one_minus", 0.00001, 0.01),
    "important": ("log", 0.5, 20.0), "doubt_needed": ("int", 2, 10),
    "relearn_plast": ("log", 2.0, 100.0), "consolidated_rate": ("log", 0.005, 1.0),
    "action_min": ("lin", 0.0, 5.0), "curiosity_idle": ("lin", 0.0, 1.0),
    "neg_rate": ("log", 0.005, 1.0), "neg_needed": ("int", 1, 8),
    "necessity_on": ("bool",),
    "ban_on": ("bool",), "ban_up": ("lin", 0.05, 1.0), "ban_down": ("lin", 0.05, 1.0),
    "ban_power": ("lin", 1.0, 12.0),
}


def apply(p):
    """Выставляет числа нейрона, связи, синапса и миелина (константы классов)."""
    Neuron.THRESHOLD = p["threshold"]
    Neuron.DECAY = p["decay"]
    Neuron.REFRACTORY = int(p["refractory"])
    Synapse.MAX_WEIGHT = p["max_weight"]
    Synapse.CONSOLIDATE_WEIGHT = p["consolidate_weight"]
    Myelin.THICKNESS = p["default_thickness"]
    Synapse.STRENGTHEN = p["strengthen"]
    Synapse.WEAKEN = p["weaken"]
    Synapse.TRACE_DECAY = p["trace_decay"]
    Synapse.REWARD_RATE = p["reward_rate"]
    Synapse.ELIGIBILITY_DECAY = p["eligibility_decay"]
    Myelin.GROW = p["myelin_grow"]
    Myelin.SHRINK = p["myelin_shrink"]
    Myelin.FADE = p["myelin_fade"]
    Myelin.SUPPRESS_STEPS = int(p["suppress_steps"])
    Myelin.PROTECT_INHIBITED = bool(p["protect_inhibited"])
    Synapse.CREDIT = bool(p["credit_on"])
    Synapse.CREDIT_HORIZON = int(p["credit_horizon"])
    Myelin.TIMING = bool(p["myelin_timing"])
    Myelin.TIMING_WINDOW = int(p["timing_window"])
    Myelin.IDLE_STEPS = int(p["myelin_idle"])
    Synapse.CONTEXTS = bool(p["ctx_on"])
    Synapse.CONTEXT_HORIZON = int(p["ctx_horizon"])
    Synapse.FORGET = bool(p["forget_on"])
    Synapse.IMPORTANCE_DECAY = p["imp_decay"]
    Synapse.IMPORTANT = p["important"]
    Synapse.DOUBT_NEEDED = int(p["doubt_needed"])
    Synapse.RELEARN_PLASTICITY = p["relearn_plast"]
    Synapse.CONSOLIDATED_RATE = p["consolidated_rate"]
    Synapse.ACTION_MIN_WEIGHT = p["action_min"]
    Synapse.NEG_RATE = p["neg_rate"]
    Synapse.NEG_NEEDED = int(p["neg_needed"])
    Synapse.NECESSITY = bool(p["necessity_on"])
    Synapse.BAN = bool(p["ban_on"])
    Synapse.BAN_UP = p["ban_up"]
    Synapse.BAN_DOWN = p["ban_down"]
    Synapse.BAN_POWER = p["ban_power"]


def attach(b, p, seed, curiosity=True, neurons=True):
    """Подключает к мозгу рост, любопытство, запас и изменение нейронов по набору p."""
    Growth(b, connection_threshold=int(p["connection_threshold"]),
           min_reliability=p["min_reliability"], min_lift=p["min_lift"],
           explain_weight=p["explain_weight"], young_plasticity=p["young_plasticity"],
           new_weight=p["new_weight"], pair_weight=p["pair_weight"],
           neuron_threshold=int(p["neuron_threshold"]),
           fast_new_outputs=bool(p["fast_new_outputs"]), max_neurons=12,
           hypotheses=bool(p["hyp_on"]), hyp_needed=int(p["hyp_needed"]),
           hyp_min_rel=p["hyp_min_rel"], hyp_lift=p["hyp_lift"],
           dead_weight=p["dead_weight"], dead_patience=int(p["dead_patience"]))
    if curiosity:
        Curiosity(b, calm=p["calm"], curious=p["curious"], up=p["curiosity_up"],
                  down=p["curiosity_down"], decay=p["curiosity_decay"],
                  idle=p["curiosity_idle"], seed=seed)
    b.input_budget = p["input_budget"] if p["budget_on"] else 0.0
    b.protect_consolidated = bool(p["protect_consolidated"])
    if p["fam_on"]:
        Familiarity(b, habituation=p["fam_habituation"], max_ignore=p["fam_max_ignore"],
                    trace_decay=p["fam_trace_decay"], dishabituation=p["fam_dishabituation"],
                    explain_weight=p["fam_explain"], recovery=p["fam_recovery"])
    if neurons and p["neurons_on"]:
        NeuronChange(b, rate=p["nc_rate"], target=p["nc_target"],
                     threshold_spread=p["nc_threshold_spread"], seed=seed)


# ---------- задачи по награде ----------

def make_reward_brain(p, seed):
    b = Brain()
    for n in ("S1", "S2"):
        b.add_neuron(n)
    for n in ("L", "R"):
        b.add_neuron(n, explores=True)
    for s in ("S1", "S2"):
        for o in ("L", "R"):
            b.connect(s, o, weight=p["init_weight"], myelin_thickness=1.0,
                      plasticity=p["init_plasticity"])
    b.connect("L", "R", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("R", "L", weight=p["inhibit_weight"], inhibitory=True)
    attach(b, p, seed)
    return b


def trial(b, inputs):
    """Толкнуть входы; ответ = первый выход, загоревшийся за 7 шагов
    (смотрим с первого шага после показа)."""
    for s in inputs:
        b.stimulate(s, 20.0)
    for _ in range(7):
        f = b.step()
        if "L" in f or "R" in f:
            return "L" if "R" not in f else "R" if "L" not in f else "оба"
    return "никто"


def rest(b, steps=20):
    for _ in range(steps):
        b.step()


def exam(b, inputs):
    """Ответ без шума любопытства, на копии (мозг не меняется)."""
    t = copy.deepcopy(b)
    t.curiosity = None
    return trial(t, inputs)


def learn(b, inputs, good, n=60):
    hits = 0
    for _ in range(n):
        out = trial(b, inputs)
        hits += out == good
        b.reward(+1 if out == good else -1 if out in ("L", "R", "оба") else 0)
        rest(b)
    return hits


TASKS = {
    "single_L": [(("S1",), "L")],
    "single_R": [(("S1",), "R")],
    "A": [(("S1",), "L"), (("S2",), "R")],
    "A_mirror": [(("S1",), "R"), (("S2",), "L")],
    "same_L": [(("S1",), "L"), (("S2",), "L")],
    "same_R": [(("S1",), "R"), (("S2",), "R")],
    "B": [(("S1",), "L"), (("S1", "S2"), "R")],
    "B_mirror": [(("S1",), "R"), (("S1", "S2"), "L")],
}
REWARD_TASKS = list(TASKS)
# переучивание: только для сведения, в оценку не входит
TASK_C = [(("S1",), "L"), (("S1",), "R")]


def run_task(p, name, seed):
    """Возвращает (все части верны в конце, доля верных ответов при обучении,
    верна ли вторая часть ещё до её обучения или None)."""
    parts = TASKS[name]
    b = make_reward_brain(p, seed)
    hits, before = [], None
    for k, (inputs, good) in enumerate(parts):
        if k == 1:
            before = exam(b, inputs) == good
        hits.append(learn(b, inputs, good))
    ok = all(exam(b, inputs) == good for inputs, good in parts)
    return ok, sum(hits) / (60 * len(hits)), before


# ---------- память и рост ----------

def memory_test(p):
    """Доля промежутков 1..4, где к 10-му повтору A одна зажигает B и помнит после паузы."""
    good = 0
    for gap in (1, 2, 3, 4):
        b = Brain()
        b.add_neuron("A")
        b.add_neuron("B")
        attach(b, p, seed=0, curiosity=False, neurons=False)
        for rep in range(1, 11):
            b.stimulate("A", 20.0)
            for s in range(30):
                if s == gap:
                    b.stimulate("B", 20.0)
                b.step()
        t = copy.deepcopy(b)
        before = t.neurons["B"].spikes
        t.stimulate("A", 20.0)
        rest(t, 10)
        ok10 = t.neurons["B"].spikes > before
        rest(b, 2000)
        before = b.neurons["B"].spikes
        b.stimulate("A", 20.0)
        rest(b, 10)
        good += ok10 and b.neurons["B"].spikes > before
    return good / 4


def growth_test(p, seed, steps=2000):
    """Шум 10 нейронов (2% шагов) и пара A -> B (B через 2 шага после A).

    Балл = (пара найдена ? 1 : 0) * (1 - 0.25 * число ложных связей в обоих прогонах),
    не меньше 0.
    """
    found, false = False, 0
    for link in (True, False):
        rnd = random.Random(seed)
        names = ["A", "B"] + [f"N{i}" for i in range(10)]
        b = Brain()
        for n in names:
            b.add_neuron(n)
        attach(b, p, seed, curiosity=False, neurons=False)
        plan = {}
        for t in range(steps):
            for n in names[2:]:
                if rnd.random() < 0.02:
                    b.stimulate(n, 20.0)
            if rnd.random() < 0.02:
                b.stimulate("A", 20.0)
                plan[t + 2] = True
            if link and plan.pop(t, False):
                b.stimulate("B", 20.0)
            b.step()
        # считаются только связи, которые в конце живы и подтверждены
        grown = [(c.source, c.target) for c in b.growth.grown_connections
                 if c.alive and not c.hypothesis]
        false += sum(1 for g in grown if g != ("A", "B"))
        if link:
            found = ("A", "B") in grown
    return (1.0 if found else 0.0) * max(0.0, 1.0 - 0.25 * false)


# ---------- контексты ----------
# Входы S1, S2 (стимулы) и C1, C2 (контексты), выходы L, R. Задача = блоки,
# блок = список пар (входы, ответ), попытки внутри блока идут по очереди.
# Блоки учатся один за другим; в конце проверяются все пары всех блоков.
CONTEXT_TASKS = {
    "ctx_simple": [[(("S1", "C1"), "L")], [(("S1", "C2"), "R")]],
    "ctx_simple_mirror": [[(("S1", "C1"), "R")], [(("S1", "C2"), "L")]],
    # разворот: ответ задаёт только сочетание стимула и контекста
    "ctx_reverse": [[(("S1", "C1"), "L"), (("S2", "C1"), "R")],
                    [(("S1", "C2"), "R"), (("S2", "C2"), "L")]],
    "ctx_reverse_mirror": [[(("S1", "C1"), "R"), (("S2", "C1"), "L")],
                           [(("S1", "C2"), "L"), (("S2", "C2"), "R")]],
}
CONTEXT_INPUTS = ("S1", "S2", "C1", "C2")


def make_context_brain(p, seed):
    b = Brain()
    for n in CONTEXT_INPUTS:
        b.add_neuron(n, context=n.startswith("C"))
    for n in ("L", "R"):
        b.add_neuron(n, explores=True)
    for s in CONTEXT_INPUTS:
        if p["ctx_on"] and s.startswith("C"):
            continue   # с ветвями контекст только настраивает пути, действием не командует
        for o in ("L", "R"):
            b.connect(s, o, weight=p["init_weight"], myelin_thickness=1.0,
                      plasticity=p["init_plasticity"])
    b.connect("L", "R", weight=p["inhibit_weight"], inhibitory=True)
    b.connect("R", "L", weight=p["inhibit_weight"], inhibitory=True)
    attach(b, p, seed)
    return b


def learn_block(b, items, n=60):
    hits = 0
    for k in range(n):
        inputs, good = items[k % len(items)]
        out = trial(b, inputs)
        hits += out == good
        b.reward(+1 if out == good else -1 if out in ("L", "R", "оба") else 0)
        rest(b)
    return hits


def run_context_task(p, name, seed, n=60):
    """Возвращает (все пары верны в конце, доля верных при обучении,
    доля пар первого блока, верных в конце: помнит ли старый контекст)."""
    blocks = CONTEXT_TASKS[name]
    b = make_context_brain(p, seed)
    hits = sum(learn_block(b, items, n) for items in blocks)
    ok = all(exam(b, i) == g for items in blocks for i, g in items)
    first = sum(exam(b, i) == g for i, g in blocks[0]) / len(blocks[0])
    return ok, hits / (n * len(blocks)), first


def grass_test(p, presentations=200):
    """«Трава»: X показывают много раз без последствий, потом новый Y,
    потом X вместе с наградой. Возвращает (сколько из первых 10 показов X
    сработал, сколько из последних 10, заметили ли Y, заметили ли снова X)."""
    b = Brain()
    for n in ("X", "Y", "O"):
        b.add_neuron(n)
    b.connect("X", "O", weight=18.0)
    b.connect("Y", "O", weight=18.0)
    attach(b, p, seed=0, curiosity=False, neurons=False)
    early = late = 0
    for k in range(presentations):
        before = b.neurons["X"].spikes
        b.stimulate("X", 20.0)
        rest(b, 30)
        fired = b.neurons["X"].spikes > before
        early += k < 10 and fired
        late += k >= presentations - 10 and fired
    before = b.neurons["Y"].spikes
    b.stimulate("Y", 20.0)
    rest(b, 30)
    y_noticed = b.neurons["Y"].spikes > before
    for _ in range(5):
        b.stimulate("X", 20.0)
        rest(b, 3)
        b.reward(1.0)
        rest(b, 27)
    before = b.neurons["X"].spikes
    b.stimulate("X", 20.0)
    rest(b, 30)
    return early, late, y_noticed, b.neurons["X"].spikes > before


def grass_score(p):
    early, late, y_noticed, regained = grass_test(p)
    ignored = early >= 8 and late <= 1
    return (float(ignored) + float(y_noticed) + float(regained)) / 3.0


def evaluate(p, seeds):
    """Оценка набора параметров. Возвращает (оценка, подробности)."""
    apply(p)
    detail = {}
    hits, before = [], []
    for name in REWARD_TASKS:
        res = [run_task(p, name, s) for s in seeds]
        detail[name] = sum(ok for ok, _, _ in res) / len(seeds)
        hits += [h for _, h, _ in res]
        before += [b for _, _, b in res if b is not None]
    detail["reward"] = sum(detail[n] for n in REWARD_TASKS) / len(REWARD_TASKS)
    detail["memory"] = memory_test(p)
    detail["growth"] = growth_test(p, seed=seeds[0])
    detail["grass"] = grass_score(p)
    detail["hits"] = sum(hits) / len(hits)
    detail["before"] = sum(before) / len(before)   # для сведения: верно ли до обучения
    score = (3 * detail["reward"] + detail["memory"] + detail["growth"] + detail["grass"]
             + 0.1 * detail["hits"])
    return score, detail
