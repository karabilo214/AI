"""Голова: одно устройство мозга для всех задач (задаётся один раз, на глаз).

Задача не подключает нейроны и связи; она только говорит, что показать
(see) или проиграть (hear), и решает, наградить ли действие (act).

Устройство:
    зрение         V0..V7   вход (картинка = несколько нейронов сразу)
    слух           Aa..Af   вход (слово = звуки один за другим)
    действия       D0..D3   выходы, соревнуются через торможение (врождённое)
    память         M0..M7   заряд почти не стекает; Vi слабо заряжает Mi,
                            любой звук слабо подталкивает всю память
    память слуха   нейроны порядка звуков растут сами (Hearing)
Связи при рождении (одни правила для всех задач):
    зрение -> действия    все со всеми, сила init_weight / 2
    память -> действия    все со всеми, сила init_weight / 2
    Vi -> Mi              0.75 расстояния до порога (сам до порога не доводит)
    звук -> память        0.5 расстояния до порога каждому
Всегда включены: рост (Growth), ассоциации зрение <-> память слуха
(Association), память слуха (Hearing), знакомость, любопытство,
изменение нейронов, забывание — как в tasks.attach.

Правки 2026-11-09 (каждую можно выключить, FIXES):
    budget     запас силы входов считается только на входы, сигнал по
               которым пришёл за последние BUDGET_WINDOW шагов
    pairs      нейроны-сочетания только из зрения, слуха и памяти
               (не из действий и не из других сочетаний)
    inhibit    обучаемое торможение зрение -> действия: в начале 0;
               действие сразу после сигнала источника (INH_WINDOW шагов)
               и наказание -> торможение сильнее, награда -> слабее
    same_mem   память и память слуха без случайного разброса порогов
    one_event  знакомость: показанное на одном шаге — одно событие
               (иначе нейроны одной картинки сбрасывали привыкание друг другу)

Контекст (схема владельца, 2026-11-09), вместо M0..M7:
    wm         рабочая память W0..W15 без заданных значений: случайные редкие
               учащиеся связи от зрения и слуха (доля WM_P, сила WM_W), соревнование
               (взаимное торможение); у каждого W медленный след m (WM_DECAY за шаг).
               Память не возбуждает действия, а меняет силу связей вход -> действие:
               сила = w + сумма m_k * g_k (плавные контекстные ветви, kernel), g учится
               по награде пропорционально m. Слух тоже связан с действиями.
    rpe        связи учатся на ошибку предсказания награды (RewardPrediction)
    ip_push    порог подстраивается по полученному толчку, а не по срабатыванию
    floor      торможение не опускает заряд ниже покоя − FLOOR_BELOW_REST
    boredom    ловушка молчания: попытки с результатом 0 подряд поднимают постоянный
               толчок действиям (до BOREDOM мВ за шаг; сбрасывает только награда
               или наказание)
    recruit    найм памяти (Recruit.py): новое событие получает свой нейрон
               рабочей памяти; связи памяти тогда задаёт найм (learn = 0)

Правки по разбору кода (2026-11-09):
    elig_reset   награда сбрасывает метки (иначе ~2/3 метки доставалось
                 следующей попытке)
    last_action  новое действие стирает метки прошлого (рефлекс в паузе не
                 получает награду за ответ после неё)
    decision_ctx награда учит контекст момента решения, а не момента награды
    inh_short    торможение снимает текущий заряд, но заряда ниже покоя после
                 шага не остаётся (раньше соперник ~10 шагов не мог ответить)
    same_act     у действий тоже нет случайного разброса порогов (иначе
                 постоянный перекос выбора)
    myelin_keep  сигнал, пришедший вовремя, поддерживает миелин (иначе связь
                 «вовремя» в долгих паузах таяла, задержка росла, и закреплённый
                 нейрон памяти узнавал событие на шаг позже)
В ядре без выключателя: с ошибкой предсказания правила, смотрящие на знак
(запрет, сомнения, важность), берут знак у самой награды; у плавных контекстов
общий вес учится долей 1 - сумма m; обучаемое торможение зависит от контекста.
"""
import copy

import numpy as np

from human_brain.stages import tasks
from human_brain.core.Brain import Brain
from human_brain.core.Neuron import Neuron
from human_brain.core.Regions import add_standard_regions
from human_brain.parts.Hearing import Hearing
from human_brain.parts.Association import Association
from human_brain.core.RewardPrediction import RewardPrediction
from human_brain.parts.Recruit import Recruit

N_VISION = 8
SOUNDS = "abcdef"
N_ACTIONS = 4
N_MEMORY = 8

FIXES = {"budget", "pairs", "inhibit", "same_mem", "one_event",
         "wm", "rpe", "ip_push", "floor", "boredom", "recruit",
         "elig_reset", "last_action", "decision_ctx", "inh_short", "same_act",
         "myelin_keep"}   # что включено
FLOOR_BELOW_REST = 10.0  # покой −70 -> граница −80
BOREDOM = 1.3            # при стекании 0.85 полная скука даёт 1.3 / 0.15 = 8.7 мВ (до порога 8)
N_WM = 16
WM_P = 0.6              # доля входов к каждому W: без нейрона памяти 2 картинки из 80 (при 0.4 — 21)
WM_W = (2.0, 6.0)       # их начальная сила, равномерно (порог ~8 выше покоя)
WM_W_FREE = (1.0, 2.5)  # с наймом: свободный нейрон сам не срабатывает (3 входа < 8),
                        # иначе он перебивает закреплённого и каждое повторение тратит новый
WM_DECAY = 0.9          # след: через 10 шагов (самая длинная пауза в задачах) 0.35,
                        # через 29 шагов ниже 0.05 — к следующей попытке гаснет
                        # (при 0.98 через 36 шагов оставалось ~0.5)
BUDGET_WINDOW = 10      # шагов: «недавно работали вместе»
INH_WINDOW = 3          # шагов от сигнала источника до действия
INH_PLASTICITY = 5.0    # скорость обучения торможения (на глаз)
REAL_REGIONS = ("зрение", "слух", "память", "память слуха", "рабочая память")


def build_head(p, seed):
    tasks.apply(p)
    b = Brain()
    add_standard_regions(b)
    for i in range(N_VISION):
        b.add_neuron(f"V{i}", region="зрение")
    for s in SOUNDS:
        b.add_neuron(f"A{s}", region="слух")
    for i in range(N_ACTIONS):
        b.add_neuron(f"D{i}", explores=True, region="действия")
    wm = "wm" in FIXES
    if wm:
        b.add_region("рабочая память", learn=0.0 if "recruit" in FIXES else 1.0)
        for i in range(N_WM):
            b.add_neuron(f"W{i}", region="рабочая память", context=True)
    else:
        for i in range(N_MEMORY):
            b.add_neuron(f"M{i}", region="память")

    reach = b.rules["Neuron"]["THRESHOLD"] - b.rules["Neuron"]["REST"]
    for i in range(N_VISION):
        for k in range(N_ACTIONS):
            b.connect(f"V{i}", f"D{k}", weight=p["init_weight"] / 2, myelin_thickness=1.0,
                      plasticity=p["init_plasticity"])
    if wm:
        rng = np.random.default_rng(seed)
        inputs = [f"V{i}" for i in range(N_VISION)] + [f"A{s}" for s in SOUNDS]
        for s in SOUNDS:
            for k in range(N_ACTIONS):
                b.connect(f"A{s}", f"D{k}", weight=p["init_weight"] / 2, myelin_thickness=1.0,
                          plasticity=p["init_plasticity"])
        for i in range(N_WM):
            for x in inputs:
                if rng.random() < WM_P:
                    b.connect(x, f"W{i}", weight=float(rng.uniform(*(WM_W_FREE if "recruit" in FIXES else WM_W))),
                              myelin_thickness=1.0)
            for j in range(N_WM):
                if i != j:
                    b.connect(f"W{i}", f"W{j}", weight=p["inhibit_weight"], inhibitory=True)
    else:
        for i in range(N_MEMORY):
            for k in range(N_ACTIONS):
                b.connect(f"M{i}", f"D{k}", weight=p["init_weight"] / 2, myelin_thickness=1.0,
                          plasticity=p["init_plasticity"])
        for i in range(N_VISION):
            b.connect(f"V{i}", f"M{i}", weight=0.75 * reach, myelin_thickness=1.0)
        for s in SOUNDS:
            for i in range(N_MEMORY):
                b.connect(f"A{s}", f"M{i}", weight=0.5 * reach, myelin_thickness=1.0)
    for k in range(N_ACTIONS):
        for j in range(N_ACTIONS):
            if k != j:
                b.connect(f"D{k}", f"D{j}", weight=p["inhibit_weight"], inhibitory=True)

    if "inhibit" in FIXES:
        # только от зрения: у V -> D уже есть связи, а лишняя связь A -> D
        # помешала бы росту (Growth не растит связь, если пара уже связана)
        for i in range(N_VISION):
            for k in range(N_ACTIONS):
                b.connect(f"V{i}", f"D{k}", weight=0.0, inhibitory=True,
                          plasticity=INH_PLASTICITY)
        b.inh_window = INH_WINDOW

    tasks.attach(b, p, seed)
    b.growth.max_neurons = b.n_neurons + 12     # место для новых нейронов-сочетаний
    if "budget" in FIXES and b.input_budget > 0:
        b.budget_window = BUDGET_WINDOW
    if "pairs" in FIXES:
        b.growth.pair_regions = sorted(REAL_REGIONS)
    if "one_event" in FIXES and b.familiarity is not None:
        b.familiarity.one_event = True
    if "same_mem" in FIXES and b.neuron_change is not None:
        b.neuron_change.make_same(("память", "память слуха", "рабочая память"))
    if "same_act" in FIXES and b.neuron_change is not None:
        b.neuron_change.make_same(("действия",))
    b.reset_eligibility = "elig_reset" in FIXES
    b.last_action_only = "last_action" in FIXES
    b.decision_ctx = "decision_ctx" in FIXES
    b.inh_short = "inh_short" in FIXES
    b.myelin_keep = "myelin_keep" in FIXES
    if wm:
        b.ctx_graded = WM_DECAY
    if "rpe" in FIXES:
        RewardPrediction(b, ("зрение", "слух", "рабочая память" if wm else "память"))
    if "ip_push" in FIXES:
        b.ip_push = True
    if "floor" in FIXES:
        b.v_floor = b.rest - FLOOR_BELOW_REST
    if "boredom" in FIXES and b.curiosity is not None:
        b.curiosity.boredom = BOREDOM
    if "recruit" in FIXES and wm:
        Recruit(b)
    Hearing(b)
    Association(b, "зрение", "память слуха", scale=reach + 1.0)
    return b


# ---------- что задача может делать с головой ----------

def see(b, picture):
    for i in picture:
        b.stimulate(f"V{i}", 20.0)


def hear(b, sound):
    b.stimulate(f"A{sound}", 20.0)


def is_action(name):
    """Нейрон-действие D0..D3 (сочетания вроде «D1+V2» действиями не считаются)."""
    return name[:1] == "D" and name[1:].isdigit()


def action(fired):
    """Какое действие выбрано на этом шаге: None, имя D или 'несколько'."""
    acts = sorted(n for n in fired if is_action(n))
    if not acts:
        return None
    return acts[0] if len(acts) == 1 else "несколько"


def quiet_copy(b):
    t = copy.deepcopy(b)
    t.curiosity = None
    return t
