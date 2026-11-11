"""Задача «сова» (similarity.py) на нашем импульсном мозге: обучение в две фазы.

Мозг: наши нейроны и связи (Brain, ядро kernel), задержка связей 1 шаг.
Слои: животные (9) -> образ (N_REP) ; образ и глаголы (3) -> скрытый (N_HID)
-> слова (9). У каждой прямой связи есть обратная (слово -> скрытый,
скрытый -> образ) с силой FB * прямая. Слова тормозят друг друга
(неизменное торможение). Прежние правила обучения выключены (plasticity 0,
без роста и наград): видно только новое правило.

Проба = две фазы по W шагов (животное и глагол толкаются каждый шаг):
    1. свободная — мозг сам приходит к ответу;
    2. с ответом — верное слово ещё и «слышно» (толчок каждый шаг); его
       влияние идёт назад по обратным связям, торможение гасит другие слова.
Активность нейрона = доля шагов со срабатыванием (за последние W - SETTLE
шагов фазы). Правило (местное): каждая прямая связь i -> j меняется на
ETA * (r_i r_j с ответом - r_i r_j свободно), сила в [0, W_MAX]; обратная
связь = FB * прямая (одинаковая сила в обе стороны — оговорка, как в
предсказывающем кодировании). Это обучение по разнице фаз (контрастное
Хеббово / Scellier, Bengio 2017) — родственник предсказывающего кодирования,
удобный для импульсов.
Ответ мозга: слово с наибольшим числом срабатываний в свободной фазе.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.two_phase [--seed 1] [--hand] [--epochs 150]
"""
import argparse
import time

import numpy as np

from human_brain.core.Brain import Brain
from human_brain.stages import tasks
from human_brain.stages.similarity import ANIMALS, VERBS, WORDS, GROUPS, facts

N_REP, N_HID = 8, 24
W, SETTLE, QUIET = 16, 4, 6
FB = 0.5          # обратная связь = FB * прямая
RULE = "generec"  # "generec": r_i(своб) * (r_j(отв) - r_j(своб)); "chl": разница совместной активности
BIAS = True       # нейрон-смещение
HOMEOSTASIS = True  # подстройка порога NeuronChange (исходные значения мозга)
SIGNED = True     # у связи есть тормоз-близнец: вес со знаком (возбуждение - торможение)
ETA = 30.0        # мВ на единицу разницы совместной активности
W_MAX = 12.0
W_INH = 10.0      # торможение между словами


def make_brain(seed):
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    b = Brain()
    for reg, names in (("животные", ANIMALS), ("глаголы", VERBS),
                       ("образ", [f"R{k}" for k in range(N_REP)]),
                       ("скрытый", [f"H{k}" for k in range(N_HID)]), ("слова", WORDS)):
        b.add_region(reg)
        for n in names:
            b.add_neuron(n, region=reg)
    rng = np.random.default_rng(seed)
    b.fwd = []                      # (прямая, обратная или None, источник, цель)
    b.signed = {}                   # возбуждающая связь -> её тормоз-близнец
    b.w_eff = {}                    # прямая связь -> вес со знаком
    def pair(s, t, w):
        c = b.connect(s, t, weight=max(w, 0.0), myelin_thickness=1.0, plasticity=0.0)
        if SIGNED:
            ci = b.connect(s, t, weight=max(-w, 0.0), inhibitory=True, plasticity=0.0)
            b.signed[c.index] = ci.index
        return c
    def link(srcs, tgts, back):
        for s in srcs:
            for t in tgts:
                w = float(rng.uniform(0.5, 3.0))
                c = pair(s, t, w)
                r = pair(t, s, FB * w) if back else None
                b.w_eff[c.index] = w
                b.fwd.append((c.index, None if r is None else r.index, b.neurons[s].index, b.neurons[t].index))
    R = [f"R{k}" for k in range(N_REP)]
    H = [f"H{k}" for k in range(N_HID)]
    link(ANIMALS, R, back=False)
    link(R, H, back=True)
    link(VERBS, H, back=False)
    link(H, WORDS, back=True)
    if BIAS:
        # смещение: нейрон «всегда горит», связи со знаком ко всем слоям (с 0)
        b.add_neuron("BIAS", region="животные")
        link(["BIAS"], R + H + WORDS, back=False)
        for c, back, i, j in b.fwd:
            if b.names[i] == "BIAS":
                set_w(b, c, back, 0.0)
    if HOMEOSTASIS:
        # подстройка порога (есть в мозге): частый нейрон поднимает порог,
        # редкий опускает — держит средний уровень, ошибка может идти и вниз
        from human_brain.core.NeuronChange import NeuronChange
        NeuronChange(b, rate=p["nc_rate"], target=p["nc_target"],
                     threshold_spread=p["nc_threshold_spread"], seed=seed)
    for x in WORDS:
        for y in WORDS:
            if x != y:
                b.connect(x, y, weight=W_INH, inhibitory=True, plasticity=0.0)
    return b


def quiet(b):
    """Пауза между фазами: заряд держится ниже покоя, чтобы активность,
    ходящая по кругу через обратные связи, погасла (иначе ответ прошлой
    пробы переходит в следующую)."""
    rest = b.rules["Neuron"]["REST"]
    for _ in range(QUIET):
        b.potential[:b.n_neurons] = rest - 40.0
        b.step()
    b.potential[:b.n_neurons] = rest


def phase(b, animal, verb, target=None):
    """W шагов; возвращает долю шагов со срабатыванием для каждого нейрона."""
    n = b.n_neurons
    counts = np.zeros(n)
    for k in range(W):
        b.stimulate(animal, 20.0)
        b.stimulate(verb, 20.0)
        if BIAS:
            b.stimulate("BIAS", 20.0)
        if target is not None:
            b.stimulate(target, 20.0)
        before = b.spikes[:n].copy()
        b.step()
        if k >= SETTLE:
            counts += b.spikes[:n] - before
    quiet(b)
    return counts / (W - SETTLE)


def answer(b, animal, verb):
    r = phase(b, animal, verb)
    idx = [b.neurons[w].index for w in WORDS]
    v = r[idx]
    return WORDS[int(np.argmax(v))] if v.max() > 0 else None


def learn(b, animal, verb, target):
    rf = phase(b, animal, verb)
    rc = phase(b, animal, verb, target)
    lo = -W_MAX if SIGNED else 0.0
    for c, back, i, j in b.fwd:
        if RULE == "generec":
            # активность источника * местная ошибка получателя (с ответом - свободно)
            d = rf[i] * (rc[j] - rf[j])
        else:
            d = rc[i] * rc[j] - rf[i] * rf[j]
        w = b.w_eff[c] + ETA * d
        w = min(W_MAX, max(lo, w))
        set_w(b, c, back, w)


def set_w(b, c, back, w):
    """Вес со знаком: плюс — возбуждающая связь, минус — её тормоз-близнец."""
    b.w_eff[c] = w
    for k, v in ((c, w), (back, FB * w)):
        if k is None:
            continue
        b.weight[k] = max(v, 0.0)
        if k in b.signed:
            b.weight[b.signed[k]] = max(-v, 0.0)


def hand_wire(b):
    """Заведомо решаемая проверка: образ = группа, скрытый = (группа, глагол)."""
    for c, back, i, j in b.fwd:
        set_w(b, c, back, 0.0)
    names = b.names
    groups = list(GROUPS)
    reach = b.reach(0)
    for c, back, i, j in b.fwd:
        s, t = names[i], names[j]
        w = 0.0
        if s in ANIMALS and t.startswith("R"):
            g = next(k for k, gname in enumerate(groups) if s in GROUPS[gname][0])
            w = 2 * reach if int(t[1:]) == g else 0.0
        elif t.startswith("H") and int(t[1:]) < 9:
            g, v = divmod(int(t[1:]), 3)
            # заряд копится несколько шагов: при срабатывании через шаг один
            # вход даёт в покое около 3.3 * w, нужно, чтобы порог (8) брали только два
            if s.startswith("R"):
                w = 1.8 if int(s[1:]) == g else 0.0
            elif s in VERBS:
                w = 1.8 if VERBS.index(s) == v else 0.0
        elif s.startswith("H") and int(s[1:]) < 9 and t in WORDS:
            g, v = divmod(int(s[1:]), 3)
            w = 2 * reach if GROUPS[groups[g]][1][VERBS[v]] == t else 0.0
        set_w(b, c, back, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hand", action="store_true")
    a = ap.parse_args()
    train, test = facts()
    b = make_brain(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    def score():
        return (np.mean([answer(b, an, vb) == w for an, vb, w in train]),
                np.mean([answer(b, an, vb) == w for an, vb, w in test]))
    if a.hand:
        hand_wire(b)
    else:
        for ep in range(a.epochs):
            for k in rng.permutation(len(train)):
                learn(b, *train[k])
            if (ep + 1) % 25 == 0:
                tr, te = score()
                print(f"эпоха {ep + 1} ({time.time() - t0:.0f} с): обучение {tr:.2f}, не было {te:.2f}", flush=True)
    tr, te = score()
    print(f"итог: обучение {tr:.2f}, не было {te:.2f}")
    print("ответы на «не было»:", [(an, vb, answer(b, an, vb), w) for an, vb, w in test])
    # образы животных: активность слоя образа
    for an in ANIMALS:
        r = phase(b, an, VERBS[0])
        print(f"  {an:8s}", np.round(r[[b.neurons[f'R{k}'].index for k in range(N_REP)]], 2))


if __name__ == "__main__":
    main()
