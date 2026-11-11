"""Проверка C: точная ошибка местными правилами (предсказывающее кодирование).

Задача «сова» (обобщение по сходству — то, чего таблица не умеет):
    9 животных в 3 группах; у группы общая еда, питьё и умение:
        птицы  (воробей, голубь, сова)  — ест зерно,  пьёт росу,  умеет летать
        рыбы   (карп, щука, сом)        — ест червя,  пьёт тину,  умеет плавать
        звери  (кот, пёс, волк)         — ест мясо,   пьёт воду,  умеет бегать
    В обучении про сову, сома и волка — по одному факту (сова ест зерно,
    сом умеет плавать, волк пьёт воду). Экзамен: остальные 6 фактов о них —
    этих пар «животное + глагол» не было ни разу.

Кто решает (ответ — одно из 9 слов):
    таблица      — пара (животное, глагол) -> слово; пары не было — откат к
                   самому частому слову после глагола;
    мозг         — наш импульсный мозг, как в language.py (рост сочетаний);
    обр. распр.  — сеть того же строения, обратное распространение (образец);
    пред. код.   — та же сеть, предсказывающее кодирование (Whittington и
                   Bogacz, 2017): у каждого слоя «значения» и «ошибки»
                   (значение минус предсказание сверху вниз по сети);
                   значения сначала успокаиваются (несколько шагов, только
                   местные сигналы), потом каждая связь меняется на
                   (ошибка своего получателя) * (активность своего источника).
                   Никакого прохода ошибки назад.
Строение сети (как у Румельхарта): животное -> слой образа (4 нейрона) ->
вместе с глаголом -> скрытый слой (12) -> слово.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.similarity [--seed 1]
"""
import argparse
import copy
from collections import Counter

import numpy as np

GROUPS = {"птицы": (["воробей", "голубь", "сова"], {"ест": "зерно", "пьёт": "росу", "умеет": "летать"}),
          "рыбы": (["карп", "щука", "сом"], {"ест": "червя", "пьёт": "тину", "умеет": "плавать"}),
          "звери": (["кот", "пёс", "волк"], {"ест": "мясо", "пьёт": "воду", "умеет": "бегать"})}
ANIMALS = [a for g in GROUPS.values() for a in g[0]]
VERBS = ["ест", "пьёт", "умеет"]
WORDS = sorted({w for g in GROUPS.values() for w in g[1].values()})
KNOWN_ONLY = {"сова": "ест", "сом": "умеет", "волк": "пьёт"}   # единственный факт о них


def facts():
    out = []
    for animals, rel in GROUPS.values():
        for a in animals:
            for v in VERBS:
                out.append((a, v, rel[v]))
    train = [f for f in out if f[0] not in KNOWN_ONLY or KNOWN_ONLY[f[0]] == f[1]]
    test = [f for f in out if f not in train]
    return train, test


# ---------- таблица ----------

def table_scores(train, sets):
    pair = {(a, v): w for a, v, w in train}
    by_verb = {v: Counter(w for a, vv, w in train if vv == v).most_common(1)[0][0] for v in VERBS}
    return {n: np.mean([pair.get((a, v), by_verb[v]) == w for a, v, w in s]) for n, s in sets.items()}


# ---------- сеть: общий вид ----------

def onehot(items, x):
    v = np.zeros(len(items))
    v[items.index(x)] = 1.0
    return v


class Net:
    """животное -> образ (R) ; [f(образ), глагол] -> скрытый (H) ; f(H) -> слово."""
    def __init__(self, seed, n_rep=4, n_hid=12):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.5, (n_rep, len(ANIMALS)))
        self.W2 = r.normal(0, 0.5, (n_hid, n_rep + len(VERBS)))
        self.b2 = np.zeros(n_hid)
        self.W3 = r.normal(0, 0.5, (len(WORDS), n_hid))
        self.b3 = np.zeros(len(WORDS))

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def df(x):
        return 1.0 - np.tanh(x) ** 2

    def forward(self, a, v):
        x1 = self.W1 @ a
        u = np.concatenate([self.f(x1), v])
        x2 = self.W2 @ u + self.b2
        x3 = self.W3 @ self.f(x2) + self.b3
        return x1, u, x2, x3

    def answer(self, an, vb):
        return WORDS[int(np.argmax(self.forward(onehot(ANIMALS, an), onehot(VERBS, vb))[3]))]


def scores(net, sets):
    return {n: np.mean([net.answer(a, v) == w for a, v, w in s]) for n, s in sets.items()}


def train_backprop(net, train, epochs, lr, rng):
    for _ in range(epochs):
        for k in rng.permutation(len(train)):
            an, vb, w = train[k]
            a, v, t = onehot(ANIMALS, an), onehot(VERBS, vb), onehot(WORDS, w)
            x1, u, x2, x3 = net.forward(a, v)
            e3 = x3 - t
            e2 = (net.W3.T @ e3) * net.df(x2)
            e1 = (net.W2[:, :x1.size].T @ e2) * net.df(x1)
            net.W3 -= lr * np.outer(e3, net.f(x2)); net.b3 -= lr * e3
            net.W2 -= lr * np.outer(e2, u);         net.b2 -= lr * e2
            net.W1 -= lr * np.outer(e1, a)


def train_pc(net, train, epochs, lr, rng, steps=40, gamma=0.2):
    """Предсказывающее кодирование. Значения x1, x2 (слова — зажаты верным
    ответом) успокаиваются по местным сигналам: dx = -ошибка_своя +
    f'(x) * (связи вверх)^T ошибка_выше. Потом связи: dW = ошибка_получателя *
    активность_источника. Ошибка слоя = значение - предсказание от слоя ниже."""
    nr = net.W1.shape[0]
    for _ in range(epochs):
        for k in rng.permutation(len(train)):
            an, vb, w = train[k]
            a, v, t = onehot(ANIMALS, an), onehot(VERBS, vb), onehot(WORDS, w)
            x1, u, x2, _ = net.forward(a, v)          # начало: прямое предсказание
            x1, x2 = x1.copy(), x2.copy()
            x3 = t                                    # выход зажат ответом
            for _ in range(steps):
                u = np.concatenate([net.f(x1), v])
                e1 = x1 - net.W1 @ a
                e2 = x2 - (net.W2 @ u + net.b2)
                e3 = x3 - (net.W3 @ net.f(x2) + net.b3)
                x1 += gamma * (-e1 + net.df(x1) * (net.W2[:, :nr].T @ e2))
                x2 += gamma * (-e2 + net.df(x2) * (net.W3.T @ e3))
            u = np.concatenate([net.f(x1), v])
            e1 = x1 - net.W1 @ a
            e2 = x2 - (net.W2 @ u + net.b2)
            e3 = x3 - (net.W3 @ net.f(x2) + net.b3)
            net.W3 += lr * np.outer(e3, net.f(x2)); net.b3 += lr * e3
            net.W2 += lr * np.outer(e2, u);         net.b2 += lr * e2
            net.W1 += lr * np.outer(e1, a)


# ---------- наш мозг ----------

def brain_scores(train, sets, seed, reps=20):
    from human_brain.stages import language as L
    import random
    L.ANIMALS, L.VERBS, L.ADVERBS, L.OBJECTS = ANIMALS, VERBS, [], WORDS
    L.VOCAB = ANIMALS + VERBS + WORDS
    b = L.make_brain(seed)
    phrases = [[a, v, w] for a, v, w in train] * reps
    random.Random(seed).shuffle(phrases)
    L.train(b, phrases)
    return {n: np.mean([L.predict(b, [a, v]) == w for a, v, w in s]) for n, s in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    train, test = facts()
    sets = {"обучение": train, "сова и др. (не было)": test}
    rows = {"таблица": table_scores(train, sets), "мозг": brain_scores(train, sets, a.seed)}
    net = Net(a.seed)
    train_backprop(net, train, a.epochs, 0.05, np.random.default_rng(a.seed))
    rows["обр. распр."] = scores(net, sets)
    bp = net
    net = Net(a.seed)
    train_pc(net, train, a.epochs, 0.05, np.random.default_rng(a.seed))
    rows["пред. код."] = scores(net, sets)
    print(f"{'':14s}" + "".join(f"{n:>24s}" for n in sets))
    for k, r in rows.items():
        print(f"{k:14s}" + "".join(f"{r[n]:24.2f}" for n in sets))
    print("ответы пред. код. на «не было»:", [(an, vb, net.answer(an, vb), w) for an, vb, w in test])
    rep = {an: np.round(np.tanh(net.W1 @ onehot(ANIMALS, an)), 2) for an in ANIMALS}
    print("образы животных (пред. код.):")
    for an in ANIMALS:
        print(f"  {an:8s}", rep[an])


if __name__ == "__main__":
    main()
