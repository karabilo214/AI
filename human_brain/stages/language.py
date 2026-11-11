"""Проверка B: предсказание следующего слова (как у LLM) текущим мозгом.

Мир: фразы «животное [наречие] глагол еда». У каждой пары (животное, глагол)
своя еда (8 разных слов). В обучении наречия («быстро», «медленно») бывают
только у кота и пса. Экзамен — без обучения, предсказать еду:
    простые     — «птица ест ___» (было в обучении);
    наречие-old — «кот быстро ест ___» (было);
    наречие-new — «птица быстро ест ___» (НЕ было: нужно помнить животное
                  через вставку, которой с ним не встречалось).
Таблица частот (n-граммы до 3 слов с откатом) на наречие-new обречена:
«быстро ест» видела только с котом и псом.

Мозг: по нейрону на слово, слова подаются через GAP шагов, между фразами
пауза. Без наград: только общий рост (связи «X, потом Y через gap»,
нейроны-сочетания с окном совпадения PAIR_WINDOW) и обычная пластичность.
Ответ мозга: еда, у которой на месте следующего слова наибольший заряд
(или которая сработала сама).
Сравнение: ленивая (самая частая еда), n-граммы, крошечный трансформер.

Запуск из корня проекта:
    .venv/bin/python -m human_brain.stages.language [--seed 1] [--reps 20] [--hand]
"""
import argparse
import copy
import random
import time
from collections import Counter, defaultdict

import numpy as np

from human_brain.stages import tasks
from human_brain.core.Brain import Brain

ANIMALS = ["кот", "пёс", "птица", "рыба"]
VERBS = ["ест", "пьёт"]
ADVERBS = ["быстро", "медленно"]
FOOD = {("кот", "ест"): "мясо", ("кот", "пьёт"): "молоко",
        ("пёс", "ест"): "кость", ("пёс", "пьёт"): "воду",
        ("птица", "ест"): "зерно", ("птица", "пьёт"): "росу",
        ("рыба", "ест"): "червя", ("рыба", "пьёт"): "тину"}
OBJECTS = sorted(set(FOOD.values()))
VOCAB = ANIMALS + VERBS + ADVERBS + OBJECTS
WITH_ADVERB = ["кот", "пёс"]       # наречия в обучении только у них
GAP = 2                            # шагов между словами
REST = 14                          # пауза между фразами
PAIR_WINDOW = 4                    # окно совпадения сочетаний (шагов)


def corpus(reps, seed):
    phrases = [[a, v, FOOD[(a, v)]] for a in ANIMALS for v in VERBS]
    phrases += [[a, d, v, FOOD[(a, v)]] for a in WITH_ADVERB for d in ADVERBS for v in VERBS]
    out = phrases * reps
    random.Random(seed).shuffle(out)
    return out


def exams():
    return {
        "простые": [([a, v], FOOD[(a, v)]) for a in ANIMALS for v in VERBS],
        "наречие-old": [([a, d, v], FOOD[(a, v)]) for a in WITH_ADVERB for d in ADVERBS for v in VERBS],
        "наречие-new": [([a, d, v], FOOD[(a, v)]) for a in ANIMALS if a not in WITH_ADVERB
                        for d in ADVERBS for v in VERBS],
    }


# ---------- мозг ----------

def make_brain(seed):
    p = dict(tasks.PARAMS_DEFAULT, fam_on=False)
    tasks.apply(p)
    b = Brain()
    b.add_region("слова")
    for w in VOCAB:
        b.add_neuron(w, region="слова")
    tasks.attach(b, p, seed, curiosity=False, neurons=False)
    g = b.growth
    g.max_neurons = 400
    g.pair_window = PAIR_WINDOW
    g.pair_regions = ["слова"]      # сочетания только из слов (не из сочетаний)
    return b


def say(b, words):
    for w in words:
        b.stimulate(w, 20.0)
        for _ in range(GAP):
            b.step()


def train(b, phrases):
    for ph in phrases:
        say(b, ph)
        for _ in range(REST):
            b.step()


def predict(b, context):
    """Еда с наибольшим зарядом на месте следующего слова (на копии)."""
    t = copy.deepcopy(b)
    # рост на экзамене не идёт, но сам Growth нужен: окно совпадения сочетаний
    # считается в Brain.step через него
    t.growth.connection_threshold = t.growth.neuron_threshold = 10 ** 9
    for w in context[:-1]:
        t.stimulate(w, 20.0)
        for _ in range(GAP):
            t.step()
    t.stimulate(context[-1], 20.0)
    best, best_v = None, -1e9
    for _ in range(GAP + 1):
        fired = t.step()
        for o in OBJECTS:
            i = t.neurons[o].index
            v = 1e6 if o in fired else t.potential[i]
            if v > best_v:
                best, best_v = o, v
    rest = t.rules["Neuron"]["REST"]
    return best if best_v > rest + 1e-6 else None


def hand_wire(b):
    """Заведомо решаемая проверка: сочетания «животное+глагол» и связи к еде руками."""
    g = b.growth
    for (a, v), o in FOOD.items():
        g._grow_neuron((a, v), 99)
        name = g.pair_neurons[(a, v)]
        for c in b.connections:
            if c.source == name and c.target == o:
                break
        else:
            b.connect(name, o, weight=12.0, myelin_thickness=1.0)
        for c in b.connections:
            if c.source == name and c.target == o:
                b.weight[c.index] = 12.0


# ---------- сравнения ----------

def ngram_predict(phrases, context):
    counts = defaultdict(Counter)
    for ph in phrases:
        for i in range(1, len(ph)):
            for n in (1, 2, 3):
                if i - n >= 0:
                    counts[tuple(ph[i - n:i])][ph[i]] += 1
    for n in (3, 2, 1):
        key = tuple(context[-n:])
        c = Counter({w: k for w, k in counts.get(key, {}).items() if w in OBJECTS})
        if c:
            return c.most_common(1)[0][0]
    return None


def skip_predict(phrases, context):
    """Таблица пар «любое прежнее слово + последнее слово -> следующее»
    (промежуток не важен) — честная ленивая стратегия для этой задачи."""
    cnt = defaultdict(Counter)
    for ph in phrases:
        for i in range(1, len(ph)):
            for j in range(i - 1):
                cnt[(ph[j], ph[i - 1])][ph[i]] += 1
    vote = Counter()
    for j in range(len(context) - 1):
        vote.update({w: k for w, k in cnt.get((context[j], context[-1]), {}).items() if w in OBJECTS})
    return vote.most_common(1)[0][0] if vote else None


def transformer_scores(phrases, ex, seed, epochs=300):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    idx = {w: i + 1 for i, w in enumerate(VOCAB)}   # 0 = начало
    L = 5
    d = 32

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(VOCAB) + 1, d)
            self.pos = nn.Embedding(L, d)
            self.layer = nn.TransformerEncoderLayer(d, 2, 64, dropout=0.0, batch_first=True)
            self.out = nn.Linear(d, len(VOCAB) + 1)

        def forward(self, x):
            h = self.emb(x) + self.pos(torch.arange(x.shape[1]))[None]
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
            return self.out(self.layer(h, src_mask=mask, is_causal=True))

    uniq = [list(p) for p in {tuple(p) for p in phrases}]
    seqs = []
    for ph in phrases:
        s = [0] + [idx[w] for w in ph]
        seqs.append(s + [0] * (L + 1 - len(s)))
    X = torch.tensor(seqs)
    m = Tiny()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    n_params = sum(p.numel() for p in m.parameters())
    t0 = time.time()
    for _ in range(epochs):
        logits = m(X[:, :-1])
        tgt = X[:, 1:]
        mask = tgt != 0
        loss = nn.functional.cross_entropy(logits[mask], tgt[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_s = time.time() - t0
    obj = torch.tensor([idx[o] for o in OBJECTS])
    res = {}
    with torch.no_grad():
        for name, items in ex.items():
            ok = 0
            for ctx, want in items:
                x = torch.tensor([[0] + [idx[w] for w in ctx]])
                sc = m(x)[0, -1, obj]
                ok += OBJECTS[int(sc.argmax())] == want
            res[name] = ok / len(items)
    tokens = int((X[:, 1:] != 0).sum()) * epochs
    return res, {"параметров": n_params, "обучение, с": round(train_s, 1),
                 "≈ операций обучения": f"{6 * n_params * tokens:.1e}", "уникальных фраз": len(uniq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--hand", action="store_true", help="заведомо решаемая проверка")
    ap.add_argument("--control", action="store_true", help="еда случайна (связи учить нечему)")
    a = ap.parse_args()
    phrases = corpus(a.reps, a.seed)
    if a.control:
        r = random.Random(a.seed + 1)
        phrases = [ph[:-1] + [r.choice(OBJECTS)] for ph in phrases]
    ex = exams()

    b = make_brain(a.seed)
    t0 = time.time()
    if a.hand:
        hand_wire(b)
    else:
        train(b, phrases)
    brain_s = time.time() - t0
    print(f"мозг: шагов {b.time}, нейронов {b.n_neurons}, связей {sum(1 for c in b.connections)}, "
          f"обучение {brain_s:.1f} с")
    rows = {}
    rows["мозг"] = {n: sum(predict(b, c) == w for c, w in it) / len(it) for n, it in ex.items()}
    lazy = Counter(ph[-1] for ph in phrases).most_common(1)[0][0]
    rows["ленивая"] = {n: sum(lazy == w for _, w in it) / len(it) for n, it in ex.items()}
    rows["n-граммы"] = {n: sum(ngram_predict(phrases, c) == w for c, w in it) / len(it) for n, it in ex.items()}
    rows["таблица пар"] = {n: sum(skip_predict(phrases, c) == w for c, w in it) / len(it) for n, it in ex.items()}
    tr, info = transformer_scores(phrases, ex, a.seed)
    rows["трансформер"] = tr
    print(f"{'':12s}" + "".join(f"{n:>14s}" for n in ex))
    for k, r in rows.items():
        print(f"{k:12s}" + "".join(f"{r[n]:14.2f}" for n in ex))
    print("трансформер:", info)
    if not a.hand:
        print("мозг на «птица быстро ест»:", predict(b, ["птица", "быстро", "ест"]))
        grown = [n for n in b.growth.grown_neurons]
        print("сочетаний выросло:", len(grown), "пример:", grown[:12])


if __name__ == "__main__":
    main()
