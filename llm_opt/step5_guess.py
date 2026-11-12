"""Замер 2: как часто дешёвые угадыватели называют те же слова, что выбрала бы модель (жадная генерация).

16 продолжений по 128 слов от начал из test (полная модель, жадно). Для каждого места j угадыватель видит
настоящий префикс; если все прошлые угадывания приняты, префикс у него такой же, как при реальной работе,
поэтому проверка по префиксу точна. Симулируется сама схема: черновик из k слов, проверка одним проходом,
принимаются совпавшие подряд + 1 слово от модели.
Угадыватели:
  full    — сама полная модель (заведомо решаемая проверка: должна принимать всё);
  lookup  — повтор из уже написанного: найти последнее прошлое вхождение последних n слов (n=3,2,1), взять следующее;
  ngram   — частые сочетания из train (3 слова -> 2 -> 1);
  lookup+ngram — сначала повтор, иначе сочетания;
  layerM  — первые M слоёв модели + её словарь (стоит M/28 прохода на слово).
"""
import json
import time

import numpy as np
import torch

from core import L, Qwen, wiki_tokens

V = 151936
t0 = time.time()
m = Qwen("mps")
test = wiki_tokens("test")
P, G, NP = 128, 128, 16

# ---------- траектории полной модели ----------
seqs = []
for j in range(NP):
    prompt = test[5000 + j * 15000: 5000 + j * 15000 + P].to("mps")
    cache = [None] * L
    lg = m.forward(prompt[None], cache=cache)
    tok = lg[0, -1].argmax()
    out = [tok.item()]
    for _ in range(G - 1):
        lg = m.forward(tok.view(1, 1), cache=cache)
        tok = lg[0, -1].argmax()
        out.append(tok.item())
    seqs.append(np.concatenate([prompt.cpu().numpy(), np.array(out)]))
print(f"траектории: {NP} x {G} слов, {time.time() - t0:.0f} с")

# ---------- предсказания по префиксу ----------
preds = {}  # имя -> массив [NP, P+G]: что угадыватель называет для позиции j (по seq[:j]); -1 = нет догадки

# полная модель и первые слои: один проход с учителем
LAY = (4, 8, 12, 16, 20, 24)
for name in ["full"] + [f"layer{M}" for M in LAY]:
    preds[name] = np.full((NP, P + G), -1, np.int64)
for j, s in enumerate(seqs):
    x = torch.tensor(s, device="mps")[None]
    logits, hid = m.forward(x, return_hidden=True)
    preds["full"][j, 1:] = logits[0, :-1].argmax(-1).cpu().numpy()
    for M in LAY:
        preds[f"layer{M}"][j, 1:] = m.logits_from_hidden(hid[M])[0, :-1].argmax(-1).cpu().numpy()

# повтор из написанного
def lookup(s, j, ns=(3, 2, 1)):
    for n in ns:
        if j < n + 1:
            continue
        key = s[j - n:j]
        for p in range(j - n - 1, -1, -1):  # последнее прошлое вхождение (не само окончание)
            if np.array_equal(s[p:p + n], key):
                return s[p + n]
    return -1


preds["lookup"] = np.array([[lookup(s, j) for j in range(P + G)] for s in seqs])

# сочетания из train
tr = wiki_tokens("train").numpy().astype(np.int64)


def best_next(ctx_ids, nxt):
    """Для каждого контекста самое частое следующее слово. Возвращает (отсортированные контексты, слова)."""
    key = ctx_ids * V + nxt
    uk, cnt = np.unique(key, return_counts=True)
    ctx, w = uk // V, uk % V
    order = np.lexsort((cnt, ctx))  # по контексту, внутри по частоте
    ctx, w = ctx[order], w[order]
    last = np.r_[ctx[1:] != ctx[:-1], True]
    return ctx[last], w[last]


tri = best_next(tr[:-2] * V + tr[1:-1], tr[2:])
bi = best_next(tr[:-1], tr[1:])
uni = np.bincount(tr, minlength=V).argmax()


def ngram(s, j):
    if j >= 2:
        c = s[j - 2] * V + s[j - 1]
        i = np.searchsorted(tri[0], c)
        if i < len(tri[0]) and tri[0][i] == c:
            return tri[1][i]
    if j >= 1:
        i = np.searchsorted(bi[0], s[j - 1])
        if i < len(bi[0]) and bi[0][i] == s[j - 1]:
            return bi[1][i]
    return uni


preds["ngram"] = np.array([[ngram(s, j) for j in range(P + G)] for s in seqs])
preds["lookup+ngram"] = np.where(preds["lookup"] >= 0, preds["lookup"], preds["ngram"])
print(f"угадыватели готовы, {time.time() - t0:.0f} с")

# ---------- симуляция схемы ----------
S = np.stack(seqs)


def simulate(name, k):
    hit = preds[name] == S  # hit[j, p]: угадано слово на позиции p
    passes, made = 0, 0
    for j in range(NP):
        p = P
        while p < P + G:
            a = 0
            while a < k and p + a < P + G and hit[j, p + a]:
                a += 1
            p += a + 1
            made += a + 1
            passes += 1
    return made / passes


res = {"first_hit": {}, "tokens_per_pass": {}}
print("\nдоля мест, где угадано следующее слово (в сгенерированной части):")
for name in preds:
    h = (preds[name][:, P:] == S[:, P:]).mean()
    res["first_hit"][name] = float(h)
    print(f"  {name:13s}: {100 * h:5.1f}%")
print("\nслов за один проход проверки (черновик из k слов):")
for name in preds:
    row = {k: simulate(name, k) for k in (1, 2, 3, 4, 7, 15)}
    res["tokens_per_pass"][name] = row
    print(f"  {name:13s}: " + "  ".join(f"k={k}: {v:.2f}" for k, v in row.items()))
json.dump(res, open("results/step5_guess.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
