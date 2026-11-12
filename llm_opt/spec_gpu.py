"""Угадывание вперёд на видеокарте: черновик из k слов (повтор из написанного, иначе частые сочетания train),
проверка k+1 слов одним проходом (граф), принимаются совпавшие подряд + 1 слово модели.
Сравнение с обычной жадной генерацией того же движка: текст должен совпасть точно; время — реальное, с угадывателем.
16 начал из test по 128 слов, генерация 128 слов."""
import json
import statistics
import sys
import time

import numpy as np
import torch

from core import wiki_tokens
from gpu_engine import GpuEngine

V = 151936
P, GEN, NP = 128, 128, 16

# ---------- угадыватель ----------
tr = wiki_tokens("train").numpy().astype(np.int64)


def best_next(ctx, nxt):
    key = ctx * V + nxt
    uk, cnt = np.unique(key, return_counts=True)
    c, w = uk // V, uk % V
    o = np.lexsort((cnt, c))
    c, w = c[o], w[o]
    last = np.r_[c[1:] != c[:-1], True]
    return dict(zip(c[last].tolist(), w[last].tolist()))


TRI = best_next(tr[:-2] * V + tr[1:-1], tr[2:])
BI = best_next(tr[:-1], tr[1:])
UNI = int(np.bincount(tr, minlength=V).argmax())


class Guesser:
    """Повтор: для n=3,2,1 помним последнее место, где встречались последние n слов; иначе сочетания из train."""

    def __init__(self, seq):
        self.seq = []
        self.last = [dict(), dict(), dict()]  # n-1 -> {кортеж n слов: позиция сразу после них}
        for t in seq:
            self.push(t)

    def push(self, t):
        self.seq.append(t)
        j = len(self.seq)
        for n in (1, 2, 3):
            if j - 1 >= n:  # n слов, кончающихся перед последним словом, -> следующее за ними = последнее слово
                self.last[n - 1][tuple(self.seq[j - 1 - n:j - 1])] = j - 1

    def guess_one(self, seq):
        for n in (3, 2, 1):
            if len(seq) >= n:
                p = self.last[n - 1].get(tuple(seq[-n:]))
                if p is not None:
                    return seq[p] if p < len(seq) else None
        if len(seq) >= 2:
            g = TRI.get(seq[-2] * V + seq[-1])
            if g is not None:
                return g
        return BI.get(seq[-1], UNI)

    def draft(self, k):
        s = list(self.seq)
        out = []
        for _ in range(k):
            g = self.guess_one(s)
            if g is None:
                break
            out.append(g)
            s.append(g)
        return out


# ---------- генерация ----------
test = wiki_tokens("test").tolist()
prompts = [test[5000 + j * 15000: 5000 + j * 15000 + P] for j in range(NP)]


def plain(eng, prompt):
    out = eng.prefill(prompt)
    tok = int(out[-1]); gen = [tok]
    for k in range(GEN - 1):
        tok = int(eng.run([tok], P + k)[0]); gen.append(tok)
    return gen


def spec(eng, prompt, k):
    out = eng.prefill(prompt)
    gen = [int(out[-1])]
    gs = Guesser(prompt + gen)
    passes = 0
    while len(gen) < GEN:
        d = gs.draft(k)
        d = d[:max(0, GEN - len(gen) - 1)]
        nreal = len(d)
        T = len(d) + 1
        while T not in eng.graphs:  # подогнать к записанным графам (1, 2, 4, 8, 16)
            d.append(0); T += 1
        pos = P + len(gen) - 1
        res = eng.run([gen[-1]] + d, pos).tolist()
        passes += 1
        n = 0
        while n < nreal and d[n] == res[n] and len(gen) + n < GEN:
            n += 1
        for t in d[:n] + [res[n]]:
            if len(gen) < GEN:
                gen.append(t); gs.push(t)
    return gen, passes


def timed(f):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = f()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0
