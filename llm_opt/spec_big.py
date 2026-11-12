"""Угадывание вперёд на большой Qwen3: python spec_big.py <модель> <режим> <k> [начал] [слов].
Чтение запроса пачками по 16; черновик k слов (повтор из написанного + сочетания train, как spec_gpu), проверка
k+1 слов одним проходом. Пишет big_<модель>_<режим>_spec<k>.json (формат как run_big — для verify_big.py)."""
import json
import sys
import time

import torch

from big_engine import BigEngine
from core import wiki_tokens
from spec_gpu import Guesser

name, mode, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
DRAFT = sys.argv[6] if len(sys.argv) > 6 else None  # маленькая модель-угадыватель, например Qwen3-0.6B


class ModelGuesser:
    """Черновик — жадная генерация маленькой модели (целиком в видеокарте, сжатые веса). Её кэш догоняет
    принятые слова одной пачкой, затем k шагов по одному слову."""

    def __init__(self, eng, seq):
        self.eng, self.seq = eng, list(seq)
        eng.prefill(self.seq[:-1])
        self.n = len(self.seq) - 1  # сколько слов уже прошло через маленькую модель

    def push(self, t):
        self.seq.append(t)

    def draft(self, k):
        if k <= 0:
            return []
        pend = self.seq[self.n:]
        g = self.eng.verify(pend, self.n)[-1]
        self.n = len(self.seq)
        out = [g]
        for i in range(k - 1):
            g = self.eng.verify([g], self.n + i)[0]
            out.append(g)
        return out

NP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
GEN = int(sys.argv[5]) if len(sys.argv) > 5 else 64
P = 128
test = wiki_tokens("test").tolist()
prompts = [test[5000 + j * 15000: 5000 + j * 15000 + P] for j in range(NP)]
import os
R = int(os.environ.get("RESIDENT", "0"))
CPU = os.environ.get("CPU_ALT") == "1"
eng = BigEngine(name, mode, resident=R, cpu_alt=CPU)
small = None
if DRAFT == "q4":  # 4-битная копия той же модели целиком в видеокарте, словарь общий
    small = BigEngine(name, "q4", share=eng)
    print(f"  угадыватель q4: память видеокарты всего {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ", flush=True)
elif DRAFT:
    small = BigEngine(DRAFT, "c12")
if small:
    t0 = time.perf_counter()
    for T in range(1, 17):
        small.verify(prompts[0][:T], 0)
print(f"{name} {mode} k={k}: слоёв в видеокарте {R}, на процессоре {len(eng.cpu_ids)}, память видеокарты {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ, в RAM закреплено {eng.pinned / 2**30:.2f} ГиБ, обычной {eng.unpinned / 2**30:.2f}", flush=True)
eng.prefill(prompts[0][:32])  # прогрев: все размеры пачки (компиляция ядер)
for T in range(1, 17):
    eng.verify(prompts[0][:T], 0)
out = []
for i, p in enumerate(prompts):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen = [eng.prefill(p)]
    torch.cuda.synchronize(); t1 = time.perf_counter()
    gs = ModelGuesser(small, p + gen) if small else Guesser(p + gen)
    passes = 0
    while len(gen) < GEN:
        d = gs.draft(k)[:max(0, GEN - len(gen) - 1)]
        res = eng.verify([gen[-1]] + d, P + len(gen) - 1)
        passes += 1
        n = 0
        while n < len(d) and d[n] == res[n]:
            n += 1
        for t in d[:n] + [res[n]]:
            if len(gen) < GEN:
                gen.append(t); gs.push(t)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    ms = (t2 - t1) * 1000 / (GEN - 1)
    out.append({"prompt": p, "gen": gen, "prefill_ms": (t1 - t0) * 1000, "ms_per_token": ms, "passes": passes})
    print(f"  {i}: чтение запроса {(t1 - t0) * 1000:.0f} мс, генерация {ms:.2f} мс/слово, слов за проход "
          f"{(GEN - 1) / passes:.2f}", flush=True)
ms = sorted(o["ms_per_token"] for o in out)[len(out) // 2]
print(f"медиана генерации {ms:.2f} мс/слово ({1000 / ms:.1f} слов/с); пик памяти {torch.cuda.max_memory_allocated() / 2**30:.2f} ГиБ")
json.dump({"name": name, "mode": mode, "out": out, "mem_gib": torch.cuda.max_memory_allocated() / 2**30},
          open(f"big_{name}_{mode}{'_R%d' % R if R else ''}{'_cpu' if CPU else ''}_spec{k}{'_' + DRAFT if DRAFT else ''}.json", "w"))
