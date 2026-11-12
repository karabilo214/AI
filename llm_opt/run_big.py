"""Генерация большой Qwen3 своим движком: python run_big.py <модель> <режим> [начал] [слов].
Пишет big_<модель>_<режим>.json: слова, время чтения запроса и одного слова, память видеокарты."""
import json
import sys

import torch

from big_engine import BigEngine
from core import wiki_tokens

name, mode = sys.argv[1], sys.argv[2]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 64
P = 128
test = wiki_tokens("test").tolist()
prompts = [test[5000 + j * 15000: 5000 + j * 15000 + P] for j in range(NP)]
import os
R = int(os.environ.get("RESIDENT", "0"))
CPU = os.environ.get("CPU_ALT") == "1"
eng = BigEngine(name, mode, resident=R, cpu_alt=CPU)
print(f"{name} {mode}: загрузка {eng.load_s:.0f} с, слоёв в видеокарте {R}, на процессоре {len(eng.cpu_ids)}, память видеокарты {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ, в RAM закреплено {eng.pinned / 2**30:.2f} ГиБ, обычной {eng.unpinned / 2**30:.2f} "
      f"(пик {torch.cuda.max_memory_allocated() / 2**30:.2f})", flush=True)
eng.generate(prompts[0][:40], 4)  # прогрев, запись графа
out = []
for i, p in enumerate(prompts):
    toks, tp, tg = eng.generate(p, GEN)
    out.append({"prompt": p, "gen": toks, "prefill_ms": tp, "ms_per_token": tg})
    print(f"  {i}: чтение запроса {tp:.0f} мс, генерация {tg:.2f} мс/слово", flush=True)
ms = sorted(o["ms_per_token"] for o in out)[len(out) // 2]
print(f"медиана генерации {ms:.2f} мс/слово ({1000 / ms:.1f} слов/с); пик памяти {torch.cuda.max_memory_allocated() / 2**30:.2f} ГиБ")
json.dump({"name": name, "mode": mode, "out": out, "mem_gib": torch.cuda.max_memory_allocated() / 2**30},
          open(f"big_{name}_{mode}{'_R%d' % R if R else ''}{'_cpu' if CPU else ''}.json", "w"))
