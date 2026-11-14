"""Самопроверка: python -m ramvi.engine.tools check selftest -- --model Qwen3-4B [--resident R] [--draft Qwen3-0.6B] [--no-cpu]
Жадная генерация на нескольких запросах сверяется с эталоном fp32 (reference.py); повторный запрос с тем же
началом (кэш начала); с угадывателем текст должен совпасть с текстом без него."""
import argparse
import time
import warnings

warnings.filterwarnings("ignore", message=".*non-writable.*")

import torch
from transformers import AutoTokenizer

from ramvi_engine.core.engine import Engine
from ramvi_engine.formats.model import find_model
from ramvi_engine.core.reference import Reference

TEXTS = [
    "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg",
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. The process takes place in",
    "In mathematics, a prime number is a natural number greater than one that has no positive divisors other than one and itself. "
    "The first few prime numbers are 2, 3, 5, 7, 11, and 13. Euclid proved that",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resident", type=int, default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    if a.n < 1:
        ap.error("n должно быть положительным")
    torch.set_grad_enabled(False)
    tk = AutoTokenizer.from_pretrained(find_model(a.model))
    draft = Engine(a.draft, maxt=1024, cpu=False) if a.draft else None
    eng = Engine(a.model, maxt=1024, cpu=not a.no_cpu, resident=a.resident)
    outs = []
    extra_ok = True
    for t in TEXTS:
        ids = tk.encode(t)
        st = {}
        g = list(eng.generate(ids, a.n, stats=st))
        outs.append((ids, g))
        print(f"  запрос {len(ids)} слов: чтение {st['prompt_ms']:.0f} мс, генерация {st['gen_ms'] / max(st['decode_n'], 1):.1f} мс/слово"
              f" | {tk.decode(g)[:70]!r}", flush=True)
    # то же начало + продолжение: кэш начала
    previous = outs[2][1]
    split = min(20, len(previous) - 1)
    expected = previous[split:split + 16]
    ids2 = outs[2][0] + previous[:split]
    st = {}
    g2 = list(eng.generate(ids2, len(expected), stats=st))
    print(f"  с кэшем начала: пересчитано {st['prompt_n']} слов из {len(ids2)}, совпадение {g2 == expected}")
    extra_ok &= g2 == expected
    if draft:
        for ids, g in outs:
            st = {}
            gd = list(eng.generate(ids, a.n, draft=draft, stats=st))
            extra_ok &= gd == g
            print(f"  с угадывателем: тот же текст {gd == g}, генерация {st['gen_ms'] / max(st['decode_n'], 1):.1f} мс/слово")
    eng.close()
    if draft is not None:
        draft.close()
    del eng, draft
    torch.cuda.empty_cache()
    ref = Reference(a.model)
    same = 0
    for ids, g in outs:
        r = ref.argmax(ids + g[:-1])[len(ids) - 1:]
        first = next((i for i in range(len(g)) if r[i] != g[i]), len(g))
        same += first == len(g)
        if first < len(g):
            print(f"  расхождение с эталоном на слове {first}")
    print(f"текст как у эталона fp32: {same}/{len(outs)}")
    if same != len(outs) or not extra_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
