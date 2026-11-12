"""Движок GGUF с угадывателем: текст тот же, что без него? скорость. python -m llm_app.check_gspec qwen3:14b Qwen3-0.6B [k]"""
import sys

import torch
from transformers import AutoTokenizer

from .engine import Engine
from .gengine import GGUFEngine
from .model import find_model
from .selftest import TEXTS


def main():
    name, dname = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    N = 64
    torch.set_grad_enabled(False)
    tk = AutoTokenizer.from_pretrained(find_model("Qwen3-8B"))
    draft = GGUFEngine(dname, maxt=1024, vram_reserve=0) if ":" in dname else Engine(dname, maxt=1024, cpu=False)
    eng = GGUFEngine(name, maxt=1024)
    for t in TEXTS:
        ids = tk.encode(t)
        s0, s1 = {}, {}
        g0 = list(eng.generate(ids, N, stats=s0))
        g1 = list(eng.generate(ids, N, draft=draft, k=k, stats=s1))
        v = lambda s: 1000 * s["gen_n"] / max(s["gen_ms"], 1)
        print(f"  без угадывателя {v(s0):.1f} слов/с, с угадывателем {v(s1):.1f} слов/с; текст тот же: {g0 == g1}", flush=True)


if __name__ == "__main__":
    main()
