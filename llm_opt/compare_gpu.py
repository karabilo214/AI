"""Сравнение на видеокарте ASUS: наш движок (режимы, с угадыванием и без) против HF transformers.
16 начал из test по 128 слов, жадная генерация 128 слов; время целиком (с чтением запроса и угадывателем).
Эталон текста — наш движок в fp32 (проверен против обычного torch: 64/64)."""
import gc
import json
import sys

import torch
from transformers import AutoModelForCausalLM

from core import SNAP
from gpu_engine import GpuEngine
from spec_gpu import GEN, NP, P, plain, prompts, spec, timed

N = NP * GEN
res = {}


def compare(name, outs, t):
    same = sum(o == r for o, r in zip(outs, ref))
    first = [next((i for i in range(GEN) if o[i] != r[i]), GEN) for o, r in zip(outs, ref)]
    res[name] = {"ms_per_token": 1000 * t / N, "same": same, "first_div": first}
    print(f"  {name:28s}: {1000 * t / N:6.2f} мс/слово, текст как у эталона {same:2d}/{NP}, "
          f"первое расхождение (медиана) {sorted(first)[NP // 2]}", flush=True)


ref = None
MODES = [a for a in sys.argv[1:] if a != "hf"] or ["fp32", "w16", "w16s2", "bf16"]
for mode in (["fp32"] if MODES[0] != "fp32" else []) + MODES:
    if mode.startswith("fused"):
        import importlib
        if mode == "fused2_11":
            eng = importlib.import_module("fused2").FusedEngine(wbits=11)
        else:
            eng = importlib.import_module("fused_decode" if mode == "fused" else mode).FusedEngine()
    else:
        eng = GpuEngine(mode=mode)
    for p in prompts[:2]:
        plain(eng, p); spec(eng, p, 7)
    outs, t = timed(lambda: [plain(eng, p) for p in prompts])
    if ref is None:
        ref = outs
    compare(f"наш {mode}", outs, t)
    if hasattr(eng, "generate"):
        for p in prompts[:2]: eng.generate(p, GEN)
        o3, t3 = timed(lambda: [eng.generate(p, GEN) for p in prompts])
        compare(f"наш {mode} без ожидания", o3, t3)
    if mode == "fp32" and "fp32" not in MODES:
        del eng; gc.collect(); torch.cuda.empty_cache(); continue
    for k in ((1, 3, 7, 15) if mode.startswith("fused") else (3, 7, 15)):
        o2, t2 = timed(lambda: [spec(eng, p, k)[0] for p in prompts])
        compare(f"наш {mode} + угадывание k={k}", o2, t2)
    del eng; gc.collect(); torch.cuda.empty_cache()

import sys
for name, dt in ((("fp32", torch.float32), ("bf16", torch.bfloat16)) if "hf" in sys.argv else ()):
    hf = AutoModelForCausalLM.from_pretrained(SNAP, dtype=dt).cuda().eval()
    def run():
        outs = []
        for p in prompts:
            x = torch.tensor([p], device="cuda")
            g = hf.generate(x, max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False, temperature=None, top_p=None, top_k=None)
            outs.append(g[0, P:].tolist())
        return outs
    with torch.no_grad():
        run()  # прогрев
        outs, t = timed(run)
    compare(f"HF transformers {name}", outs, t)
    del hf; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open("compare_gpu_" + "_".join(sys.argv[1:] or ["all"]) + ".json", "w"), indent=1)
