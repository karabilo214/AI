"""Проверка гибрида: текст не зависит от истории переключений «шаг на процессоре / пачка в видеокарте», обе копии
кэша k, v гибридных слоёв совпадают по всем принятым позициям.
python -m ramvi.engine.tools check switch -- qwen3:14b qwen3:0.6b"""
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from ramvi.engine.tools.bench.bench_wide import WIDE
from ramvi_engine.core.gengine import GGUFEngine
from ramvi_engine.formats.model import find_model
from ramvi.engine.tools.checks.selftest import TEXTS


def kv_diff(eng, n):
    d = 0.0
    for l in eng.hyb_ids:
        cl = eng.W[l]
        for C, G in ((cl.K, eng.Kc[l]), (cl.V, eng.Vc[l])):
            d = max(d, float(np.abs(C[:, :n] - G[:, :n].float().cpu().numpy()).max()))
    return d


def main():
    name, dname = sys.argv[1], sys.argv[2]
    N = next((int(arg.split("=", 1)[1]) for arg in sys.argv if arg.startswith("--tokens=")), 64)
    if not 1 <= N <= 1024:
        raise ValueError("неверная длина проверки")
    torch.set_grad_enabled(False)
    tk = AutoTokenizer.from_pretrained(find_model("Qwen3-8B"))
    draft = GGUFEngine(dname, maxt=1024, vram_reserve=0)
    eng = GGUFEngine(name, maxt=1024, cpu="hybrid")
    eng.autofit()
    prompts = [tk.encode(t) for t in TEXTS] + [
        tk.encode(tk.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False), add_special_tokens=False) for _, q in (WIDE[8], WIDE[13], WIDE[29])]
    rnd = random.Random(0)
    allok, worst = True, 0.0
    for i, ids in enumerate(prompts):
        ref = list(eng.generate(ids, N))  # только одиночные шаги
        modes = {
            "каждое слово": lambda j: "step" if j % 2 else "spec",
            "раз в 1-3 слова": (lambda pat: (lambda j: pat[j % len(pat)]))(
                sum([[m] * rnd.randint(1, 3) for m in ["step", "spec"] * 40], [])),
            "постоянное k=7": None,
            "адаптивно": None,
        }
        res = []
        for mname, f in modes.items():
            st = {}
            if mname == "постоянное k=7":
                g = list(eng.generate(ids, N, draft=draft, k=7, stats=st))
            else:
                g = list(eng.generate(ids, N, draft=draft, force=f, stats=st))
            dkv = kv_diff(eng, len(ids) + len(g) - 1)
            worst = max(worst, dkv)
            ok = g == ref and dkv == 0.0
            allok &= ok
            res.append(f"{mname}: {'тот же' if ok else 'ОТЛИЧАЕТСЯ'} (переключений {st.get('switches', 0)}, кэш {dkv:.1e})")
        print(f"  запрос {i + 1}: " + "; ".join(res), flush=True)
    print(f"ИТОГ: текст не зависит от переключений: {allok}; наибольшее расхождение копий кэша {worst:.1e}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
