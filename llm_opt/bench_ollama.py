"""Сравнение с Ollama (llama.cpp) на ASUS: те же 16 начал по 128 слов, жадно 128 слов, текст подаётся как есть (raw).
Время — по счётчикам Ollama (чтение запроса + генерация), отдельно чистая генерация.
Текст сравнивается с эталоном (наш движок fp32 = HF fp32): сколько слов эталона совпали подряд."""
import gc
import json
import statistics
import sys
import urllib.request

import torch

from core import tokenizer
from gpu_engine import GpuEngine
from spec_gpu import GEN, NP, P, plain, prompts

tk = tokenizer()
# эталон
eng = GpuEngine(mode="fp32")
ref = [plain(eng, p) for p in prompts]
del eng; gc.collect(); torch.cuda.empty_cache()
ref_text = [[tk.decode(r[:i]) for i in range(GEN + 1)] for r in ref]


def ollama(model, text):
    body = json.dumps({"model": model, "prompt": text, "raw": True, "stream": False, "keep_alive": "5m",
                       "options": {"temperature": 0, "top_k": 1, "seed": 0, "num_predict": GEN, "num_ctx": 1024}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


res = {}
for model in sys.argv[1:]:
    ollama(model, tk.decode(prompts[0]))  # загрузка модели и прогрев
    tot, dec, same, first, pe = [], [], 0, [], []
    for p, rt in zip(prompts, ref_text):
        r = ollama(model, tk.decode(p))
        pe.append(r["prompt_eval_count"])
        tot.append((r["prompt_eval_duration"] + r["eval_duration"]) / 1e6 / GEN)
        dec.append(r["eval_duration"] / 1e6 / r["eval_count"])
        out = r["response"]
        n = max(i for i in range(GEN + 1) if out.startswith(rt[i]))  # сколько слов эталона совпали подряд
        first.append(n)
        same += n == GEN
    res[model] = {"ms_per_token_total": statistics.mean(tot), "ms_per_token_decode": statistics.mean(dec),
                  "same": same, "first_div": first, "prompt_tokens": pe}
    print(f"  Ollama {model:18s}: {statistics.mean(tot):6.2f} мс/слово (с чтением запроса), чистая генерация "
          f"{statistics.mean(dec):6.2f} мс/слово, текст как у эталона {same}/{NP}, совпало слов подряд (медиана) "
          f"{sorted(first)[NP // 2]}; длина запроса у Ollama {min(pe)}-{max(pe)} (у нас {P})", flush=True)
json.dump(res, open("bench_ollama.json", "w"), indent=1)
