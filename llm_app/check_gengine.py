"""Движок GGUF против Ollama на тех же весах: жадно, сырой текст (без шаблона), сравнение продолжений.
python -m llm_app.check_gengine qwen3:8b [слоёв_в_видеокарте]  (Ollama должна быть запущена)"""
import json
import sys
import time
import urllib.request

import torch
from transformers import AutoTokenizer

from .gengine import open_gguf
from .model import find_model
from .selftest import TEXTS


def ollama(model, text, n):
    body = json.dumps({"model": model, "prompt": text, "raw": True, "stream": False, "keep_alive": 0,
                       "options": {"temperature": 0, "top_k": 1, "num_predict": n, "num_ctx": 1024}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", body,
                                                                  {"Content-Type": "application/json"}), timeout=600).read())
    return r["response"], r["eval_count"] / (r["eval_duration"] / 1e9)


def main():
    name = sys.argv[1]
    R = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    N = 64
    tk = AutoTokenizer.from_pretrained(find_model("Qwen3-8B"))
    ref = [ollama(name, t, N) for t in TEXTS]  # сначала Ollama (потом она выгружается: keep_alive 0)
    time.sleep(3)
    torch.set_grad_enabled(False)
    eng = open_gguf(name, maxt=1024, resident=R, cpu="--no-cpu" not in sys.argv)
    for t, (ro, ospeed) in zip(TEXTS, ref):
        ids = tk.encode(t)
        st = {}
        g = list(eng.generate(ids, N, stats=st))
        mine = tk.decode(g)
        same = 0
        while same < min(len(mine), len(ro)) and mine[same] == ro[same]:
            same += 1
        if hasattr(eng, "moe_s"):
            print(f"    из них эксперты на процессоре: {eng.moe_s * 1000 / max(st['gen_n'], 1):.1f} мс/слово"); eng.moe_s = 0.0
        print(f"  наш {st['gen_ms'] / max(st['gen_n'], 1):.1f} мс/слово ({1000 * st['gen_n'] / max(st['gen_ms'], 1):.1f} слов/с), "
              f"Ollama {ospeed:.1f} слов/с; совпало с начала {same} из {len(ro)} символов\n    наш:    {mine[:90]!r}\n    Ollama: {ro[:90]!r}", flush=True)


if __name__ == "__main__":
    main()
