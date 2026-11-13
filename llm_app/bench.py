"""Сравнение серверов в реальных условиях через интерфейс Ollama (/api/chat):
python -m llm_app.bench <имя> <url> <модель> [задач]
  имя — подпись в результатах; url — http://127.0.0.1:11435 (наш) или :11434 (Ollama); модель — имя у сервера.
1) скорость: 5 разных просьб, 256 слов, жадно: чтение запроса, генерация (по счётчикам сервера), время до
   первого куска и всё время (по часам клиента);
2) качество: первые N задач GSM8K (test), жадно, без «размышлений»: доля верных ответов.
Пишет bench_<имя>.json (ответы целиком — для сравнения текстов между участниками)."""
import json
import re
import sys
import time
import urllib.request

ASKS = [
    "Explain how a refrigerator works in simple terms.",
    "Write a Python function that checks whether a string is a palindrome, with a short explanation.",
    "Write a short story about a lighthouse keeper who finds a mysterious message in a bottle.",
    "What are the main causes of the French Revolution? Give a structured answer.",
    "List ten practical tips for learning a new language as an adult.",
]
GSM_HINT = "\nSolve the problem step by step. At the end, write the final answer as a number after '####'."


_RUN_CACHE = {}

def run_metadata(url, model, n):
    key = (url, model)
    if key not in _RUN_CACHE:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                server = json.load(response).get("run_metadata", {})
        except (OSError, ValueError):
            server = {}
        _RUN_CACHE[key] = server
    return {"version": 1, "model": model, "server": _RUN_CACHE[key],
            "parameters": {"temperature": 0, "top_k": 1, "seed": 0, "num_ctx": 2048, "think": False, "num_predict": n}}


def chat(url, model, text, n, stream=True):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "stream": stream, "think": False,
                       "keep_alive": "30m", "options": {"temperature": 0, "top_k": 1, "seed": 0, "num_predict": n,
                                                        "num_ctx": 2048}}).encode()
    req = urllib.request.Request(url + "/api/chat", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, out, fin = None, "", {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("error"):
                raise RuntimeError(str(o["error"]))
            piece = o.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            out += piece
            if o.get("done"):
                fin = o
    if not fin:
        raise RuntimeError("сервер оборвал ответ без финального события")
    wall = time.perf_counter() - t0
    fin["_run"] = run_metadata(url, model, n)
    return out, fin, ttft or 0.0, wall


GSM_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def gsm8k(n):
    import os
    path = os.path.join(os.path.expanduser("~"), ".cache", "llm_app", "gsm8k_test.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(GSM_URL, path)
    t = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in t]


def last_number(s):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"(-?[\d,]*\.?\d+)", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    name, url, model = sys.argv[1], sys.argv[2], sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    res = {"name": name, "model": model, "speed": [], "gsm": []}
    chat(url, model, "Hello!", 8)  # загрузка модели, прогрев
    for a in ASKS:
        out, fin, ttft, wall = chat(url, model, a, 256)
        e = {"ask": a, "out": out, "ttft_s": ttft, "wall_s": wall, "prompt_n": fin.get("prompt_eval_count"),
             "prompt_s": fin.get("prompt_eval_duration", 0) / 1e9, "gen_n": fin.get("eval_count"),
             "gen_s": fin.get("eval_duration", 0) / 1e9, "run": fin["_run"]}
        res["speed"].append(e)
        print(f"  {e['gen_n']} слов: генерация {e['gen_n'] / max(e['gen_s'], 1e-9):.1f} слов/с, первое слово через "
              f"{ttft:.2f} с, всего {wall:.1f} с", flush=True)
    ok = 0
    t0 = time.perf_counter()
    for i, (q, ans) in enumerate(gsm8k(N)):
        out, fin, ttft, wall = chat(url, model, q + GSM_HINT, 512)
        got = last_number(out)
        good = got is not None and abs(float(got) - float(ans)) < 1e-6
        ok += good
        res["gsm"].append({"q": q, "ans": ans, "out": out, "got": got, "ok": good, "gen_n": fin.get("eval_count"),
                           "wall_s": wall})
        print(f"  задача {i + 1}: {'верно' if good else 'неверно'} ({got} / {ans}), {wall:.1f} с, итого {ok}/{i + 1}", flush=True)
    res["gsm_acc"] = ok / max(N, 1)
    res["gsm_wall_s"] = time.perf_counter() - t0
    sp = res["speed"]
    res["gen_tps"] = sum(e["gen_n"] for e in sp) / sum(e["gen_s"] for e in sp)
    res["ttft_s"] = sorted(e["ttft_s"] for e in sp)[len(sp) // 2]
    print(f"{name}: генерация {res['gen_tps']:.1f} слов/с, первое слово (медиана) {res['ttft_s']:.2f} с, GSM8K {ok}/{N}, "
          f"время задач {res['gsm_wall_s']:.0f} с")
    json.dump(res, open(f"bench_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
