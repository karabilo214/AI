"""Сводка сравнения: python -m llm_app.compare_bench bench_a.json bench_b.json ...
Первый файл — эталон текста (наш точный режим = жадная fp32-модель): для остальных — сколько ответов на задачи
совпали с ним целиком и сколько символов совпадает с начала (медиана)."""
import json
import sys


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    rs = [json.load(open(f, encoding="utf-8")) for f in sys.argv[1:]]
    ref = rs[0]
    print(f"{'участник':28s} {'генерация':>10s} {'1-е слово':>9s} {'GSM8K':>7s} {'время задач':>12s} {'текст = точный':>15s} {'общее начало':>13s}")
    for r in rs:
        same = [a["out"] == b["out"] for a, b in zip(ref["gsm"], r["gsm"])]
        pre = sorted(prefix(a["out"], b["out"]) / max(len(a["out"]), 1) for a, b in zip(ref["gsm"], r["gsm"]))
        ok = sum(g["ok"] for g in r["gsm"])
        print(f"{r['name']:28s} {r['gen_tps']:7.1f} с/с {r['ttft_s']:8.2f}с {ok:3d}/{len(r['gsm']):<3d} {r['gsm_wall_s']:10.0f} с "
              f"{sum(same):9d}/{len(same):<4d} {100 * pre[len(pre) // 2]:11.0f}%")


if __name__ == "__main__":
    main()
