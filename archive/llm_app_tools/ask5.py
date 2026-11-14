"""Набор вопросов серверу (интерфейс Ollama /api/chat), ответы и скорость — в ask5_<имя>.json.
python -m llm_app.ask5 <имя> <url> <модель> [набор: hard5 | php3]"""
import json
import sys

from ramvi.engine.tools.bench.bench import chat

QS = [
    "How many positive integers less than 1000 are divisible by 7 but not by 11? Explain your reasoning and give the final number.",
    "I have a 3-liter jug and a 5-liter jug and unlimited water. What is the minimum number of steps (fill, empty, or pour) "
    "needed to get exactly 4 liters in one jug? List the steps.",
    "Write a Python function fib_mod(n) that returns the n-th Fibonacci number modulo 1_000_000_007 (fib_mod(0)=0, fib_mod(1)=1). "
    "It must work fast for n up to 10**18. Give only the code with brief comments.",
    "Найди сумму всех двузначных чисел, у которых сумма цифр равна 9. Объясни решение и дай ответ.",
    "Two fair six-sided dice are rolled. Given that at least one of them shows a 6, what is the probability that both show 6? "
    "Explain and give the answer as a fraction.",
]


PHP3 = [
    "Write two PHP functions: int_to_roman(int $n): string and roman_to_int(string $s): int, for numbers 1..3999 "
    "(standard Roman numerals with subtractive notation, e.g. 4 = IV, 9 = IX, 40 = XL, 90 = XC, 400 = CD, 900 = CM). "
    "Give only the PHP code (starting with <?php), no explanation.",
    "Write a PHP function parse_csv_line(string $line): array that splits one CSV line into fields WITHOUT using "
    "str_getcsv or fgetcsv. Rules: fields are separated by commas; a field may be enclosed in double quotes; inside a "
    "quoted field commas are literal and a doubled quote (\"\") means one quote character; empty fields are allowed. "
    "Give only the PHP code (starting with <?php), no explanation.",
    "Write a PHP class LRUCache with: __construct(int $capacity), get(int $key): int (returns the value, or -1 if the key "
    "is missing; a successful get makes the key most recently used), put(int $key, int $value): void (inserts or updates; "
    "if the cache exceeds capacity, evicts the least recently used key). Both operations should be O(1) on average. "
    "Give only the PHP code (starting with <?php), no explanation.",
]
SETS = {"hard5": QS, "php3": PHP3}


def main():
    name, url, model = sys.argv[1:4]
    QS_ = SETS[sys.argv[4]] if len(sys.argv) > 4 else QS
    chat(url, model, "Hello!", 8)
    res = []
    for q in QS_:
        out, fin, ttft, wall = chat(url, model, q, 1024)
        n, s = fin.get("eval_count", 0), fin.get("eval_duration", 0) / 1e9
        res.append({"q": q, "out": out, "gen_n": n, "gen_s": s, "ttft_s": ttft, "wall_s": wall})
        print(f"  {n} слов за {wall:.1f} с ({n / max(s, 1e-9):.1f} слов/с, первое слово {ttft:.2f} с)", flush=True)
    json.dump(res, open(f"ask5_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ASK_DONE", flush=True)


if __name__ == "__main__":
    main()
