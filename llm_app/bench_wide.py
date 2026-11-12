"""Широкий набор запросов (30, английский и русский) серверу Ollama-совместимому: скорость по каждому.
python -m llm_app.bench_wide <имя> <url> <модель>  -> wide_<имя>.json
python -m llm_app.bench_wide --compare wide_a.json wide_b.json  -> сравнение по запросам."""
import json
import statistics
import sys

from .bench import chat

PARA = ("The Industrial Revolution began in Britain in the late 18th century and spread to Europe and North America. "
        "It transformed economies that had been based on agriculture and handicrafts into economies based on large-scale "
        "industry, mechanized manufacturing, and the factory system. New machines, new power sources, and new ways of "
        "organizing work made existing industries more productive and efficient.")
WIDE = [
    ("объяснение", "Explain how the internet works to a 12-year-old."),
    ("объяснение", "What is the difference between a virus and a bacterium?"),
    ("объяснение", "Объясни простыми словами, что такое инфляция и почему она возникает."),
    ("факты", "Who was Marie Curie and what is she known for?"),
    ("факты", "Расскажи кратко историю Санкт-Петербурга."),
    ("совет", "Give me practical advice on how to prepare for a job interview."),
    ("совет", "Как правильно начать бегать, если раньше не занимался спортом?"),
    ("творчество", "Write a short poem about autumn in the city."),
    ("творчество", "Напиши короткую сказку про кота, который хотел стать капитаном корабля."),
    ("творчество", "Write the opening paragraph of a mystery novel set in Venice."),
    ("письмо", "Write a polite email to a landlord asking to fix a broken heater."),
    ("письмо", "Напиши деловое письмо с просьбой перенести встречу на следующую неделю."),
    ("код", "Write a Python function that merges two sorted lists into one sorted list, with tests."),
    ("код", "Write a JavaScript function that debounces another function, and explain how it works."),
    ("код", "Write a SQL query to find the top 5 customers by total order amount from tables customers and orders."),
    ("код", "Write a regular expression that validates an email address and explain each part."),
    ("код", "Напиши на Go функцию, которая читает файл построчно и считает количество слов."),
    ("код", "Write a C function that reverses a string in place."),
    ("математика", "A store sells apples at 3 for $2. How much do 18 apples cost? Explain step by step."),
    ("математика", "Solve the equation 2x^2 - 8x + 6 = 0 and show the steps."),
    ("математика", "Поезд проехал 240 км за 3 часа. С какой скоростью он ехал и сколько проедет за 5 часов?"),
    ("рассуждение", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain."),
    ("перевод", "Translate into Russian: 'The early bird catches the worm, but the second mouse gets the cheese.'"),
    ("перевод", "Translate into English: 'Тише едешь — дальше будешь. Не откладывай на завтра то, что можно сделать сегодня.'"),
    ("пересказ", "Summarize the following text in two sentences: " + PARA),
    ("пересказ", "Перескажи этот текст по-русски в трёх предложениях: " + PARA),
    ("список", "List 15 countries in Africa with their capitals."),
    ("список", "Перечисли 10 полезных привычек для здоровья с кратким пояснением."),
    ("json", "Return a JSON array of 5 books with fields title, author, year. Output only JSON."),
    ("диалог", "Write a short dialogue between a customer and a barista in a coffee shop."),
]


def run(name, url, model):
    chat(url, model, "Hello!", 8)
    res = []
    for cat, q in WIDE:
        out, fin, ttft, wall = chat(url, model, q, 200)
        n, s = fin.get("eval_count", 0), fin.get("eval_duration", 0) / 1e9
        res.append({"cat": cat, "q": q, "out": out, "n": n, "gen_s": s, "tps": n / max(s, 1e-9), "ttft": ttft, "wall": wall})
        print(f"  {cat:12s} {n:4d} слов, {n / max(s, 1e-9):5.1f} слов/с, первое слово {ttft:.2f} с", flush=True)
    json.dump(res, open(f"wide_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WIDE_DONE", flush=True)


def compare(fa, fb):
    a = json.load(open(fa, encoding="utf-8"))
    b = json.load(open(fb, encoding="utf-8"))
    r = [x["tps"] / y["tps"] for x, y in zip(a, b)]
    print(f"{'запрос':12s} {'A слов/с':>9s} {'B слов/с':>9s} {'A/B':>6s}")
    for x, y, q in zip(a, b, r):
        print(f"{x['cat']:12s} {x['tps']:9.1f} {y['tps']:9.1f} {q:6.2f}")
    sa, sb = sum(x["n"] for x in a), sum(y["n"] for y in b)
    ta, tb = sum(x["gen_s"] for x in a), sum(y["gen_s"] for y in b)
    print(f"итого: A {sa / ta:.1f} слов/с, B {sb / tb:.1f}; A/B по запросам: медиана {statistics.median(r):.2f}, "
          f"мин {min(r):.2f}, макс {max(r):.2f}; A не медленнее B в {sum(q >= 0.98 for q in r)}/{len(r)}; "
          f"первое слово медиана A {statistics.median(x['ttft'] for x in a):.2f} с, B {statistics.median(y['ttft'] for y in b):.2f} с")
    cats = sorted(set(x["cat"] for x in a))
    print("по видам: " + "; ".join(f"{c} {statistics.median(q for x, q in zip(a, r) if x['cat'] == c):.2f}" for c in cats))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(*sys.argv[1:4])
