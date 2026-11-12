"""Проверка текста честным эталоном (torch fp32, веса слой за слоем): python verify_big.py big_<...>.json
Для каждого начала: эталон считает самое вероятное слово в каждой позиции запрос+текст за один проход;
текст совпадает с жадной генерацией эталона, если все выбранные слова равны его выбору."""
import json
import sys

from big_engine import Reference

d = json.load(open(sys.argv[1]))
ref = Reference(d["name"])
same, firsts = 0, []
for o in d["out"]:
    P, g = len(o["prompt"]), o["gen"]
    arg, gap = ref.argmax(o["prompt"] + g[:-1])
    pred = arg[P - 1:]
    first = next((i for i in range(len(g)) if pred[i] != g[i]), len(g))
    firsts.append(first)
    same += first == len(g)
    if first < len(g):
        print(f"  расхождение на слове {first}: у нас {g[first]}, у эталона {pred[first]}, отрыв эталона {gap[P - 1 + first]:.4f}")
print(f"{d['name']} {d['mode']}: текст как у эталона {same}/{len(d['out'])}, первые расхождения {firsts}")
