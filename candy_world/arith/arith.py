"""Один понятный проход: агент сам учится складывать, вычитать, умножать и делить.

День. Агент видит события по одному:
  карточка (N штук подряд = "каждому по N"; для сложения и вычитания N = 1, для деления карточки нет),
  конфета (у меня есть), дверь (пришла новая группа гостей), гость (в группе минимум 2 гостя),
  вопрос (сколько принести / сколько останется / сколько каждому).
Потом он отвечает шагами: "ещё одну" или "готово". Ответом считается число шагов "ещё".

Задачи:
  сложение:   пришли 5, потом ещё 7, конфет нет, каждому по 1. Сколько принести? 12
  вычитание:  у меня 5, пришли 7. Сколько не хватило? 2. Сколько останется? 0 (и наоборот)
  умножение:  пришли 8, каждому по 2. Сколько принести? 16
  деление:    у меня 17, пришли 8. Сколько каждому? 2 (1 останется)

Учится по последствиям: после праздника видно, скольким не хватило или сколько осталось,
значит видно, насколько ошибся; на этом сеть понемногу поправляется.
Порядок: сложение -> + вычитание -> + умножение -> + деление; прежние задачи остаются в обучении.
Учим на числах до 10, часть сочетаний не показываем. Экзамен: свежие дни, скрытые сочетания,
числа 11-20 и 21-50, карточка 6-9 (не видел). Рядом всегда ленивая стратегия "всегда 0".

    python arith.py --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

CARD, CANDY, DOOR, GUEST, Q_BRING, Q_LEFT, Q_EACH, ACT = range(8)
TASKS = ("сложение", "вычитание: не хватило", "вычитание: осталось", "умножение", "деление")
STAGES = (("сложение",), ("вычитание: не хватило", "вычитание: осталось"), ("умножение",), ("деление",))
STEPS_PER_STAGE, BATCH, CHECK_EVERY = 3000, 256, 250
TRAIN_TOP, CARD_TOP = 10, 5


def hidden(a, b):
    """Сочетания, которых нет в обучении (примерно каждое шестое)."""
    return (a * 7 + b * 3) % 6 == 0


def groups(guests, rng, at_least=1):
    """Разбить гостей на группы по 2 и больше."""
    sizes = [2] * max(at_least, rng.randint(1, guests // 2))
    while sum(sizes) > guests:
        sizes.pop()
    for _ in range(guests - sum(sizes)):
        sizes[rng.randrange(len(sizes))] += 1
    out = []
    for s in sizes:
        out += [DOOR] + [GUEST] * s
    return out


def make_day(task, rng, top=TRAIN_TOP, low=0, card=(2, CARD_TOP), want_hidden=False):
    """Вернуть (события, верный ответ). top/low задают диапазон чисел, want_hidden - только скрытые сочетания."""
    for _ in range(10000):
        if task == "сложение":
            g = rng.randint(max(4, low), top)
            seq, ans, key = [CARD] + groups(g, rng, 2) + [Q_BRING], g, None
        elif task.startswith("вычитание"):
            c, g = rng.randint(low, top), rng.randint(max(2, low), top)
            q = Q_BRING if task.endswith("хватило") else Q_LEFT
            seq = [CARD] + [CANDY] * c + groups(g, rng) + [q]
            ans, key = (max(g - c, 0) if q == Q_BRING else max(c - g, 0)), (c, g)
        elif task == "умножение":
            n, g = rng.randint(*card), rng.randint(max(2, low), top)
            seq, ans, key = [CARD] * n + groups(g, rng) + [Q_BRING], n * g, (n, g)
        else:
            c, g = rng.randint(low, top), rng.randint(2, TRAIN_TOP)
            seq, ans, key = [CANDY] * c + groups(g, rng) + [Q_EACH], c // g, (c, g)
        if task == "сложение" or hidden(*key) == want_hidden:
            return seq, ans
    raise RuntimeError(task)


class Agent(nn.Module):
    def __init__(self, memory=32):
        super().__init__()
        self.cell = nn.GRU(8, memory, batch_first=True)
        self.choose = nn.Linear(memory, 2)   # 1 = "ещё одну", 0 = "готово"

    def logits(self, days, steps):
        """Пройти события, потом steps раз сигнал "отвечай". Вход на шагах ответа всегда одинаков,
        поэтому решения не меняют память, и весь день считается одним проходом."""
        length = max(len(s) for s, _ in days) + steps
        x = torch.zeros(len(days), length, 8)
        starts = []
        for i, (seq, _) in enumerate(days):
            x[i, torch.arange(len(seq)), torch.tensor(seq)] = 1
            x[i, len(seq):len(seq) + steps, ACT] = 1
            starts.append(len(seq))
        out = self.choose(self.cell(x)[0])
        idx = torch.tensor(starts)[:, None] + torch.arange(steps)[None]
        return out[torch.arange(len(days))[:, None], idx]          # [дни, шаги ответа, 2]

    def answer(self, days, steps):
        with torch.no_grad():
            more = self.logits(days, steps).argmax(-1) == 1
            stop = (~more).float().argmax(-1)
            return torch.where((~more).any(-1), stop, torch.full_like(stop, steps)).tolist()


def learn(agent, opt, days):
    """После праздника видно, насколько ошибся: верный ответ = принёс + не хватило - осталось.
    Поправка: на каждом шаге до верного числа надо было сказать "ещё", на верном - "готово"."""
    steps = max(a for _, a in days) + 1
    out = agent.logits(days, steps)
    target = torch.tensor([[1 if k < a else 0 for k in range(steps)] for _, a in days])
    mask = torch.tensor([[k <= a for k in range(steps)] for _, a in days])
    loss = nn.functional.cross_entropy(out[mask], target[mask])
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()


def rate(agent, days):
    got = agent.answer(days, max(a for _, a in days) + 10)
    return round(sum(g == a for g, (_, a) in zip(got, days)) / len(days), 3)


def lazy(days):
    return round(sum(a == 0 for _, a in days) / len(days), 3)


def exam_sets(seed):
    rng = random.Random(1000 + seed)
    sets = {}
    for t in TASKS:
        sets[f"{t}: числа до 10, свежие дни"] = [make_day(t, rng) for _ in range(300)]
        if t != "сложение":
            sets[f"{t}: скрытые сочетания"] = [make_day(t, rng, want_hidden=True) for _ in range(300)]
        sets[f"{t}: числа 11-20"] = [make_day(t, rng, 20, 11) for _ in range(300)]
        sets[f"{t}: числа 21-50"] = [make_day(t, rng, 50, 21) for _ in range(300)]
    sets["умножение: карточка 6-9, гостей до 10"] = [make_day("умножение", rng, card=(6, 9)) for _ in range(300)]
    return sets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory", type=int, default=32)
    p.add_argument("--steps", type=int, default=STEPS_PER_STAGE)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    agent = Agent(a.memory)
    opt = torch.optim.Adam(agent.parameters(), lr=2e-3)
    check_rng = random.Random(77)
    check = {t: [make_day(t, check_rng) for _ in range(200)] for t in TASKS}
    print("ленивая 'всегда 0' на днях для кривой:", {t: lazy(d) for t, d in check.items()}, flush=True)
    curve, started, seen = [], time.time(), []
    for stage, new in enumerate(STAGES):
        seen += list(new)
        for step in range(1, a.steps + 1):
            old = [t for t in seen if t not in new]
            tasks = [rng.choice(new) if (not old or i % 2) else rng.choice(old) for i in range(BATCH)]
            learn(agent, opt, [make_day(t, rng) for t in tasks])
            if step % CHECK_EVERY == 0:
                row = {"шаг": stage * a.steps + step, "ступень": stage + 1}
                row.update({t: rate(agent, check[t]) for t in seen})
                curve.append(row)
                print(row, f"{time.time() - started:.0f} с", flush=True)
    results = {"seed": a.seed, "память": a.memory, "весов": sum(x.numel() for x in agent.parameters()),
               "кривая": curve, "экзамен": {}}
    print("\nЭКЗАМЕН (агент / ленивая 'всегда 0')")
    for name, days in exam_sets(a.seed).items():
        results["экзамен"][name] = {"агент": rate(agent, days), "всегда 0": lazy(days)}
        print(f"  {name:45} {results['экзамен'][name]['агент']:6.3f} / {results['экзамен'][name]['всегда 0']:.3f}")
    results["секунд"] = round(time.time() - started)
    out = Path(__file__).parent / a.out
    out.mkdir(exist_ok=True)
    (out / f"seed_{a.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(agent.state_dict(), out / f"seed_{a.seed}.pt")


if __name__ == "__main__":
    main()
