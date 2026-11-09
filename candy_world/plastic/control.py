"""Заведомо решаемая задача: скрытый бит на весь день.

Первая попытка это угадывание. Итог говорит, что было верно. Со второй попытки ошибок быть не должно.
Ожидаемо: "reset" около 50% на всех попытках, ему нечем помнить. "keep" около 100% со второй попытки.
"plastic" обязан дать около 100% со второй, иначе новый нейрон не работает.
Внешнее обучение знает верный ответ, чтобы настроить медленные веса. Агент в работе его не видит.

    python control.py
"""
import json
import random

import torch

from net import Net
from world import ACT, CARD, result_token, token

ATTEMPTS = 6
BATCH = 64
STEPS = 2500


def play(net, rng, greedy=False):
    net.start_day(BATCH)
    bits = torch.tensor([rng.randint(0, 1) for _ in range(BATCH)])
    total, by_attempt = 0.0, []
    for _ in range(ATTEMPTS):
        net.start_attempt()
        net.step(torch.tensor([token(CARD, need=1)] * BATCH))
        state = net.step(torch.tensor([token(ACT)] * BATCH))
        log_policy = net.answer(state).log_softmax(-1)
        policy = log_policy.exp()
        # Правильный ответ знает только внешнее обучение. Сам агент видит лишь итог попытки.
        total = total + torch.nn.functional.nll_loss(log_policy, bits)
        picked = policy.argmax(-1) if greedy else torch.multinomial(policy, 1).squeeze(-1)
        by_attempt.append((picked == bits).float().mean().item())
        net.step(torch.tensor([result_token(a, b) for a, b in zip(picked.tolist(), bits.tolist())]))
    return total / ATTEMPTS, by_attempt


def exam(net, days=50):
    rng = random.Random(77)
    with torch.no_grad():
        rows = [play(net, rng, greedy=True)[1] for _ in range(days)]
    return [round(sum(r[i] for r in rows) / days, 2) for i in range(ATTEMPTS)]


def run(condition, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    net = Net(condition)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random(seed)
    for _ in range(STEPS):
        loss, _ = play(net, rng)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    row = {"условие": condition, "seed": seed, "экзамен": exam(net)}
    if condition == "plastic":
        net.plasticity_on = False
        row["быстрая часть выключена"] = exam(net)
    return row


if __name__ == "__main__":
    results = []
    for condition in ("reset", "keep", "plastic"):
        for seed in range(3):
            row = run(condition, seed)
            results.append(row)
            print(row, flush=True)
    with open("results/control.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=1)
