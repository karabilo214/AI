"""Живут ли знания агента в виде устойчивых структур активности? Только анализ, без обучения.

1. Держится ли счёт, когда ничего не происходит (нулевой сигнал, его агент в обучении не видел).
2. Если толкнуть память шумом, вернётся ли она на свой путь.
3. Одно ли и то же движение памяти делает "ещё один гость" при 5 и при 50 гостях.
4. Есть ли точки покоя (долины), куда память скатывается и остаётся.

    python analyze_dynamics.py
"""
import random

import torch

from agent import Agent
from world import CARD, GUEST, LOOKS, OLD_GUESTS, OLD_THINGS, THING, TOKEN_SIZE, token

rng = random.Random(1)
cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()


def load(seed):
    agent = Agent(32, "cpu")
    agent.load_state_dict(torch.load(f"results_long/memory_32__rate_0.002__seed_{seed}.pt"))
    return agent.eval()


def item(kind):
    names = OLD_GUESTS if kind == GUEST else OLD_THINGS
    return torch.tensor([token(kind, LOOKS[rng.choice(names)])], dtype=torch.float)


@torch.no_grad()
def feed(agent, memory, items):
    for x in items:
        memory = agent.memory_cell(x, memory)
    return memory


def start(agent, n):
    card = torch.tensor([token(CARD, need=1, pack=1)], dtype=torch.float)
    return feed(agent, torch.zeros(1, 32), [card] + [item(GUEST) for _ in range(n)])


def reader(agent):
    """Линейный читатель счёта: учится по памяти после 0..60 гостей. Сам агент не меняется."""
    counts = [n for n in range(61) for _ in range(4)]
    X = torch.cat([start(agent, n) for n in counts])
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.linalg.solve(X1.T @ X1 + 1e-3 * torch.eye(33), X1.T @ torch.tensor(counts, dtype=torch.float))
    return lambda m: (torch.cat([m, torch.ones(len(m), 1)], 1) @ w)


for seed in range(3):
    agent = load(seed)
    read = reader(agent)
    silence = torch.zeros(1, TOKEN_SIZE)
    print(f"\n=== агент {seed}")

    row = {}
    for n in (5, 20, 40):
        m, trace = start(agent, n), []
        for t in range(31):
            if t in (0, 5, 10, 30):
                trace.append(round(read(m).item(), 1))
            m = feed(agent, m, [silence])
        row[n] = trace
    print("1. счёт в тишине, шаги 0/5/10/30:", row)

    for sigma in (0.1, 0.3):
        out = []
        for n in (5, 20):
            clean = start(agent, n)
            torch.manual_seed(n)
            noisy = clean + sigma * torch.randn_like(clean)
            first = (noisy - clean).norm().item()
            guests = [item(GUEST) for _ in range(10)]
            ratios = []
            for k in (1, 3, 10):
                ratios.append(round((feed(agent, noisy, guests[:k]) - feed(agent, clean, guests[:k])).norm().item() / first, 2))
            error = (read(feed(agent, noisy, guests)) - read(feed(agent, clean, guests))).item()
            out.append({"гостей": n, "отклонение после 1/3/10 гостей (1 = как было)": ratios, "ошибка счёта": round(error, 1)})
        print(f"2. толчок {sigma}:", out)

    moves = {}
    for n in (2, 5, 10, 20, 40, 60):
        m = start(agent, n)
        moves[n] = (feed(agent, m, [item(GUEST)]) - m)[0]
    print("3. похоже ли 'ещё гость' на то же при 5 гостях:", {n: round(cos(moves[5], d), 2) for n, d in moves.items()})
    print("   размер этого движения:", {n: round(d.norm().item(), 3) for n, d in moves.items()})
    m = start(agent, 10)
    print("   'ещё предмет' против 'ещё гость' при 10 (−1 = точно обратное):", round(cos((feed(agent, m, [item(THING)]) - m)[0], moves[10]), 2))

    ends = [feed(agent, start(agent, n), [silence] * 200) for n in range(0, 61, 4)]
    groups = []
    for e in ends:
        if not any((e - g).norm() < 0.05 for g in groups):
            groups.append(e)
    print("4. точек покоя после долгой тишины, начиная из 16 разных счетов:", len(groups),
          "| какой счёт в них:", sorted(round(read(g).item(), 1) for g in groups))
