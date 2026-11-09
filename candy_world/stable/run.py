"""Контроль и мир с паузами и помехами. Потом экзамен и анализ устойчивости памяти.

    python run.py --condition stable --seed 0
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from agent import Agent, learn_from_days
from world import (
    GUEST, LOOKS, NEW_GUESTS, NEW_SITUATIONS, NEW_THINGS, OLD_GUESTS, OLD_SITUATIONS, OLD_THINGS, PAUSE, THING,
    TOKEN_SIZE, CARD, NEED_TICK, PACK_TICK, Day, make_day, show, token,
)

STEPS = 36000
BATCH = 128
CHECK_EVERY = 250
STAGES = ((0.0, 3), (1 / 3, 6), (2 / 3, 10))
SETTINGS = {"control": {"pauses": 0.0, "noise": 0.0}, "stable": {"pauses": 0.5, "noise": 0.1}}


def fetched_on(agent, day, things, guests, rng, pauses=False):
    with torch.no_grad():
        memory = agent.look([show(day, things, guests, rng, pauses)])
        return agent.prepare(memory, greedy=True)[0][0].item()


def exam_rate(agent, limit, situations, things, guests, count=300, seed=5, pauses=False):
    rng = random.Random(seed)
    right = done = 0
    while done < count:
        day = make_day(limit, situations, rng)
        if day.right_fetches() == 0:
            continue
        right += fetched_on(agent, day, things, guests, rng, pauses) == day.right_fetches()
        done += 1
    return round(right / count, 3)


def train(condition, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    settings = SETTINGS[condition]
    agent = Agent(32, "cpu", noise=settings["noise"])
    rate = 2e-3
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    best, best_state, stage = -1.0, None, None
    for step in range(1, STEPS + 1):
        limit = [top for start, top in STAGES if step / STEPS >= start][-1]
        if limit != stage:
            if stage is not None:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            stage, best = limit, -1.0
        days = [make_day(limit, OLD_SITUATIONS) for _ in range(BATCH)]
        shown = [show(d, OLD_THINGS, OLD_GUESTS, pauses=random.random() < settings["pauses"]) for d in days]
        learn_from_days(agent, optimizer, shown, days, 0.02 * max(0.0, 1 - step / (STEPS * 0.8)))
        if step % CHECK_EVERY == 0:
            agent.eval()
            score = exam_rate(agent, limit, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS, count=100, seed=11)
            agent.train()
            if score >= best:
                best, best_state = score, copy.deepcopy(agent.state_dict())
            elif score < best - 0.3:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            if step % 2000 == 0:
                print(f"{condition} {seed} шаг {step}, до {limit}: точно {score:.2f}", flush=True)
    agent.load_state_dict(best_state)
    return agent.eval()


def analyse(agent):
    """Те же четыре проверки устойчивости, что и у прошлого агента. Тишина здесь это пауза."""
    rng = random.Random(1)
    item = lambda kind: torch.tensor([token(kind, LOOKS[rng.choice(OLD_GUESTS if kind == GUEST else OLD_THINGS)])], dtype=torch.float)
    pause = torch.tensor([token(PAUSE)], dtype=torch.float)

    @torch.no_grad()
    def feed(m, items):
        for x in items:
            m = agent.memory_cell(x, m)
        return m

    def start(n):
        card = [torch.tensor([token(CARD)], dtype=torch.float), torch.tensor([token(NEED_TICK)], dtype=torch.float),
                torch.tensor([token(PACK_TICK)], dtype=torch.float)]
        return feed(torch.zeros(1, 32), card + [item(GUEST) for _ in range(n)])

    counts = [n for n in range(41) for _ in range(4)]
    X = torch.cat([start(n) for n in counts])
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.linalg.solve(X1.T @ X1 + 1e-3 * torch.eye(33), X1.T @ torch.tensor(counts, dtype=torch.float))
    read = lambda m: (torch.cat([m, torch.ones(len(m), 1)], 1) @ w).item()
    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    result = {}
    result["счёт в паузе, шаги 0/10/30"] = {n: [round(read(feed(start(n), [pause] * t)), 1) for t in (0, 10, 30)] for n in (5, 20)}
    moves = {}
    for n in (5, 10, 20, 30, 40):
        m = start(n)
        moves[n] = (feed(m, [item(GUEST)]) - m)[0]
    result["'ещё гость' похож на то же при 5"] = {n: round(cos(moves[5], d), 2) for n, d in moves.items()}
    result["размер шага 'ещё гость'"] = {n: round(d.norm().item(), 3) for n, d in moves.items()}
    m = start(10)
    result["'ещё предмет' против 'ещё гость' (−1 обратное)"] = round(cos((feed(m, [item(THING)]) - m)[0], moves[10]), 2)
    ends = [feed(start(n), [pause] * 200) for n in range(0, 41, 4)]
    groups = []
    for e in ends:
        if not any((e - g).norm() < 0.05 for g in groups):
            groups.append(e)
    result["точек покоя из 11 разных счетов"] = len(groups)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=list(SETTINGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent = train(args.condition, args.seed)
    results = {"условие": args.condition, "seed": args.seed, "экзамен": {
        "знакомое, до 10": exam_rate(agent, 10, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS),
        "новые предметы, до 10": exam_rate(agent, 10, OLD_SITUATIONS, NEW_THINGS, NEW_GUESTS),
        "новое значение 5 (не учился), до 10": exam_rate(agent, 10, NEW_SITUATIONS, OLD_THINGS, OLD_GUESTS),
        "числа до 20": exam_rate(agent, 20, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS),
        "числа до 40": exam_rate(agent, 40, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS, count=150),
        "знакомое, но с паузами": exam_rate(agent, 10, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS, pauses=True),
    }, "анализ": analyse(agent)}
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.condition}__seed_{args.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(agent.state_dict(), out / f"{args.condition}__seed_{args.seed}.pt")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
