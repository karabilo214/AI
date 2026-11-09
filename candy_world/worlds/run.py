"""Обучение и экзамен из EXAM.md.

    python run.py --memory 32 128 512 --seeds 0 1 2 --device cuda
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from agent import Agent, learn_from_days
from world import MAIN, NEW_GUESTS, NEW_SITUATIONS, NEW_THINGS, OLD_GUESTS, OLD_SITUATIONS, OLD_THINGS, make_day, show

STEPS = 12000
BATCH = 128
CHECK_EVERY = 250
STAGES = ((0.0, 3), (1 / 3, 6), (2 / 3, 10))  # с какой доли обучения и до скольких гостей и предметов


def fetched_on(agent, day, things, guests, rng):
    with torch.no_grad():
        memory = agent.look([show(day, things, guests, rng)])
        return agent.prepare(memory, greedy=True)[0][0].item()


def exam_rate(agent, limit, situations, things, guests, count=300, seed=5, rules=MAIN):
    """Доля дней, когда не хватает, и сходил ровно столько, сколько надо. И средний размер ошибки."""
    rng = random.Random(seed)
    right = errors = done = 0
    while done < count:
        day = make_day(limit, situations, rng)
        if day.right_fetches(rules) == 0:
            continue
        got = fetched_on(agent, day, things, guests, rng)
        right += got == day.right_fetches(rules)
        errors += abs(got - day.right_fetches(rules))
        done += 1
    return {"точно": round(right / count, 3), "средняя ошибка в походах": round(errors / count, 2)}


def quality(agent, limit):
    return exam_rate(agent, limit, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS, count=100, seed=11)["точно"]


def train(seed, memory_size, device, rate):
    random.seed(seed)
    torch.manual_seed(seed)
    agent = Agent(memory_size, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    window = []
    best_score, best_state, best_limit = -1.0, None, None
    rollbacks = 0
    for step in range(1, STEPS + 1):
        limit = [top for start, top in STAGES if step / STEPS >= start][-1]
        if limit != best_limit:
            if best_limit is not None:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            best_limit, best_score = limit, -1.0
        days = [make_day(limit, OLD_SITUATIONS) for _ in range(BATCH)]
        shown = [show(day, OLD_THINGS, OLD_GUESTS) for day in days]
        entropy = 0.02 * max(0.0, 1 - step / (STEPS * 0.8))
        window += learn_from_days(agent, optimizer, shown, days, entropy)
        if step % CHECK_EVERY == 0:
            score = quality(agent, limit)
            n = len(window)
            print(
                f"день {step * BATCH}, до {limit}: точно {score:.2f}, лучшее {max(best_score, 0):.2f}, "
                f"мимо {sum(o['short'] + o['extra'] for o in window) / n:.2f}, походов {sum(o['fetched'] for o in window) / n:.1f}",
                flush=True,
            )
            window = []
            if score >= best_score:
                best_score, best_state = score, copy.deepcopy(agent.state_dict())
            elif score < best_score - 0.3:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
                rollbacks += 1
                print(f"  откат, номер {rollbacks}, сила поправок {rate:.5f}", flush=True)
    agent.load_state_dict(best_state)
    return agent, rollbacks


def exam(agent, limit=10):
    return {
        "1. знакомые ситуации и предметы": exam_rate(agent, limit, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS),
        "2. новые предметы и существа": exam_rate(agent, limit, OLD_SITUATIONS, NEW_THINGS, NEW_GUESTS),
        "3. новые ситуации": exam_rate(agent, limit, NEW_SITUATIONS, OLD_THINGS, OLD_GUESTS),
        "4. новые ситуации и предметы": exam_rate(agent, limit, NEW_SITUATIONS, NEW_THINGS, NEW_GUESTS),
        "5. числа до 20, больше обучения": exam_rate(agent, 20, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS),
    }


def run(seed, memory_size, device, out, rate):
    name = f"memory_{memory_size}__rate_{rate:g}__seed_{seed}"
    if (out / f"{name}.json").exists():
        print("уже есть", name, flush=True)
        return
    print(f"\n=== память {memory_size}, запуск {seed}, {device} ===", flush=True)
    started = time.perf_counter()
    agent, rollbacks = train(seed, memory_size, device, rate)
    results = {
        "seed": seed, "память": memory_size, "сила поправок": rate, "откатов": rollbacks,
        "весов": sum(p.numel() for p in agent.parameters()),
        "обученная": exam(agent), "необученная": exam(Agent(memory_size, device)),
    }
    results["секунд"] = round(time.perf_counter() - started)
    for key in results["обученная"]:
        print(f"{key}: {results['обученная'][key]}", flush=True)
    print("секунд", results["секунд"], "весов", results["весов"], "откатов", rollbacks, flush=True)
    torch.save(agent.state_dict(), out / f"{name}.pt")
    (out / f"{name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--memory", type=int, nargs="+", default=[32])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", default="results_size")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    STEPS = args.steps
    out = Path(__file__).parent / args.out
    out.mkdir(exist_ok=True)
    for memory_size in args.memory:
        for seed in args.seeds:
            run(seed, memory_size, args.device, out, args.rate)
