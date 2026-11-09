"""Лестница миров. Обученный агент идёт дальше, рядом учится новичок в тех же условиях.

Мир 1: как было (агент уже обучен там, в папке v100).
Мир 2: те же предметы и гости, но подзатыльник одинаковый: и за меньше, и за больше.
Мир 3: предметов и гостей больше нет. Жуки и многоножки это еда, рыбки в аквариуме ждут её.
       Правила как в мире 2.

    python ladder.py --seed 0
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from agent import Agent, learn_from_days
from run import exam_rate
from world import (
    NEW_BUGS, NEW_FISH, NEW_GUESTS, NEW_SITUATIONS, NEW_THINGS, OLD_BUGS, OLD_FISH, OLD_GUESTS, OLD_SITUATIONS,
    OLD_THINGS, make_day, show,
)

STEPS = 6000
CHECK_EVERY = 250
BATCH = 128
RATE = 3e-4
LIMIT = 10
GOAL = 0.9
STAGES = (
    ("мир 2: подзатыльник в обе стороны", OLD_THINGS, OLD_GUESTS, NEW_THINGS, NEW_GUESTS),
    ("мир 3: жуки и рыбки", OLD_BUGS, OLD_FISH, NEW_BUGS, NEW_FISH),
)


def live(agent, stage, seed, label):
    name, things, guests, new_things, new_guests = stage
    random.seed(seed)
    torch.manual_seed(seed)
    rate = RATE
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    rollbacks = 0

    def score():
        return exam_rate(agent, LIMIT, OLD_SITUATIONS, things, guests, count=100, seed=11)["точно"]

    curve = [(0, score())]
    best, best_state = curve[0][1], copy.deepcopy(agent.state_dict())
    print(f"{name}, {label}: сразу по прибытии {curve[0][1]:.2f}", flush=True)
    for step in range(1, STEPS + 1):
        days = [make_day(LIMIT, OLD_SITUATIONS) for _ in range(BATCH)]
        learn_from_days(agent, optimizer, [show(day, things, guests) for day in days], days, 0.005)
        if step % CHECK_EVERY == 0:
            value = score()
            curve.append((step, value))
            if value >= best:
                best, best_state = value, copy.deepcopy(agent.state_dict())
            elif value < best - 0.3:
                # Менять можно, только если не стало резко хуже.
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
                rollbacks += 1
            print(f"{name}, {label}: шаг {step}, точно {value:.2f}, откатов {rollbacks}", flush=True)
    agent.load_state_dict(best_state)
    return {
        "сразу по прибытии": curve[0][1],
        "шагов до 90%": next((s for s, v in curve if v >= GOAL), None),
        "лучшее": best,
        "откатов": rollbacks,
        "кривая": curve,
        "знакомое здесь": exam_rate(agent, LIMIT, OLD_SITUATIONS, things, guests),
        "новые виды": exam_rate(agent, LIMIT, OLD_SITUATIONS, new_things, new_guests),
        "новые ситуации": exam_rate(agent, LIMIT, NEW_SITUATIONS, things, guests),
        "новые ситуации и виды": exam_rate(agent, LIMIT, NEW_SITUATIONS, new_things, new_guests),
        "числа до 20": exam_rate(agent, 20, OLD_SITUATIONS, things, guests),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = Path(__file__).parent.parent / "v100" / "results_long" / f"memory_32__rate_0.002__seed_{args.seed}.pt"
    out = Path(__file__).parent / "results_ladder"
    out.mkdir(exist_ok=True)
    expert = Agent(32, "cpu")
    expert.load_state_dict(torch.load(saved))
    results = {}
    for stage in STAGES:
        results[stage[0]] = {
            "опытный": live(expert, stage, args.seed, "опытный"),
            "новичок": live(Agent(32, "cpu"), stage, args.seed, "новичок"),
        }
        (out / f"seed_{args.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(expert.state_dict(), out / f"expert_{args.seed}.pt")
    print("готово", flush=True)


if __name__ == "__main__":
    main()
