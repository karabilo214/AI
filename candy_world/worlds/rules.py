"""Настоящая проверка: мир, где правильный ответ меняется, и агента об этом не предупреждают.

В каждый мир идёт копия опытного агента и новичок. Условия у них одинаковые: шаг, число шагов
и порядок чисел. Считаем только дни, где новое правило меняет ответ: там старая привычка ошибается.

    python rules.py --seed 0
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from agent import Agent, learn_from_days
from run import fetched_on
from world import (
    MAIN, NEW_BUGS, NEW_FISH, NEW_SITUATIONS, OLD_BUGS, OLD_FISH, OLD_SITUATIONS, OTHER_WORLDS, make_day, show,
)

STEPS = 12000
CHECK_EVERY = 250
BATCH = 128
RATE = 1e-3
STAGES = ((0.0, 3), (1 / 3, 6), (2 / 3, 10))
LIMIT = 10
GOAL = 0.9


def changed_days(rules, limit, situations, count, seed):
    """Дни, где по новому правилу нужно другое число упаковок, чем по старому."""
    rng = random.Random(seed)
    days = []
    while len(days) < count:
        day = make_day(limit, situations, rng)
        if day.right_fetches(rules) != day.right_fetches(MAIN):
            days.append(day)
    return days


def rates(agent, rules, days, things, guests, seed=3):
    rng = random.Random(seed)
    new = old = 0
    for day in days:
        got = fetched_on(agent, day, things, guests, rng)
        new += got == day.right_fetches(rules)
        old += got == day.right_fetches(MAIN)
    return {"по новому правилу": round(new / len(days), 3), "по старой привычке": round(old / len(days), 3)}


def live(agent, rules, seed, label):
    random.seed(seed)
    torch.manual_seed(seed)
    rate = RATE
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    check_days = changed_days(rules, LIMIT, OLD_SITUATIONS, 100, 11)
    first = rates(agent, rules, check_days, OLD_BUGS, OLD_FISH)
    curve = [(0, first["по новому правилу"])]
    best, best_state, rollbacks, stage_limit = curve[0][1], copy.deepcopy(agent.state_dict()), 0, None
    print(f"{rules.name}, {label}: по прибытии {first}", flush=True)
    for step in range(1, STEPS + 1):
        limit = [top for start, top in STAGES if step / STEPS >= start][-1]
        if limit != stage_limit:
            stage_limit = limit
        days = [make_day(limit, OLD_SITUATIONS) for _ in range(BATCH)]
        entropy = 0.01 * max(0.0, 1 - step / (STEPS * 0.8))
        learn_from_days(agent, optimizer, [show(d, OLD_BUGS, OLD_FISH) for d in days], days, entropy, rules)
        if step % CHECK_EVERY == 0:
            value = rates(agent, rules, check_days, OLD_BUGS, OLD_FISH)["по новому правилу"]
            curve.append((step, value))
            if value >= best:
                best, best_state = value, copy.deepcopy(agent.state_dict())
            elif value < best - 0.3:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
                rollbacks += 1
            print(f"{rules.name}, {label}: шаг {step}, точно {value:.2f}, откатов {rollbacks}", flush=True)
    agent.load_state_dict(best_state)
    return {
        "по прибытии": first,
        "шагов до 90%": next((s for s, v in curve if v >= GOAL), None),
        "лучшее": best,
        "откатов": rollbacks,
        "кривая": curve,
        "знакомые ситуации и виды": rates(agent, rules, changed_days(rules, LIMIT, OLD_SITUATIONS, 300, 5), OLD_BUGS, OLD_FISH),
        "новые ситуации и новые виды": rates(agent, rules, changed_days(rules, LIMIT, NEW_SITUATIONS, 300, 5), NEW_BUGS, NEW_FISH),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = Path(__file__).parent / "results_ladder" / f"expert_{args.seed}.pt"
    out = Path(__file__).parent / "results_rules"
    out.mkdir(exist_ok=True)
    results = {}
    for rules in OTHER_WORLDS:
        expert = Agent(32, "cpu")
        expert.load_state_dict(torch.load(saved))
        results[rules.name] = {
            "опытный": live(expert, rules, args.seed, "опытный"),
            "новичок": live(Agent(32, "cpu"), rules, args.seed, "новичок"),
        }
        (out / f"seed_{args.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("готово", flush=True)


if __name__ == "__main__":
    main()
