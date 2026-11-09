"""Механизмы A и C. Сезон: 5 дней в одном скрытом мире. Миры в обучении перемешаны.

A:       предсказывает последствия, без оперативной памяти правил.
A_C:     предсказывает последствия и держит правила в отдельной маленькой памяти.
A_carry: предсказывает последствия, а правила держит та же основная память (контроль на запутанность).

    python run.py --condition A_C --seed 0
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from agent import Agent, live_day
from world import (
    GUEST, MAIN, NEW_GUESTS, NEW_SITUATIONS, NEW_THINGS, NEW_WORLDS, OLD_GUESTS, OLD_SITUATIONS, OLD_THINGS, THING,
    TRAIN_WORLDS, Day, make_day, sequence, show,
)

STEPS = 12000
BATCH = 64
DAYS = 5
CHECK_EVERY = 500
STAGES = ((0.0, 3), (1 / 3, 6), (2 / 3, 10))
SETTINGS = {"A": dict(use_context=False, carry_memory=False), "A_C": dict(use_context=True, carry_memory=False),
            "A_carry": dict(use_context=False, carry_memory=True)}


def season(agent, worlds, limit, situations, things, guests, rng, greedy=False, entropy=0.0, learn=True):
    """Один сезон у пачки агентов. Мир у каждого свой и весь сезон один и тот же."""
    count = len(worlds)
    agent.new_season(count)
    total, by_day = 0.0, []
    for _ in range(DAYS):
        days = [make_day(limit, situations, rng) for _ in range(count)]
        pairs = [show(d, things, guests, rng, consequences=True, rules=w) for d, w in zip(days, worlds)]
        loss, outcomes = live_day(agent, [p[0] for p in pairs], days, worlds, [p[1] for p in pairs],
                                  1.0 if learn else 0.0, entropy, greedy)
        total = total + loss
        need = [o for o, d, w in zip(outcomes, days, worlds) if d.right_fetches(w) > 0]
        by_day.append(sum(o["short"] + o["extra"] == 0 for o in need) / max(len(need), 1))
    return total / DAYS, by_day


@torch.no_grad()
def exam(agent, worlds, limit=10, situations=OLD_SITUATIONS, things=OLD_THINGS, guests=OLD_GUESTS, seasons=12, seed=5):
    """Точность по дням сезона, без внешнего обучения. Сеть меняется только внутри сезона."""
    rng = random.Random(seed)
    rows = [season(agent, list(worlds) * (64 // len(worlds)), limit, situations, things, guests, rng,
                   greedy=True, learn=False)[1] for _ in range(seasons)]
    return [round(sum(r[i] for r in rows) / seasons, 3) for i in range(DAYS)]


def train(condition, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    agent = Agent(32, **SETTINGS[condition])
    rate = 2e-3
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    rng = random.Random(seed + 1)
    best, best_state, stage = -1.0, None, None
    for step in range(1, STEPS + 1):
        limit = [top for start, top in STAGES if step / STEPS >= start][-1]
        if limit != stage:
            if stage is not None:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            stage, best = limit, -1.0
        worlds = [rng.choice(TRAIN_WORLDS) for _ in range(BATCH)]
        loss, _ = season(agent, worlds, limit, OLD_SITUATIONS, OLD_THINGS, OLD_GUESTS, rng,
                         entropy=0.02 * max(0.0, 1 - step / (STEPS * 0.8)))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        if step % CHECK_EVERY == 0:
            score = sum(exam(agent, TRAIN_WORLDS, limit, seasons=3, seed=11)) / DAYS
            if score >= best:
                best, best_state = score, copy.deepcopy(agent.state_dict())
            elif score < best - 0.3:
                agent.load_state_dict(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            print(f"{condition} {seed} шаг {step}, до {limit}: точно {score:.2f}", flush=True)
    agent.load_state_dict(best_state)
    return agent


def analyse(agent):
    """Проверки устойчивости в основном мире, с пустой оперативной памятью правил."""
    rng = random.Random(1)
    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    day = Day(0, 0, 0, 0, 1, 1)

    @torch.no_grad()
    def memory_after(events):
        agent.new_season(1)
        return agent.look([sequence(day, events, True, MAIN)[0]])

    guest = lambda: (GUEST, rng.choice(OLD_GUESTS))
    moves = {}
    for n in (5, 10, 20, 30, 40):
        events = [guest() for _ in range(n)]
        moves[n] = (memory_after(events + [guest()]) - memory_after(events))[0]
    events = [guest() for _ in range(10)]
    thing = (memory_after(events + [(THING, rng.choice(OLD_THINGS))]) - memory_after(events))[0]
    return {"'ещё гость' похож на то же при 5": {n: round(cos(moves[5], d), 2) for n, d in moves.items()},
            "размер шага 'ещё гость'": {n: round(d.norm().item(), 3) for n, d in moves.items()},
            "'ещё предмет' против 'ещё гость' (−1 обратное)": round(cos(thing, moves[10]), 2)}


def main():
    global STEPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=list(SETTINGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    STEPS = args.steps
    torch.set_num_threads(1)
    agent = train(args.condition, args.seed).eval()
    results = {"условие": args.condition, "seed": args.seed, "по дням сезона": {}}
    for world in TRAIN_WORLDS + NEW_WORLDS:
        results["по дням сезона"][world.name] = exam(agent, [world])
    results["по дням сезона"]["основной мир, новые предметы"] = exam(agent, [MAIN], things=NEW_THINGS, guests=NEW_GUESTS)
    results["по дням сезона"]["основной мир, значение 5 (не учился)"] = exam(agent, [MAIN], situations=NEW_SITUATIONS)
    results["по дням сезона"]["основной мир, числа до 20"] = exam(agent, [MAIN], limit=20)
    results["по дням сезона"]["основной мир, числа до 40"] = exam(agent, [MAIN], limit=40, seasons=6)
    results["анализ"] = analyse(agent)
    out = Path(__file__).parent / args.out
    out.mkdir(exist_ok=True)
    (out / f"{args.condition}__seed_{args.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(agent.state_dict(), out / f"{args.condition}__seed_{args.seed}.pt")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
