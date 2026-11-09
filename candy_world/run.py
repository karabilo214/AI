"""Запускает мир: агент живёт дни, учится, затем проходит проверки.

    python run.py --seeds 1
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch

from agent import Agent, learn_from_days
from world import FETCH_COST, NEW_GUESTS, NEW_THINGS, party, show

STEPS = 9000
BATCH = 128
CHECK_EVERY = 250
STAGES = ((0.0, 10), (1 / 3, 20), (2 / 3, 40))  # с какой доли обучения и до скольких гостей


def play(agent, candies, friends, thing=None, guest=None):
    with torch.no_grad():
        memory = agent.look([show(candies, friends, thing, guest)])
        fetched = agent.prepare(memory, greedy=True)[0][0].item()
    return party(candies, friends, fetched)


def need_rate(agent, pairs, thing=None, guest=None):
    """Только дни, когда конфет не хватает: тут "готово" не спасает. Доля точных походов."""
    pairs = [(c, f) for c, f in pairs if f > c]
    right = sum(play(agent, c, f, thing, guest)["fetched"] == f - c for c, f in pairs)
    return right / len(pairs)


def enough_rate(agent, pairs, thing=None, guest=None):
    """Дни, когда хватает: ходить незачем."""
    pairs = [(c, f) for c, f in pairs if f <= c]
    return sum(play(agent, c, f, thing, guest)["fetched"] == 0 for c, f in pairs) / len(pairs)


def pairs_in(low, high, count=150):
    rng = random.Random(7)
    return [(rng.randint(low, high), rng.randint(low, high)) for _ in range(count)]


def quality(agent, limit):
    """Отдельные от проверок примеры: и когда надо идти, и когда не надо. Выбирает лучшую версию."""
    pairs = [(random.Random(11 + i).randint(0, limit), random.Random(500 + i).randint(0, limit)) for i in range(120)]
    return (need_rate(agent, pairs) + enough_rate(agent, pairs)) / 2


def train(seed, out):
    """Менять можно, только если стало лучше: при резком падении откат и обучение осторожнее."""
    random.seed(seed)
    torch.manual_seed(seed)
    agent = Agent()
    rate = 2e-3
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
    window = []
    best_score, best_state, best_limit = -1.0, None, None
    rollbacks = 0
    finished_stages = {}
    for step in range(1, STEPS + 1):
        limit = [top for start, top in STAGES if step / STEPS >= start][-1]
        if limit != best_limit:
            if best_limit is not None:
                agent.load_state_dict(best_state)
                finished_stages[best_limit] = copy.deepcopy(best_state)
                rate *= 0.5
                optimizer = torch.optim.Adam(agent.parameters(), lr=rate)
            best_limit, best_score = limit, -1.0
        counts = [(random.randint(0, limit), random.randint(0, limit)) for _ in range(BATCH)]
        entropy = 0.02 * max(0.0, 1 - step / (STEPS * 0.8))
        window += learn_from_days(agent, optimizer, [show(c, f) for c, f in counts], counts, entropy)
        if step % CHECK_EVERY == 0:
            score = quality(agent, limit)
            n = len(window)
            print(
                f"день {step * BATCH}, гостей до {limit}: качество {score:.2f}, лучшее {max(best_score, 0):.2f}, "
                f"обижено {sum(o['upset'] for o in window) / n:.2f}, походов {sum(o['fetched'] for o in window) / n:.1f}",
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
                print(f"  откат к лучшей версии, номер {rollbacks}, сила поправок теперь {rate:.5f}", flush=True)
    agent.load_state_dict(best_state)
    finished_stages[best_limit] = copy.deepcopy(best_state)
    torch.save(finished_stages, out / f"agent_stages_{seed}.pt")
    return finished_stages, rollbacks


def exam(agent, limit):
    apple, guest = NEW_THINGS, NEW_GUESTS
    wider = limit * 2
    results = {
        f"до {limit} (учился)": need_rate(agent, pairs_in(0, limit)),
        f"{limit + 1}-{wider} (не видел)": need_rate(agent, pairs_in(limit + 1, wider)),
        f"конфет 0-5, гостей {wider - 10}-{wider} (не видел)": need_rate(
            agent, [(c, f) for c in range(6) for f in range(wider - 10, wider + 1, 2)]
        ),
        f"новые предметы и гости, до {limit}": need_rate(agent, pairs_in(0, limit), apple, guest),
        f"хватает, ходить незачем, до {limit}": enough_rate(agent, pairs_in(0, limit)),
        "хватает, ходить незачем, новые предметы": enough_rate(agent, pairs_in(0, limit), apple, guest),
    }
    for key, value in results.items():
        print(f"  {key}: {value:.1%}")
    results["ноль предметов, гостей -> сходил"] = {f: play(agent, 0, f)["fetched"] for f in (5, 10, 20, 30, 40, 60, 80)}
    results["ноль новых предметов, новых гостей -> сходил"] = {f: play(agent, 0, f, apple, guest)["fetched"] for f in (5, 10, 20, 30, 40)}
    print("  ", results["ноль предметов, гостей -> сходил"])
    print("  новые:", results["ноль новых предметов, новых гостей -> сходил"])
    return results


def run(seed, out):
    print(f"\n=== запуск {seed}, поход стоит {FETCH_COST} ===")
    stages, rollbacks = train(seed, out)
    results = {"seed": seed, "откатов": rollbacks}
    agent = Agent()
    for limit, state in stages.items():
        agent.load_state_dict(state)
        print(f"Версия с этапа до {limit} гостей:")
        results[f"этап {limit}"] = exam(agent, limit)
    (out / f"results_{seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    out = Path(__file__).parent / "results_v2"
    out.mkdir(exist_ok=True)
    for seed in range(args.seeds):
        run(seed, out)
