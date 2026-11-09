"""Обучение и экзамен. На экзамене внешнее обучение выключено.

    python run.py --condition plastic --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from net import Net
from world import NEW_RULE, TRAIN_RULES, make_scene, result_token, right_answer, scene_tokens

ATTEMPTS = 6
BATCH = 64
STEPS = 16000


def play_day(net, rules, rng, batch, greedy=False):
    """Один день у batch агентов. Возвращает ошибку по законам мира и точность по попыткам."""
    net.start_day(batch)
    total = 0.0
    right_by_attempt = []
    for _ in range(ATTEMPTS):
        net.start_attempt()
        scenes = [make_scene(rng) for _ in range(batch)]
        streams = [scene_tokens(scene, rng) for scene in scenes]
        length = max(map(len, streams))
        padded = [[[0.0] * len(stream[0])] * (length - len(stream)) + stream for stream in streams]
        for position in range(length):
            state = net.step(torch.tensor([row[position] for row in padded]))
        log_policy = net.answer(state).log_softmax(-1)
        policy = log_policy.exp()
        right = torch.tensor([right_answer(scene, rule) for scene, rule in zip(scenes, rules)])
        # Правильный ответ знает только внешнее обучение. Сам агент видит лишь итог попытки.
        total = total + torch.nn.functional.nll_loss(log_policy, right)
        picked = policy.argmax(-1) if greedy else torch.multinomial(policy, 1).squeeze(-1)
        right_by_attempt.append((picked == right).float().mean().item())
        feedback = [result_token(a, r) for a, r in zip(picked.tolist(), right.tolist())]
        net.step(torch.tensor(feedback))
    return total / ATTEMPTS, right_by_attempt


def exam(net, rule, days=30, seed=9):
    rng = random.Random(seed)
    net.eval()
    rows = []
    with torch.no_grad():
        for _ in range(days):
            rows.append(play_day(net, [rule] * BATCH, rng, BATCH, greedy=True)[1])
    net.train()
    return [round(sum(r[i] for r in rows) / days, 3) for i in range(ATTEMPTS)]


def run(condition, seed, out):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = Net(condition)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random(seed + 100)
    names = list(TRAIN_RULES)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        rules = [TRAIN_RULES[rng.choice(names)] for _ in range(BATCH)]
        loss, _ = play_day(net, rules, rng, BATCH)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"{condition} {seed} шаг {step}: ошибка {loss.item():.3f}, секунд {time.perf_counter() - started:.0f}", flush=True)
    results = {"условие": condition, "seed": seed, "секунд": round(time.perf_counter() - started),
               "весов": sum(p.numel() for p in net.parameters())}
    for name, rule in {**TRAIN_RULES, **NEW_RULE}.items():
        results[name] = exam(net, rule)
    if condition == "plastic":
        net.plasticity_on = False
        results["выключена быстрая часть"] = {name: exam(net, rule) for name, rule in {**TRAIN_RULES, **NEW_RULE}.items()}
    (out / f"{condition}__seed_{seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(net.state_dict(), out / f"{condition}__seed_{seed}.pt")
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("готово", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    target = Path(__file__).parent / "results"
    target.mkdir(exist_ok=True)
    run(args.condition, args.seed, target)
