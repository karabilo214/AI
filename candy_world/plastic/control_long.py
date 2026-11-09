"""Настройка быстрой части только на заведомо решаемой задаче, чтобы не подгонять под экзамен."""
import json
import random
import sys

import torch

from control import exam, play
from net import Net

STEPS = int(sys.argv[3])
strength, speed = float(sys.argv[1]), float(sys.argv[2])
torch.set_num_threads(2)
rows = []
for seed in range(int(sys.argv[4]), int(sys.argv[4]) + 2):
    torch.manual_seed(seed)
    net = Net("plastic", strength=strength, speed=speed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random(seed)
    for _ in range(STEPS):
        loss, _ = play(net, rng)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    on = exam(net)
    net.plasticity_on = False
    off = exam(net)
    rows.append({"seed": seed, "с быстрой частью": on, "без неё": off})
    print(strength, speed, seed, on, "| без неё", off, flush=True)
with open(f"results/control_long_{strength}_{speed}_{sys.argv[4]}.json", "w", encoding="utf-8") as file:
    json.dump(rows, file, ensure_ascii=False, indent=1)
