"""Нейроны с памятью GRU. В условии "plastic" к связям добавляется быстрая часть.

Быстрая часть (Hebb) живёт один день. Она меняется, когда нейроны работают вместе, а сила этого
решает сигнал modulation, который сеть сама вырабатывает, в том числе из подзатыльника. Внешнее
обучение настраивает только медленные веса: то, как именно связи будут меняться.
"""
import torch
from torch import nn

from world import MAX_ANSWER, TOKEN_SIZE


class Net(nn.Module):
    def __init__(self, condition, size=32, strength=0.1, speed=0.5):
        super().__init__()
        if condition not in ("reset", "keep", "plastic"):
            raise ValueError(condition)
        self.condition = condition
        self.size = size
        self.speed = speed
        self.cell = nn.GRUCell(TOKEN_SIZE, size)
        self.answer = nn.Linear(size, MAX_ANSWER + 1)
        if condition == "plastic":
            self.strength = nn.Parameter(torch.full((size, size), strength))
            self.modulation = nn.Linear(size, 1)
        self.plasticity_on = True

    def start_day(self, batch):
        self.hebb = torch.zeros(batch, self.size, self.size) if self.condition == "plastic" else None
        self.memory = torch.zeros(batch, self.size)

    def start_attempt(self):
        if self.condition != "keep":
            self.memory = torch.zeros_like(self.memory)

    def step(self, seen):
        before = self.memory
        if self.condition == "plastic":
            fast = (self.strength * self.hebb @ before.unsqueeze(-1)).squeeze(-1)
            after = self.cell(seen, torch.tanh(before + fast))
            if self.plasticity_on:
                mod = torch.tanh(self.modulation(after)).unsqueeze(-1)
                change = mod * (after.unsqueeze(1) * before.unsqueeze(2))
                self.hebb = (self.hebb + self.speed * change).clamp(-1, 1)
        else:
            after = self.cell(seen, before)
        self.memory = after
        return after
