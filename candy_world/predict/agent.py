"""Агент с двумя уровнями памяти.

Долговременный уровень: основная сеть (memory_cell, choose, predict). Учится медленно, общая для всех миров.
Оперативный уровень: маленький вектор правил context (8 чисел). Живёт один сезон в одном мире и
меняется только по итогу каждого дня. Какой это мир, агенту не говорят.
"""
import torch
from torch import nn

from world import MAX_FETCHES, TIME_TO_ACT, TOKEN_SIZE, party, token

FETCH, DONE = 0, 1
CONTEXT = 8


class Agent(nn.Module):
    def __init__(self, memory_size=32, use_context=False, carry_memory=False):
        super().__init__()
        self.memory_size = memory_size
        self.use_context = use_context      # отдельная оперативная память правил (механизм C)
        self.carry_memory = carry_memory    # контроль: правила держит та же основная память
        self.memory_cell = nn.GRUCell(TOKEN_SIZE + CONTEXT, memory_size)
        self.choose = nn.Linear(memory_size, 2)
        self.expect = nn.Linear(memory_size, 1)
        self.predict = nn.Linear(memory_size, 1)
        self.context_cell = nn.GRUCell(3 + memory_size, CONTEXT)

    def new_season(self, count):
        self.context = torch.zeros(count, CONTEXT)
        self.carried = torch.zeros(count, self.memory_size)

    def step(self, seen, memory, rows):
        return self.memory_cell(torch.cat([seen, self.context[rows]], 1), memory)

    def look(self, days, guesses=None):
        rows = torch.arange(len(days))
        memory = self.carried.clone() if self.carry_memory else torch.zeros(len(days), self.memory_size)
        for position in range(max(map(len, days))):
            active = [i for i, items in enumerate(days) if position < len(items)]
            seen = torch.tensor([days[i][position] for i in active], dtype=torch.float)
            updated = memory.clone()
            updated[active] = self.step(seen, memory[active], rows[active])
            memory = updated
            if guesses is not None:
                guesses.append((active, self.predict(memory[active]).squeeze(-1)))
        return memory

    def prepare(self, memory, greedy=False):
        count = memory.shape[0]
        rows = torch.arange(count)
        fetched = torch.zeros(count, dtype=torch.long)
        running = torch.ones(count, dtype=torch.bool)
        act = torch.tensor([token(TIME_TO_ACT)] * count, dtype=torch.float)
        log_probs, expected, masks, entropies = [], [], [], []
        for _ in range(MAX_FETCHES + 1):
            memory = torch.where(running.unsqueeze(1), self.step(act, memory, rows), memory)
            policy = torch.distributions.Categorical(logits=self.choose(memory))
            action = policy.probs.argmax(-1) if greedy else policy.sample()
            log_probs.append(policy.log_prob(action))
            entropies.append(policy.entropy())
            expected.append(self.expect(memory).squeeze(-1))
            masks.append(running.clone())
            fetching = running & (action == FETCH) & (fetched < MAX_FETCHES)
            fetched = fetched + fetching.long()
            running = fetching
            if not running.any():
                break
        return fetched, log_probs, expected, masks, entropies, memory

    def end_of_day(self, outcomes, memory):
        """Итог дня меняет только оперативную память правил (или основную, в контроле)."""
        feedback = torch.tensor(
            [[float(o["short"] > 0), float(o["extra"] > 0), min(o["short"] + o["extra"], 10) / 10] for o in outcomes]
        )
        if self.use_context:
            self.context = self.context_cell(torch.cat([feedback, memory], 1), self.context)
        if self.carry_memory:
            self.carried = memory


def live_day(agent, shown, days, rules, targets, predict_weight, entropy_weight, greedy=False):
    """Один день у всех агентов пачки. Возвращает потерю для обучения и итоги дня."""
    guesses = [] if predict_weight > 0 else None
    memory = agent.look(shown, guesses)
    fetched, log_probs, expected, masks, entropies, last = agent.prepare(memory, greedy)
    outcomes = [party(d, n, r) for d, n, r in zip(days, fetched.tolist(), rules)]
    scores = torch.tensor([o["score"] for o in outcomes]) / 10
    loss = 0.0
    for log_p, value, mask, entropy in zip(log_probs, expected, masks, entropies):
        surprise = scores - value
        loss = loss + (mask * (-log_p * surprise.detach() + 0.5 * surprise**2 - entropy_weight * entropy)).sum()
    loss = loss / len(shown)
    if guesses:
        terms, count = [], 0
        for position, (active, guess) in enumerate(guesses):
            pairs = [(k, targets[i][position]) for k, i in enumerate(active) if targets[i][position] is not None]
            if pairs:
                rows = torch.tensor([k for k, _ in pairs])
                truth = torch.tensor([float(t) for _, t in pairs])
                terms.append(nn.functional.binary_cross_entropy_with_logits(guess[rows], truth, reduction="sum"))
                count += len(pairs)
        if terms:
            loss = loss + predict_weight * torch.stack(terms).sum() / count
    agent.end_of_day(outcomes, last)
    return loss, outcomes
