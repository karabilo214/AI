"""Агент: рекуррентная сеть, то есть сеть с памятью внутри, и обучение на последствиях.

Сеть читает предметы по одному и хранит в памяти, что сочтёт нужным. Никто не говорит, что
запоминать и что считать нужно. После дня она получает только итоговый результат: чем лучше
кончился день, тем сильнее закрепляются сделанные шаги, чем хуже, тем сильнее они ослабляются.
"""
import torch
from torch import nn

from world import MAX_FETCHES, TIME_TO_ACT, TOKEN_SIZE, party, token

FETCH, DONE = 0, 1


class Agent(nn.Module):
    def __init__(self, memory_size=32, device="cpu"):
        super().__init__()
        self.memory_size = memory_size
        self.device = device
        self.memory_cell = nn.GRUCell(TOKEN_SIZE, memory_size)
        self.choose = nn.Linear(memory_size, 2)   # пойти за конфетой или "готово"
        self.expect = nn.Linear(memory_size, 1)   # какого итога дня он ждёт
        self.to(device)

    def look(self, days):
        """Просмотр предметов. `days` это список дней, в каждом свой список предметов."""
        memory = torch.zeros(len(days), self.memory_size, device=self.device)
        for step in range(max(map(len, days), default=0)):
            active = [i for i, items in enumerate(days) if step < len(items)]
            if not active:
                break
            seen = torch.tensor([days[i][step] for i in active], dtype=torch.float, device=self.device)
            updated = memory.clone()
            updated[active] = self.memory_cell(seen, memory[active])
            memory = updated
        return memory

    def prepare(self, memory, greedy=False):
        """Подготовка: по шагу, пока каждый не скажет "готово"."""
        count = memory.shape[0]
        fetched = torch.zeros(count, dtype=torch.long, device=self.device)
        running = torch.ones(count, dtype=torch.bool, device=self.device)
        act_token = torch.tensor([token(TIME_TO_ACT)] * count, dtype=torch.float, device=self.device)
        log_probs, expected, masks, entropies = [], [], [], []
        for _ in range(MAX_FETCHES + 1):
            memory = torch.where(running.unsqueeze(1), self.memory_cell(act_token, memory), memory)
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
        return fetched, log_probs, expected, masks, entropies


def learn_from_days(agent, optimizer, shown, days, entropy_weight):
    """Один шаг обучения: прожить дни, посмотреть итог, поправить сеть."""
    memory = agent.look(shown)
    fetched, log_probs, expected, masks, entropies = agent.prepare(memory)
    outcomes = [party(day, n) for day, n in zip(days, fetched.tolist())]
    scores = torch.tensor([o["score"] for o in outcomes], dtype=torch.float, device=agent.device) / 10
    loss = 0.0
    for log_p, value, mask, entropy in zip(log_probs, expected, masks, entropies):
        surprise = scores - value
        loss = loss + (mask * (-log_p * surprise.detach() + 0.5 * surprise**2 - entropy_weight * entropy)).sum()
    loss = loss / len(shown)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()
    return outcomes
