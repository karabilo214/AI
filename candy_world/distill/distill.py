"""Перевод работающей сети в дешёвую форму. Учитель: GRU-32, сдавшая экзамен. Ученик учится повторять его.

Ступень "ternary": веса только −1, 0, +1 (и один множитель на строку). Умножений нет, только сложения.
Ступень "binary": к тому же память из 32 бит (каждое число памяти только −1 или +1).

    python distill.py --level ternary --seed 0
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v100"))
from agent import Agent  # noqa: E402
from run import exam  # noqa: E402
from world import OLD_GUESTS, OLD_SITUATIONS, OLD_THINGS, TIME_TO_ACT, TOKEN_SIZE, make_day, show  # noqa: E402

TEACHERS = Path(__file__).resolve().parent.parent / "v100" / "results_long"
STEPS = 3000
BATCH = 128


class RoundSign(torch.autograd.Function):
    """Вперёд: знак (−1 или +1). Назад: пропускает градиент как есть, иначе учиться было бы нечем."""

    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        return grad


class RoundTernary(torch.autograd.Function):
    """Вперёд: −1, 0 или +1 (маленькие веса становятся нулём). Назад: градиент как есть."""

    @staticmethod
    def forward(ctx, w):
        cut = 0.5 * w.abs().mean(dim=1, keepdim=True)
        return torch.where(w > cut, torch.ones_like(w), torch.where(w < -cut, -torch.ones_like(w), torch.zeros_like(w)))

    @staticmethod
    def backward(ctx, grad):
        return grad


def cheap(weight):
    """Веса −1/0/+1 и один множитель на строку."""
    signs = RoundTernary.apply(weight)
    scale = (weight.abs() * signs.abs()).sum(1, keepdim=True) / signs.abs().sum(1, keepdim=True).clamp(min=1)
    return signs * scale


class CheapCell(nn.Module):
    """Та же клетка памяти GRU, но веса дешёвые, а память, если binary, из 32 бит."""

    def __init__(self, teacher_cell, binary):
        super().__init__()
        self.weight_ih = nn.Parameter(teacher_cell.weight_ih.detach().clone())
        self.weight_hh = nn.Parameter(teacher_cell.weight_hh.detach().clone())
        self.bias_ih = nn.Parameter(teacher_cell.bias_ih.detach().clone())
        self.bias_hh = nn.Parameter(teacher_cell.bias_hh.detach().clone())
        self.hidden_size = teacher_cell.hidden_size
        self.binary = binary

    def forward(self, x, h):
        gi = x @ cheap(self.weight_ih).T + self.bias_ih
        gh = h @ cheap(self.weight_hh).T + self.bias_hh
        ir, iz, in_ = gi.chunk(3, 1)
        hr, hz, hn = gh.chunk(3, 1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn)
        new = (1 - z) * n + z * h
        return RoundSign.apply(new) if self.binary else new


class CheapHead(nn.Module):
    def __init__(self, teacher_head):
        super().__init__()
        self.weight = nn.Parameter(teacher_head.weight.detach().clone())
        self.bias = nn.Parameter(teacher_head.bias.detach().clone())

    def forward(self, h):
        return h @ cheap(self.weight).T + self.bias


def make_student(teacher, level):
    student = Agent(32, "cpu")
    student.load_state_dict(teacher.state_dict())
    student.memory_cell = CheapCell(teacher.memory_cell, binary=(level == "binary"))
    student.choose = CheapHead(teacher.choose)
    return student


def teach(student, teacher, seed):
    """Ученик видит те же дни и на каждом шаге подготовки должен решать так же, как учитель."""
    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    for step in range(1, STEPS + 1):
        limit = random.choice((3, 6, 10))
        days = [make_day(limit, OLD_SITUATIONS) for _ in range(BATCH)]
        shown = [show(d, OLD_THINGS, OLD_GUESTS) for d in days]
        with torch.no_grad():
            teacher_fetched = teacher.prepare(teacher.look(shown), greedy=True)[0]
        memory = student.look(shown)
        act = torch.tensor([[0.0] * TOKEN_SIZE] * BATCH)
        act[:, TIME_TO_ACT] = 1.0                          # отметка "пора действовать"
        losses = []
        for move in range(int(teacher_fetched.max().item()) + 1):
            memory = student.memory_cell(act, memory)
            target = (move >= teacher_fetched).long()   # 0: сходить, 1: готово, как решил учитель
            losses.append(nn.functional.cross_entropy(student.choose(memory), target, reduction="none")
                          * (move <= teacher_fetched).float())
        loss = torch.stack(losses).sum(0).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"шаг {step}: ошибка {loss.item():.3f}", flush=True)
    return student.eval()


def answer_time(agent, days=1000, seed=3):
    rng = random.Random(seed)
    shown = [show(make_day(10, OLD_SITUATIONS, rng), OLD_THINGS, OLD_GUESTS, rng) for _ in range(days)]
    best = 1e9
    for _ in range(3):
        started = time.perf_counter()
        with torch.no_grad():
            agent.prepare(agent.look(shown), greedy=True)
        best = min(best, time.perf_counter() - started)
    return round(best * 1000, 1)


def operations(level):
    """Операции на одно событие у клетки памяти (32 нейрона, 32 входа)."""
    count = 3 * 32 * (TOKEN_SIZE + 32)
    if level == "teacher":
        return {"умножений": count, "сложений": count}
    if level == "ternary":
        return {"умножений": 3 * 32 * 2, "сложений": count, "примечание": "умножения только на множитель строки"}
    return {"умножений": 3 * 32 * 2, "сложений": count, "примечание": "память из 32 бит: можно считать битовыми операциями"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["ternary", "binary"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    teacher = Agent(32, "cpu")
    teacher.load_state_dict(torch.load(TEACHERS / f"memory_32__rate_0.002__seed_{args.seed}.pt"))
    teacher.eval()
    raw = make_student(teacher, args.level).eval()
    before = exam(raw)
    student = teach(make_student(teacher, args.level), teacher, args.seed)
    result = {
        "ступень": args.level, "seed": args.seed,
        "учитель": exam(teacher),
        "ученик сразу после округления, без обучения": before,
        "ученик после обучения у учителя": exam(student),
        "операций на событие": {"учитель": operations("teacher"), "ученик": operations(args.level)},
        "время на 1000 дней, мс": {"учитель": answer_time(teacher), "ученик": answer_time(student)},
    }
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.level}__seed_{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(student.state_dict(), out / f"{args.level}__seed_{args.seed}.pt")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
