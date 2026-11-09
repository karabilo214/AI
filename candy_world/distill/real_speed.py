"""Проверка до всякого сжатия: где у учителя уходит время и даёт ли настоящий 8-битный расчёт выигрыш.

    python real_speed.py
"""
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v100"))
from agent import Agent  # noqa: E402
from run import exam  # noqa: E402
from world import OLD_GUESTS, OLD_SITUATIONS, OLD_THINGS, TOKEN_SIZE, make_day, show  # noqa: E402

torch.set_num_threads(1)
torch.backends.quantized.engine = "qnnpack"   # единственный доступный на этом Mac движок 8-битного расчёта
TEACHERS = Path(__file__).resolve().parent.parent / "v100" / "results_long"


def load(seed):
    agent = Agent(32, "cpu")
    agent.load_state_dict(torch.load(TEACHERS / f"memory_32__rate_0.002__seed_{seed}.pt"))
    return agent.eval()


def days(count, seed=3):
    rng = random.Random(seed)
    return [show(make_day(10, OLD_SITUATIONS, rng), OLD_THINGS, OLD_GUESTS, rng) for _ in range(count)]


def answer_time(agent, shown, repeats=5):
    best = 1e9
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            agent.prepare(agent.look(shown), greedy=True)
        best = min(best, time.perf_counter() - started)
    return best


class Timed(torch.nn.Module):
    """Обёртка, считающая время внутри клетки памяти."""

    def __init__(self, cell):
        super().__init__()
        self.cell, self.seconds, self.calls = cell, 0.0, 0

    def forward(self, x, h):
        started = time.perf_counter()
        out = self.cell(x, h)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return out


print("движок 8-битного расчёта:", torch.backends.quantized.engine, "| доступны:", torch.backends.quantized.supported_engines)
teacher = load(0)
for count in (1, 1000):
    shown = days(count)
    agent = load(0)
    timed = Timed(agent.memory_cell)
    agent.memory_cell = timed
    total = answer_time(agent, shown, repeats=1)
    print(f"дней {count}: всего {total*1000:.1f} мс, из них в клетке памяти {timed.seconds*1000:.1f} мс "
          f"({100*timed.seconds/total:.0f}%), вызовов клетки {timed.calls}")

int8 = torch.ao.quantization.quantize_dynamic(load(0), {torch.nn.GRUCell, torch.nn.Linear}, dtype=torch.qint8)
print("в 8-битной версии клетка:", type(int8.memory_cell).__name__)
for count in (1, 1000):
    shown = days(count)
    t_float, t_int8 = answer_time(teacher, shown), answer_time(int8, shown)
    print(f"дней {count}: обычный расчёт {t_float*1000:.1f} мс, 8-битный {t_int8*1000:.1f} мс, "
          f"8-битный быстрее в {t_float/t_int8:.2f} раза")

x = torch.randn(1000, TOKEN_SIZE)
h = torch.randn(1000, 32)
for name, cell in (("обычная клетка", teacher.memory_cell), ("8-битная клетка", int8.memory_cell)):
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            cell(x, h)
    print(f"{name}: {(time.perf_counter() - started) / 200 * 1e6:.0f} мкс на шаг для 1000 дней сразу")

print("\nточность 8-битной версии без всякого дообучения (учитель / 8 бит):")
for seed in range(3):
    t, q = exam(load(seed)), exam(torch.ao.quantization.quantize_dynamic(load(seed), {torch.nn.GRUCell, torch.nn.Linear}, dtype=torch.qint8))
    print(f"  запуск {seed}:", {k[:22]: (t[k]["точно"], q[k]["точно"]) for k in t})
