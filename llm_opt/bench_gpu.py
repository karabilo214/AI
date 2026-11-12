"""Видеокарта: стоимость проверки T слов одним проходом против одного слова (контекст 512).
Своя реализация core.Qwen с кэшем. Медиана 15 кругов, отношения внутри круга."""
import statistics
import sys
import time

import torch

from core import L, Qwen

dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[1]]
dev = "cuda"
m = Qwen(dev, dtype)
torch.manual_seed(0)
ids = torch.randint(1000, 50000, (1, 600), device=dev)
cache0 = [None] * L
m.forward(ids[:, :512], cache=cache0)
sync = torch.cuda.synchronize


def run(T):
    cache = [(k, v) for k, v in cache0]  # тот же контекст 512 (cat создаёт новые тензоры, исходный не портится)
    sync(); t0 = time.perf_counter()
    m.forward(ids[:, 512:512 + T], cache=cache)
    sync()
    return (time.perf_counter() - t0) * 1000


Ts = (1, 2, 4, 8, 16, 32)
for T in Ts:
    for _ in range(3): run(T)
res = {T: [] for T in Ts}
for _ in range(15):
    for T in Ts:
        res[T].append(run(T))
print(f"{torch.cuda.get_device_name()}, {sys.argv[1]}")
for T in Ts:
    rat = statistics.median([a / b for a, b in zip(res[T], res[1])])
    print(f"  T={T:2d}: медиана {statistics.median(res[T]):6.2f} мс, лучший {min(res[T]):6.2f}, к одному слову {rat:.2f}")
