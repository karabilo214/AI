"""Быстрый движок обязан давать ровно те же ответы, что и обычный, на любых таблицах и устройствах."""
import random
import time

import numpy as np

import engine
import evolve
import fast_engine

rng = np.random.default_rng(0)
cases = [("count", v) for v in evolve.VARIANTS] + [("multiply", "base"), ("guests", "base"), ("parity", "base"), ("double", "base")]
for task, variant in cases:
    t, design, widths, items, exams = evolve.setup(task, variant)
    data = evolve.data(t, items, 6, 0)
    days = data[0]
    for trial in range(3):
        table = rng.integers(0, 2, design.size, dtype=np.uint8)
        for width in (widths[0], widths[-1]):
            a = engine.run_days(table, design, days, width, t.CODES, record=True)
            b = fast_engine.run_days(table, design, days, width, t.CODES, record=True)
            assert np.array_equal(a[0], b[0]), (task, variant, "fetched")
            lengths = np.array([len(d) for d in days])
            real = np.arange(a[1].shape[1])[None, :] < lengths[:, None]   # только настоящие события, не пустые места
            assert np.array_equal(a[1][real], b[1][real]), (task, variant, "said")
            assert abs(a[2]["обращений на событие"] - b[2]["обращений на событие"]) < 1e-9, (task, variant)
            assert abs(a[2]["пересчётов на событие"] - b[2]["пересчётов на событие"]) < 1e-9, (task, variant)
    s = time.perf_counter(); engine.run_days(table, design, days, widths[-1], t.CODES); slow = time.perf_counter() - s
    s = time.perf_counter(); fast_engine.run_days(table, design, days, widths[-1], t.CODES); fast = time.perf_counter() - s
    print(f"{task}/{variant}: совпадает; обычный {slow*1000:.1f} мс, быстрый {fast*1000:.2f} мс, быстрее в {slow/fast:.0f} раз", flush=True)
print("ВСЕ СОВПАДАЮТ")
