"""Где тратится время одного шага (T=1) FusedEngine: время каждой программы видеокарты внутри графа (torch.profiler)
и сколько уходит на промежутки между ними. Модуль движка — аргумент (по умолчанию fused_decode)."""
import importlib
import statistics
import sys
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from spec_gpu import prompts

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "fused_decode")
eng = mod.FusedEngine()
p = prompts[0]
out = eng.prefill(p)
tok = int(out[-1])
pos = len(p)
for _ in range(5):
    eng.run([tok], pos)
torch.cuda.synchronize()

# чистое время шага на видеокарте (события), 50 повторов
ts = []
for _ in range(50):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); eng.graphs[1].replay(); e.record(); torch.cuda.synchronize()
    ts.append(s.elapsed_time(e))
print(f"шаг T=1 (граф, события): медиана {statistics.median(ts):.3f} мс, мин {min(ts):.3f}")

R = 10
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(R):
        eng.graphs[1].replay()
    torch.cuda.synchronize()
ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
tot = defaultdict(float); cnt = defaultdict(int)
for e in ev:
    tot[e.name] += e.device_time / 1000 / R  # мс на шаг
    cnt[e.name] += 1
busy = sum(tot.values())
print(f"сумма времени программ на шаг {busy:.3f} мс; программ на шаг {sum(cnt.values()) / R:.0f}")
for n, t in sorted(tot.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t:7.3f} мс  {cnt[n] / R:5.0f} шт  {n[:90]}")
