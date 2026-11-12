"""Время программ видеокарты при чтении запроса (128 слов, граф) в fused2."""
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from fused2 import FusedEngine
from spec_gpu import prompts

eng = FusedEngine()
for p in prompts[:3]: eng.prefill(p)
torch.cuda.synchronize()
R = 5
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for p in prompts[:R]:
        eng.prefill(p)
    torch.cuda.synchronize()
tot = defaultdict(float); cnt = defaultdict(int)
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        tot[e.name] += e.device_time / 1000 / R; cnt[e.name] += 1
print(f"сумма {sum(tot.values()):.2f} мс на запрос, программ {sum(cnt.values()) / R:.0f}")
for n, t in sorted(tot.items(), key=lambda x: -x[1])[:14]:
    print(f"  {t:6.2f} мс {cnt[n] / R:5.0f} шт  {n[:100]}")
