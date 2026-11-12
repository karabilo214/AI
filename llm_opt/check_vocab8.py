"""Проверка vocab8: на настоящих состояниях модели (генерация FusedEngine) ответ = argmax полного словаря;
сколько кандидатов; время в графе против полного словаря + argmax для T=1..16."""
import statistics

import torch

from fused_decode import FusedEngine, gemv
from spec_gpu import plain, prompts
from vocab8 import Vocab8, vocab_a_k, vocab_b_k

eng = FusedEngine()
v8 = Vocab8(eng.emb16)
hs = []
orig = eng._head
for p in prompts[:4]:  # собираем конечные состояния h без графа
    out = eng.prefill(p)
    tok, pos = int(out[-1]), len(p)
    for k in range(64):
        eng._step(1) if False else None
        eng.tok[:1].fill_(tok); eng.pos.fill_(pos + k)
        torch.cuda.synchronize()
        eng.graphs[1].replay()
        hs.append(eng.h[:1].clone())
        tok = int(eng.outs[1][0])
H = torch.cat(hs)
lg = torch.empty(len(H), eng.emb16.shape[0], device="cuda")
ok, nc = 0, []
for i in range(len(H)):
    h = H[i:i + 1].contiguous()
    gemv(h, eng.emb16, lg[i:i + 1], 0, 1)
    ref = int(lg[i].argmax())
    V, K = v8.V, v8.K
    vocab_a_k[lambda m: (-(-V // m["BN"]),)](h, v8.Q, v8.S, v8.C1, v8.C2, v8.AM, v8.HI, v8.LO, 1, V, K)
    nc.append(int((v8.HI[0] >= v8.LO[0]).sum()))
    v8.LO.fill_(-float("inf"))
    got = int(v8(h, 1)[0])
    ok += got == ref
nc = torch.tensor(nc, dtype=torch.float)
print(f"совпало с полным словарём {ok}/{len(H)}; кандидатов медиана {nc.median():.0f}, макс {nc.max():.0f}")
# пачки: каждая строка своя
for T in (2, 8, 16):
    h = H[:T].contiguous()
    gemv(h, eng.emb16, lg[:T], 0, T)
    print(f"  пачка T={T}: совпало {(v8(h, T) == lg[:T].argmax(-1)).sum().item()}/{T}")


def bench_graph(f, n=30):
    f(); f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(3): g.replay()
    r = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        r.append(s.elapsed_time(e))
    return statistics.median(r)


for T in (1, 2, 4, 8, 16):
    h = H[:T].contiguous()
    t_full = bench_graph(lambda: (gemv(h, eng.emb16, lg[:T], 0, T), lg[:T].argmax(-1)))
    t_v8 = bench_graph(lambda: v8(h, T))
    print(f"T={T:2d}: полный словарь {t_full:.3f} мс, int8 {t_v8:.3f} мс", flush=True)
