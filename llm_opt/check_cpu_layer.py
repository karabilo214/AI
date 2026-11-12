"""Слой на процессоре против torch fp32 на видеокарте (та же математика, что Reference): разница и время."""
import time, statistics
import numpy as np
import torch
import numba
from big_engine import Ckpt, Reference
from cpu_layer import CpuLayer

print("потоки numba:", numba.config.NUMBA_NUM_THREADS, "слой потоков:", numba.config.THREADING_LAYER)
ck = Ckpt("Qwen3-4B"); c = ck.cfg
H, KV, HD, eps, F = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"], c["intermediate_size"]
w = ck.layer(5)
inv = (1.0 / (c["rope_theta"] ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD)))
cl = CpuLayer(w, c, 1024)
torch.manual_seed(0)
T0 = 40
X = torch.randn(T0, c["hidden_size"]) * 3
# эталон: torch fp32 на видеокарте, все T0 слов сразу с причинной маской
wg = {k: v.cuda().float() for k, v in w.items()}
x = X.cuda()
rms = Reference.rms
fr = torch.arange(T0, device="cuda").float()[:, None] * inv.cuda()[None]
e = torch.cat([fr, fr], -1); cos, sin = e.cos(), e.sin()
rot = lambda t: torch.cat([-t[..., HD // 2:], t[..., :HD // 2]], -1)
h = rms(x, wg["ln1"], eps); qkv = h @ wg["qkv"].T
q = rms(qkv[:, :H * HD].view(T0, H, HD), wg["qn"], eps); k = rms(qkv[:, H * HD:(H + KV) * HD].view(T0, KV, HD), wg["kn"], eps)
v = qkv[:, (H + KV) * HD:].view(T0, KV, HD)
q = q * cos[:, None] + rot(q) * sin[:, None]; k = k * cos[:, None] + rot(k) * sin[:, None]
k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
a = torch.softmax((q.transpose(0, 1) @ k.permute(1, 2, 0)) * HD ** -0.5 + torch.full((T0, T0), float("-inf"), device="cuda").triu(1), -1)
x = x + (a @ v.transpose(0, 1)).transpose(0, 1).reshape(T0, H * HD) @ wg["o"].T
h = rms(x, wg["ln2"], eps); gu = h @ wg["gu"].T
ref = (x + (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ wg["down"].T).cpu()
# процессор: первые 24 слова пачкой, затем по одному и пачкой 8 (как генерация и проверка)
Xn = X.numpy().copy()
a1 = Xn[:24].copy(); cl(a1, 0, inv.numpy())
outs = [a1]
for t in range(24, 32):
    r = Xn[t:t + 1].copy(); cl(r, t, inv.numpy()); outs.append(r)
r = Xn[32:40].copy(); cl(r, 32, inv.numpy()); outs.append(r)
got = torch.from_numpy(np.concatenate(outs))
print(f"разница с torch fp32: {((got - ref).abs().max() / ref.abs().max()).item():.1e}")
def tm(f, n=5):
    f(); r = []
    for _ in range(n):
        t = time.perf_counter(); f(); r.append(time.perf_counter() - t)
    return statistics.median(r)
by = sum(cl.W[n].nbytes for n in cl.W)
for T in (1, 4, 16):
    xx = np.random.randn(T, c["hidden_size"]).astype(np.float32)
    t = tm(lambda: cl(xx.copy(), 300, inv.numpy()))
    print(f"T={T:2d}: слой {t * 1000:.1f} мс ({by / t / 1e9:.1f} ГБ/с весов)", flush=True)
