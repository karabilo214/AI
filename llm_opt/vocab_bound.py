"""Точный отсев словаря: грубые оценки по сжатому словарю + строгая граница ошибки -> сколько слов надо пересчитать точно.
Сжатие по строкам (или группам по G чисел): q = round(e / s), ошибка d = e - s*q известна.
Граница для слова v: |d_v . h| <= min(||d_v||_2 * ||h||_2, sum_g max|d_vg| * ||h_g||_1).
Кандидаты: approx_v + B_v >= max_u (approx_u - B_u). Победитель среди кандидатов (точно) = победитель полного словаря."""
import torch

from core import Qwen, wiki_tokens, windows

torch.manual_seed(0)
m = Qwen("mps")
ids = windows(wiki_tokens("test").to("mps"), 2, T=512, start=0)
hs = []
m.forward(ids, {"head_fn": lambda h: hs.append(h.reshape(-1, h.shape[-1]).cpu()) or torch.zeros(1)})
H = hs[0].double()  # [1024, D]
E = m.emb.cpu().double()  # [V, D]
full = H @ E.T
best = full.argmax(-1)
print(f"позиций {H.shape[0]}; |h|_2 медиана {H.norm(dim=-1).median():.1f}")


def quant(E, bits, G):
    V, D = E.shape
    Eg = E.view(V, D // G, G)
    qmax = 2 ** (bits - 1) - 1
    s = Eg.abs().amax(-1, keepdim=True) / qmax
    q = torch.round(Eg / s).clamp(-qmax - 1, qmax)
    Ea = (q * s).view(V, D)
    return Ea, (E - Ea)


for bits, G in ((8, 1024), (8, 128), (4, 128), (4, 32)):
    Ea, d = quant(E, bits, G)
    dn = d.norm(dim=-1)  # [V]
    dmax = d.view(E.shape[0], -1, G).abs().amax(-1)  # [V, D/G]
    cnt = []
    for i in range(0, H.shape[0], 128):
        h = H[i:i + 128]
        ap = h @ Ea.T
        b1 = dn[None, :] * h.norm(dim=-1, keepdim=True)
        hg1 = h.view(h.shape[0], -1, G).abs().sum(-1)  # [B, D/G]
        b2 = hg1 @ dmax.T
        B = torch.minimum(b1, b2)
        lo = (ap - B).max(-1, keepdim=True).values
        cand = (ap + B) >= lo
        assert cand.gather(1, best[i:i + 128, None]).all()
        cnt.append(cand.sum(-1).float())
    cnt = torch.cat(cnt)
    byts = E.shape[0] * E.shape[1] * bits / 8 + E.shape[0] * (E.shape[1] // G) * 2
    print(f"{bits} бит, группа {G}: {byts / 1e6:.0f} МБ (полный 311); кандидатов: медиана {cnt.median():.0f}, "
          f"90% {cnt.quantile(0.9):.0f}, макс {cnt.max():.0f}", flush=True)
