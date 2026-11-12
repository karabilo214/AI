"""Шаг 3. Угадыватель кандидатов для словаря.

Грубые оценки всех слов: (h A) B, A [1024, r], B [r, V]  -> 1024*r + r*V умножений.
Берём S лучших по грубой оценке, для них точные оценки (S*1024). Итоговые логиты: точные у кандидатов,
грубые у остальных. Самое вероятное слово выбирается среди кандидатов по точной оценке.

Вариант 0: A, B из разложения словаря с учётом реальных состояний (train), без обучения.
Вариант 1: то же, затем обучение (h A) B повторять полный словарь (KL), градиенты вручную
(на этом macOS обратный проход torch на MPS падает).
Проверка на test, состояния для подбора и обучения только из train.
"""
import json
import sys
import time

import torch

from core import Qwen, rms, wiki_tokens, windows

DEV = "mps"
V, D = 151936, 1024
HEAD = V * D
TOTAL = 625e6
t0 = time.time()
torch.manual_seed(0)
m = Qwen(DEV)
W = m.head  # [D, V]


def states(wins):
    out = []
    for i in range(wins.shape[0]):
        _, hid = m.forward(wins[i:i + 1], return_hidden=True)
        out.append(rms(hid[-1][0], m.norm))
    return torch.cat(out)


test = windows(wiki_tokens("test").to(DEV), 8)
Ht = states(test)
Yt = test[:, 1:].reshape(-1)
pos = torch.cat([torch.arange(i * 512, i * 512 + 511) for i in range(test.shape[0])]).to(DEV)
Htr = states(windows(wiki_tokens("train").to(DEV), 64))
print(f"состояния: train {Htr.shape[0]}, test {Ht.shape[0]} ({time.time() - t0:.0f} с)")

full_top = torch.empty(Ht.shape[0], dtype=torch.long, device=DEV)
for a in range(0, Ht.shape[0], 512):
    full_top[a:a + 512] = (Ht[a:a + 512].clone() @ W).argmax(-1)


def evaluate(A, B, S, ch=128):
    agree, loss = 0.0, 0.0
    for a in range(0, Ht.shape[0], ch):
        h = Ht[a:a + ch].clone()
        ap = (h @ A) @ B
        cand = ap.topk(S, dim=-1).indices  # [n, S]
        ex = torch.bmm(m.emb[cand], h[:, :, None])[..., 0]  # точные оценки кандидатов
        best = cand.gather(1, ex.argmax(-1, keepdim=True))[:, 0]
        agree += (best == full_top[a:a + ch]).float().sum().item()
        mixed = ap.scatter(1, cand, ex)  # точные у кандидатов, грубые у остальных
        sel = (pos >= a) & (pos < a + ch)
        i = pos[sel] - a
        loss += (mixed.logsumexp(-1)[i] - mixed[i, Yt[sel]]).sum().item()
    cost = D * A.shape[1] + A.shape[1] * V + S * D
    return agree / Ht.shape[0], loss / len(pos), cost


# точная потеря полной модели на тех же позициях
bl = 0.0
for a in range(0, Ht.shape[0], 512):
    lg = Ht[a:a + 512].clone() @ W
    sel = (pos >= a) & (pos < a + 512)
    bl += (lg.logsumexp(-1)[pos[sel] - a] - lg[pos[sel] - a, Yt[sel]]).sum().item()
base_loss = bl / len(pos)
print(f"полный словарь: потеря {base_loss:.4f}")

# разложение с учётом состояний: M = C^{1/2} W, главные направления M M^T
C = (Htr.T.contiguous() @ Htr).cpu().double() / Htr.shape[0]
ev, U = torch.linalg.eigh(C)
ev = ev.clamp(min=ev.max().item() * 1e-9)
Ch, Cih = (U * ev.sqrt()) @ U.T, (U / ev.sqrt()) @ U.T
G = Ch @ (W.cpu().double() @ W.T.contiguous().cpu().double()) @ Ch
g, Ug = torch.linalg.eigh(G)
Ug = Ug.flip(1)

res = {"base_loss": base_loss, "runs": []}
SS = (256, 1024, 2048)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
RS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [32, 64, 128]
for r in RS:
    Ur = Ug[:, :r]
    A = (Cih @ Ur).float().to(DEV).contiguous()
    B = (Ur.T @ Ch @ W.cpu().double()).float().to(DEV).contiguous()
    for S in SS:
        ag, lo, cost = evaluate(A, B, S)
        res["runs"].append({"r": r, "S": S, "trained": False, "agree": ag, "loss": lo, "head_share": cost / HEAD})
        print(f"r={r:3d} S={S:4d} без обучения: топ-1 совпадает {100 * ag:5.1f}%, потеря {lo:.4f} (+{lo - base_loss:.4f}), "
              f"словарь {100 * cost / HEAD:4.1f}%, вся модель {100 * (TOTAL - HEAD + cost) / TOTAL:4.1f}%", flush=True)
    # обучение: KL(полный || грубый), Adam, градиенты вручную
    params = [A, B]
    mom = [torch.zeros_like(p) for p in params]
    vel = [torch.zeros_like(p) for p in params]
    lr, b1, b2 = 1e-3, 0.9, 0.999
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, Htr.shape[0], (512,), device=DEV)
        h = Htr[idx]
        with torch.no_grad():
            t = (h @ W).softmax(-1)
            z = h @ A
            lg = z @ B
            p = lg.softmax(-1)
            dlg = (p - t) / h.shape[0]
            gB = z.T.contiguous() @ dlg
            gA = h.T.contiguous() @ (dlg @ B.T.contiguous())
            for i, (prm, gr) in enumerate(zip(params, (gA, gB))):
                mom[i].mul_(b1).add_(gr, alpha=1 - b1)
                vel[i].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = mom[i] / (1 - b1 ** step)
                vh = vel[i] / (1 - b2 ** step)
                prm.sub_(lr * mh / (vh.sqrt() + 1e-8))
        if step % 500 == 0:
            kl = (t * (t.clamp_min(1e-12).log() - lg.log_softmax(-1))).sum(-1).mean().item()
            print(f"   шаг {step}: KL на батче train {kl:.3f} ({time.time() - t0:.0f} с)", flush=True)
    for S in SS:
        ag, lo, cost = evaluate(A, B, S)
        res["runs"].append({"r": r, "S": S, "trained": True, "agree": ag, "loss": lo, "head_share": cost / HEAD})
        print(f"r={r:3d} S={S:4d} обучен:      топ-1 совпадает {100 * ag:5.1f}%, потеря {lo:.4f} (+{lo - base_loss:.4f}), "
              f"словарь {100 * cost / HEAD:4.1f}%, вся модель {100 * (TOTAL - HEAD + cost) / TOTAL:4.1f}%", flush=True)
    torch.save({"A": A.cpu(), "B": B.cpu()}, f"results/vocab_r{r}.pt")
    json.dump(res, open(f"results/step3_vocab_{'_'.join(map(str, RS))}.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
