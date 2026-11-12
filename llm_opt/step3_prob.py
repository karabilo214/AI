"""Шаг 3г. Вероятности при угадывателе словаря: что ставить словам вне кандидатов.

Точные оценки у S кандидатов, у остальных — грубые (h A) B. Варианты для остальных:
  raw     — грубые как есть (было: потеря растёт);
  shift   — a*грубая + b, a и b подобраны по потере на train (сетка);
  const   — одна оценка log(остаточная масса) на всех, масса подобрана на train;
  only    — только кандидаты (если следующее слово не среди них — потеря бесконечна, считаем отдельно).
Проверка на test.
"""
import json

import torch

from core import Qwen, rms, wiki_tokens, windows

DEV = "mps"
m = Qwen(DEV)
ab = torch.load("results/vocab_r64.pt")
A, B = ab["A"].to(DEV).contiguous(), ab["B"].to(DEV).contiguous()


def collect(split, n):
    wins = windows(wiki_tokens(split).to(DEV), n)
    hs, ys = [], []
    for i in range(n):
        _, hid = m.forward(wins[i:i + 1], return_hidden=True)
        hs.append(rms(hid[-1][0], m.norm)[:-1].clone())
        ys.append(wins[i, 1:])
    return torch.cat(hs), torch.cat(ys)


Htr, Ytr = collect("train", 8)
Hte, Yte = collect("test", 8)


def parts(H, Y, S, ch=128):
    """Для каждой позиции: точные логиты кандидатов, грубые логиты остальных, есть ли цель среди кандидатов."""
    out = []
    for a in range(0, H.shape[0], ch):
        h, y = H[a:a + ch].clone(), Y[a:a + ch]
        ap = (h @ A) @ B
        cand = ap.topk(S, dim=-1).indices
        ex = torch.bmm(m.emb[cand], h[:, :, None])[..., 0]
        mask = torch.zeros_like(ap, dtype=torch.bool).scatter_(1, cand, True)
        y_in = mask.gather(1, y[:, None])[:, 0]
        y_ex = (h * m.emb[y]).sum(-1)  # точная оценка цели (нужна только если она среди кандидатов)
        y_ap = ap.gather(1, y[:, None])[:, 0]
        rest = ap.masked_fill(mask, float("-inf"))
        out.append((ex, rest, y_in, y_ex, y_ap))
    return out


def loss_of(pp, mode, a=1.0, b=0.0):
    tot, n = 0.0, 0
    for ex, rest, y_in, y_ex, y_ap in pp:
        lse_c = ex.logsumexp(-1)
        if mode == "only":
            l = lse_c - y_ex
            tot += l[y_in].sum().item(); n += y_in.float().sum().item()
            continue
        if mode == "const":
            lse_r = torch.full_like(lse_c, b)  # b = log остаточной массы относительно... (абсолютная оценка)
            y_r = b - torch.log(torch.tensor(float(rest.shape[1] - ex.shape[1]), device=DEV))
        else:
            lse_r = (a * rest + b).logsumexp(-1)
            y_r = a * y_ap + b
        lse = torch.logaddexp(lse_c, lse_r)
        y = torch.where(y_in, y_ex, y_r)
        tot += (lse - y).sum().item(); n += len(y)
    return tot / n


# точная потеря полной модели
full = 0.0
for a_ in range(0, Hte.shape[0], 256):
    lg = Hte[a_:a_ + 256].clone() @ m.head
    full += (lg.logsumexp(-1) - lg.gather(1, Yte[a_:a_ + 256, None])[:, 0]).sum().item()
full /= Hte.shape[0]
print(f"полный словарь, test: потеря {full:.4f}")
res = {"full": full}
for S in (1024, 2048):
    ptr, pte = parts(Htr, Ytr, S), parts(Hte, Yte, S)
    r = {"raw": loss_of(pte, "raw")}
    best = min(((loss_of(ptr, "shift", a, b), a, b) for a in (0.6, 0.8, 0.9, 1.0, 1.1) for b in (-6, -4, -3, -2, -1, 0)))
    _, a, b = best
    r["shift"] = loss_of(pte, "shift", a, b); r["shift_ab"] = (a, b)
    # const: оценка логсумэксп остальных = lse_кандидатов + c; подбираем c через b относительно lse_c
    def loss_const(pp, c):
        tot, n = 0.0, 0
        for ex, rest, y_in, y_ex, y_ap in pp:
            lse_c = ex.logsumexp(-1)
            lse_r = lse_c + c
            nrest = rest.shape[1] - ex.shape[1]
            lse = torch.logaddexp(lse_c, lse_r)
            y = torch.where(y_in, y_ex, lse_r - torch.log(torch.tensor(float(nrest), device=DEV)))
            tot += (lse - y).sum().item(); n += len(y)
        return tot / n
    cbest = min((loss_const(ptr, c), c) for c in (-8, -6, -5, -4, -3, -2))
    r["const"] = loss_const(pte, cbest[1]); r["const_c"] = cbest[1]
    r["only_loss_where_in"] = loss_of(pte, "only")
    r["target_in_cand"] = torch.cat([p[2] for p in pte]).float().mean().item()
    res[S] = r
    print(f"S={S}: грубые как есть {r['raw']:.4f} (+{r['raw'] - full:.4f}) | масштаб и сдвиг a={a} b={b}: {r['shift']:.4f} "
          f"(+{r['shift'] - full:.4f}) | одна оценка на всех c={cbest[1]}: {r['const']:.4f} (+{r['const'] - full:.4f}) | "
          f"цель среди кандидатов {100 * r['target_in_cand']:.1f}%")
json.dump(res, open("results/step3_prob.json", "w"), indent=1)

# ---------- разбор: откуда прирост потери ----------
print("\nразбор на test:")
for S in (1024, 2048):
    pte = parts(Hte, Yte, S)
    yin = torch.cat([p[2] for p in pte])
    mine, fl, cover = [], [], []
    for (ex, rest, y_in, y_ex, y_ap), a_ in zip(pte, range(0, Hte.shape[0], 128)):
        lse = torch.logaddexp(ex.logsumexp(-1), rest.logsumexp(-1))
        mine.append(lse - torch.where(y_in, y_ex, y_ap))
        lg = Hte[a_:a_ + 128].clone() @ m.head
        flse = lg.logsumexp(-1)
        fl.append(flse - lg.gather(1, Yte[a_:a_ + 128, None])[:, 0])
        cover.append((ex.logsumexp(-1) - flse).exp())  # доля вероятности полной модели у кандидатов
    mine, fl, cover = torch.cat(mine), torch.cat(fl), torch.cat(cover)
    n = len(mine)
    d_in = (mine[yin] - fl[yin]).sum().item() / n
    d_out = (mine[~yin] - fl[~yin]).sum().item() / n
    print(f"  S={S}: цель среди кандидатов ({100 * yin.float().mean():.1f}% слов) даёт +{d_in:.4f}, "
          f"вне кандидатов ({100 * (~yin).float().mean():.1f}%) даёт +{d_out:.4f}; "
          f"на словах вне кандидатов потеря полной модели {fl[~yin].mean():.2f}, у нас {mine[~yin].mean():.2f}; "
          f"кандидаты покрывают вероятности полной модели: медиана {100 * cover.median():.1f}%, среднее {100 * cover.mean():.1f}%")
