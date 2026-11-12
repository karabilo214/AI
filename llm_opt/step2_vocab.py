"""Шаг 2д. Словарь (25% умножений на слово): считать только нужную часть.

Полный словарь: 151 936 слов x 1024 = 155.6 млн умножений на слово.
1) концентрация: сколько слов покрывают 90/99% вероятности;
2) постоянный список K самых частых слов (частоты из train);
3) двухступенчато: группы похожих слов (k-means по векторам словаря, без обучения),
   на слове сначала оценки групп по центрам, затем точный расчёт для слов из m лучших групп,
   остальным — оценка их группы.
Сравнение с полным словарём на test.
"""
import json
import time

import torch

from core import Qwen, rms, wiki_tokens, windows

DEV = "mps"
V, D = 151936, 1024
FULL = V * D
t0 = time.time()
torch.manual_seed(0)
m = Qwen(DEV)
test = windows(wiki_tokens("test").to(DEV), 8)

# конечные состояния слов (после последней нормировки)
hs = []
for i in range(test.shape[0]):
    _, hid = m.forward(test[i:i + 1], return_hidden=True)
    hs.append(rms(hid[-1][0], m.norm))
Hs = torch.cat(hs)  # [N, D]
Y = test[:, 1:].reshape(-1)
idx_pred = torch.cat([torch.arange(i * 512, i * 512 + 511) for i in range(test.shape[0])]).to(DEV)  # позиции, где есть следующее слово
N = Hs.shape[0]
E = m.emb  # [V, D]


def chunks(n, s=512):
    for a in range(0, n, s):
        yield a, min(n, a + s)


# полные логиты: топ-1, логсумэксп, вероятность цели
full_top, full_lse = torch.empty(N, dtype=torch.long, device=DEV), torch.empty(N, device=DEV)
need90, need99 = [], []
for a, b in chunks(N):
    lg = Hs[a:b].clone() @ m.head
    full_top[a:b] = lg.argmax(-1)
    full_lse[a:b] = lg.logsumexp(-1)
    p = lg.softmax(-1).sort(-1, descending=True).values.cumsum(-1)
    need90.append((p < 0.9).float().sum(-1) + 1)
    need99.append((p < 0.99).float().sum(-1) + 1)
need90, need99 = torch.cat(need90).cpu(), torch.cat(need99).cpu()
logp_target = ((Hs[idx_pred] * E[Y]).sum(-1) - full_lse[idx_pred])
base_loss = -logp_target.mean().item()
base_acc = (full_top[idx_pred] == Y).float().mean().item()
print(f"полный словарь: потеря {base_loss:.4f}, топ-1 {base_acc:.4f}")
print(f"слов для 90% вероятности: медиана {need90.median():.0f}, 90-й перцентиль {need90.quantile(0.9):.0f}; "
      f"для 99%: медиана {need99.median():.0f}, 90-й перцентиль {need99.quantile(0.9):.0f}")
res = {"base": [base_loss, base_acc], "need90_median": need90.median().item(), "need99_median": need99.median().item(),
       "static": {}, "two_level": {}}


def score(lg_approx):
    """lg_approx: функция (a, b) -> приближённые логиты [b-a, V]. Сравнение с полным."""
    agree, mass, lse = 0.0, 0.0, torch.empty(N, device=DEV)
    for a, b in chunks(N):
        la = lg_approx(a, b)
        agree += (la.argmax(-1) == full_top[a:b]).float().sum().item()
        lse[a:b] = la.logsumexp(-1)
    return agree / N, lse


# ---------- 2. постоянный список частых слов ----------
tr = wiki_tokens("train")
freq = torch.bincount(tr, minlength=V).float().to(DEV)
order = freq.argsort(descending=True)
for K in (2048, 8192, 32768):
    keep = torch.zeros(V, dtype=torch.bool, device=DEV); keep[order[:K]] = True
    neg = torch.where(keep, 0.0, float("-inf"))
    agree, lse = score(lambda a, b: Hs[a:b].clone() @ m.head + neg)
    covered = keep[Y].float().mean().item()
    # потеря только там, где цель в списке (иначе бесконечность)
    ok = keep[Y]
    lt = (Hs[idx_pred] * E[Y]).sum(-1) - lse[idx_pred]
    res["static"][K] = {"agree_top1": agree, "target_in_list": covered, "loss_where_in_list": -lt[ok].mean().item(),
                        "base_loss_same_tokens": -logp_target[ok].mean().item(), "macs_share": K / V}
    print(f"список {K:6d} частых слов ({100 * K / V:4.1f}% умножений): топ-1 совпадает {100 * agree:5.1f}%, "
          f"следующее слово в списке {100 * covered:5.1f}% (вне списка — потеря бесконечна)")

# ---------- 3. двухступенчато по группам ----------
for C in (1024, 4096):
    cen = E[torch.randperm(V, device=DEV)[:C]].clone()
    for it in range(15):
        assign = torch.empty(V, dtype=torch.long, device=DEV)
        cT = cen.T.contiguous()
        half = (cen * cen).sum(-1) / 2
        for a, b in chunks(V, 16384):
            assign[a:b] = (E[a:b].clone() @ cT - half).argmax(-1)
        cnt = torch.bincount(assign, minlength=C).float()
        new = torch.zeros_like(cen).index_add_(0, assign, E)
        nz = cnt > 0
        cen[nz] = new[nz] / cnt[nz, None]
        empty = (~nz).float().sum().item()
        if empty:  # пустые группы — заново из случайных слов
            ne = int(empty)
            cen[~nz] = E[torch.randint(0, V, (ne,), device=DEV)]
    sizes = cnt
    print(f"\n{C} групп: размер медиана {sizes.median():.0f}, максимум {sizes.max():.0f}, пустых {empty:.0f}")
    cT = cen.T.contiguous()
    lg_c_tok_idx = assign  # группа каждого слова
    for mm in (4, 16, 64, 256):
        if mm >= C:
            continue
        computed = torch.zeros(N, device=DEV)

        def approx(a, b, mm=mm):
            sc = Hs[a:b].clone() @ cT  # [n, C] оценки групп
            top = sc.topk(mm, dim=-1).indices
            sel = torch.zeros(b - a, C, dtype=torch.bool, device=DEV).scatter_(1, top, True)
            tok_sel = sel[:, lg_c_tok_idx]  # [n, V]
            computed[a:b] = (sizes[top]).sum(-1)
            exact = Hs[a:b].clone() @ m.head
            return torch.where(tok_sel, exact, sc[:, lg_c_tok_idx])
        agree, lse = score(approx)
        lt = (Hs[idx_pred] * E[Y]).sum(-1)
        # цель: точная оценка, если её группа выбрана, иначе оценка группы
        sc_t = (Hs[idx_pred] * cen[assign[Y]]).sum(-1)
        sel_t = torch.zeros(len(Y), dtype=torch.bool, device=DEV)
        for a, b in chunks(len(Y)):
            top = (Hs[idx_pred[a:b]] @ cT).topk(mm, dim=-1).indices
            sel_t[a:b] = (top == assign[Y[a:b], None]).any(-1)
        lt = torch.where(sel_t, lt, sc_t) - lse[idx_pred]
        loss = -lt.mean().item()
        macs = (C + computed.mean().item()) * D
        res["two_level"][f"{C}_{mm}"] = {"agree_top1": agree, "target_group_selected": sel_t.float().mean().item(),
                                        "loss": loss, "macs_share": macs / FULL}
        print(f"  {mm:3d} лучших групп: умножений {100 * macs / FULL:5.1f}%, топ-1 совпадает {100 * agree:5.1f}%, "
              f"группа следующего слова выбрана {100 * sel_t.float().mean():5.1f}%, потеря {loss:.4f} (+{loss - base_loss:.4f})")
json.dump(res, open("results/step2_vocab.json", "w"), indent=1)
print(f"время {time.time() - t0:.0f} с")
