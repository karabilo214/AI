"""Шаг 2е. Точный поиск самого вероятного слова без полного словаря (гарантия совпадения топ-1).

K частых слов считаются точно -> лучшая оценка b.
Остальные слова разбиты на группы (k-means), у группы центр c и радиус r = max |e - c|.
Верхняя граница оценки любого слова группы: h·c + |h|·r. Группу считаем точно, только если граница > b
(группы по убыванию границы, b растёт). Ответ всегда совпадает с полным argmax.
Сколько слов придётся открыть, оценивается с двух сторон:
  не меньше — группы с границей выше настоящего максимума,
  не больше — группы с границей выше лучшего из частого списка.
"""
import time

import torch

from core import Qwen, rms, wiki_tokens, windows

DEV = "mps"
V, D = 151936, 1024
torch.manual_seed(0)
t0 = time.time()
m = Qwen(DEV)
test = windows(wiki_tokens("test").to(DEV), 8)
hs = []
for i in range(test.shape[0]):
    _, hid = m.forward(test[i:i + 1], return_hidden=True)
    hs.append(rms(hid[-1][0], m.norm))
Hs = torch.cat(hs)
N = Hs.shape[0]
E = m.emb
true_max = torch.cat([(Hs[a:a + 512].clone() @ m.head).max(-1).values for a in range(0, N, 512)])
freq = torch.bincount(wiki_tokens("train"), minlength=V).to(DEV)
order = freq.argsort(descending=True)
hn = Hs.norm(dim=-1)

for K in (2048, 8192, 32768):
    b_list = (Hs @ E[order[:K]].T.contiguous()).max(-1).values
    rest = order[K:]
    Er = E[rest].contiguous()
    for C in (1024, 4096, 16384):
        cen = Er[torch.randperm(len(rest), device=DEV)[:C]].clone()
        for it in range(15):
            cT, half = cen.T.contiguous(), (cen * cen).sum(-1) / 2
            assign = torch.cat([(Er[a:a + 16384].clone() @ cT - half).argmax(-1) for a in range(0, len(rest), 16384)])
            cnt = torch.bincount(assign, minlength=C).float()
            new = torch.zeros_like(cen).index_add_(0, assign, Er)
            nz = cnt > 0
            cen[nz] = new[nz] / cnt[nz, None]
        rad = torch.zeros(C, device=DEV).scatter_reduce_(0, assign, (Er - cen[assign]).norm(dim=-1), "amax")
        bound = Hs @ cen.T.contiguous() + hn[:, None] * rad[None]
        # проверка границы: оценка любого слова не выше границы его группы (на 256 словах текста)
        lg = Hs[:256] @ Er.T.contiguous()
        assert (lg <= bound[:256][:, assign] + 1e-3).all().item(), "граница нарушена"
        lo = ((bound > true_max[:, None]).float() @ cnt)
        hi = ((bound > torch.maximum(b_list, torch.zeros_like(b_list) - 1e9)[:, None]).float() @ cnt)
        need_none = (true_max <= b_list).float().mean().item()  # частый список уже содержит максимум
        print(f"K={K:5d} групп={C:5d}: открыть слов не меньше {lo.mean():7.0f}, не больше {hi.mean():7.0f} (среднее); "
              f"доля словаря по умножениям {100 * (K + C + lo.mean().item()) / V:5.1f}%–{100 * (K + C + hi.mean().item()) / V:5.1f}%; "
              f"максимум в частом списке у {100 * need_none:.1f}% слов", flush=True)
print(f"время {time.time() - t0:.0f} с")
