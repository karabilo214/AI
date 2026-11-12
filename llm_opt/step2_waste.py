"""Шаг 2. Где модель делает лишнюю работу. Всё на тестовой части wikitext-2.

Части (запуск: python step2_waste.py <часть>):
  mlp    — сколько нейронов MLP реально работает на каждый токен; качество, если оставить только самые сильные
  layers — что будет, если убрать один слой внимания / MLP; на каком слое ответ уже ясен
  heads  — на что смотрят головы внимания и что будет, если убрать одну голову
"""
import json
import os
import sys
import time

import torch

from core import FF, H, L, Qwen, loss_of, wiki_tokens, windows

DEV = "mps"
os.makedirs("results", exist_ok=True)


def save(name, obj):
    json.dump(obj, open(f"results/step2_{name}.json", "w"), indent=1)


def evaluate(m, wins, opts=None):
    ls, ac = [], []
    for i in range(wins.shape[0]):
        x = wins[i:i + 1]
        lo, a = loss_of(m.forward(x, opts), x)
        ls.append(lo); ac.append(a)
    return sum(ls) / len(ls), sum(ac) / len(ac)


def part_mlp(m, wins):
    dn = [w["down"].norm(dim=1) for w in m.layers]  # длина вклада каждого нейрона (строка W_down^T)
    base = evaluate(m, wins)
    print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}")

    # 1) статистика вкладов |h_i|*|down_i|
    cover = {p: torch.zeros(L) for p in (0.5, 0.9, 0.99)}
    freq = torch.zeros(L, FF, device=DEV)  # как часто нейрон в топ-10% токена
    jacc_next = torch.zeros(L)
    ntok = 0

    def rec(name, l, h):
        nonlocal ntok
        if name != "mlp_h":
            return
        c = (h[0].abs() * dn[l]).float()  # [T, FF]
        cs = c.sort(dim=-1, descending=True).values.cumsum(-1)
        cs = cs / cs[:, -1:]
        for p in cover:
            cover[p][l] += ((cs < p).sum(-1) + 1).float().sum().cpu()
        k = FF // 10
        top = torch.zeros_like(c, dtype=torch.bool).scatter_(1, c.topk(k, dim=-1).indices, True)
        freq[l] += top.float().sum(0)
        inter = (top[1:] & top[:-1]).sum(-1).float()
        jacc_next[l] += (inter / (2 * k - inter)).sum().cpu()
        if l == L - 1:
            ntok += c.shape[0]

    for i in range(wins.shape[0]):
        m.forward(wins[i:i + 1], {"rec": rec})
    T = wins.shape[1]
    res = {"base": base, "per_layer": []}
    for l in range(L):
        f = freq[l] / ntok
        # какая доля нейронов покрывает 90% всех попаданий в топ-10% (постоянный набор или разный?)
        fs = f.sort(descending=True).values
        need = int(((fs.cumsum(0) / fs.sum()) < 0.9).float().sum().item()) + 1
        res["per_layer"].append({
            "layer": l,
            "neurons_for_50%": (cover[0.5][l] / ntok).item(),
            "neurons_for_90%": (cover[0.9][l] / ntok).item(),
            "neurons_for_99%": (cover[0.99][l] / ntok).item(),
            "jaccard_top10_next_token": (jacc_next[l] / (ntok - wins.shape[0])).item(),
            "never_in_top10": int((f == 0).float().sum().item()),
            "share_of_neurons_holding_90%_of_top10_hits": need / FF,
        })
    print("слой | нейронов из 3072 для 50/90/99% вклада | совпадение топ-10% у соседних токенов | доля нейронов, дающих 90% попаданий в топ-10%")
    for r in res["per_layer"]:
        print(f"{r['layer']:4d} | {r['neurons_for_50%']:6.0f} {r['neurons_for_90%']:6.0f} {r['neurons_for_99%']:6.0f} "
              f"| {r['jaccard_top10_next_token']:.2f} | {r['share_of_neurons_holding_90%_of_top10_hits']:.2f}")

    # 2) оракул: оставить на каждом токене только k% самых сильных нейронов (во всех слоях)
    res["oracle_topk"] = {}
    for frac in (0.5, 0.25, 0.1, 0.05, 0.02):
        k = int(FF * frac)

        def mask(l, h, k=k):
            c = h.abs() * dn[l]
            idx = c.topk(k, dim=-1).indices
            return torch.zeros_like(h).scatter_(-1, idx, h.gather(-1, idx))
        r = evaluate(m, wins, {"mlp_mask": mask})
        res["oracle_topk"][frac] = r
        print(f"оставить {100 * frac:4.0f}% нейронов MLP (оракул по токену): потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), топ-1 {r[1]:.4f}")

    # 3) постоянный набор: k% нейронов, чаще всего попадающих в топ-10% (без выбора по токену)
    res["static_topk"] = {}
    for frac in (0.5, 0.25):
        k = int(FF * frac)
        keep = [torch.zeros(FF, device=DEV).scatter_(0, freq[l].topk(k).indices, 1.0) for l in range(L)]
        r = evaluate(m, wins, {"mlp_mask": lambda l, h: h * keep[l]})
        res["static_topk"][frac] = r
        print(f"постоянный набор {100 * frac:3.0f}% нейронов (один на все токены): потеря {r[0]:.4f} (+{r[0] - base[0]:.4f}), топ-1 {r[1]:.4f}")
    save("mlp", res)


def part_layers(m, wins):
    base = evaluate(m, wins)
    print(f"исходная: потеря {base[0]:.4f}, топ-1 {base[1]:.4f}")
    res = {"base": base, "skip": {}}
    for kind in ("attn", "mlp"):
        row = []
        for l in range(L):
            r = evaluate(m, wins, {"skip": {(kind, l)}})
            res["skip"][f"{kind}{l}"] = r
            row.append(r[0] - base[0])
        print(f"убрать {kind:4s} слоя 0..27, прирост потери:", " ".join(f"{d:.2f}" for d in row))

    # логит-линза: после какого слоя топ-1 уже совпадает с итоговым
    agree = torch.zeros(L + 1)
    settle = []  # слой, с которого ответ больше не меняется
    conf_final = []
    for i in range(wins.shape[0]):
        x = wins[i:i + 1]
        logits, hid = m.forward(x, return_hidden=True)
        fin = logits[0].argmax(-1)
        conf_final.append(logits[0].float().softmax(-1).max(-1).values.cpu())
        same = []
        for l, hl in enumerate(hid):
            s = (m.logits_from_hidden(hl)[0].argmax(-1) == fin).cpu()
            agree[l] += s.float().sum()
            same.append(s)
        same = torch.stack(same)  # [L+1, T]
        # первый слой, начиная с которого все дальнейшие совпадают
        tail = same.flip(0).cumprod(0).flip(0)
        settle.append(tail.float().argmax(0) + (1 - tail[-1].float()) * 0)  # tail[-1] всегда 1
    ntok = wins.numel()
    settle = torch.cat(settle); conf = torch.cat(conf_final)
    res["lens_agree"] = (agree / ntok).tolist()
    print("доля токенов, где после слоя l ответ уже итоговый:",
          " ".join(f"{l}:{a:.2f}" for l, a in enumerate(res["lens_agree"]) if l % 2 == 0 or l == L))
    for lo, hi in [(0.9, 1.01), (0.5, 0.9), (0.0, 0.5)]:
        sel = (conf >= lo) & (conf < hi)
        s = settle[sel]
        q = [s.quantile(p).item() for p in (0.5, 0.9)]
        res[f"settle_conf_{lo}-{hi}"] = {"share_tokens": sel.float().mean().item(), "median": q[0], "p90": q[1],
                                         "hist": torch.bincount(s.long(), minlength=L + 1).tolist()}
        print(f"уверенность модели {lo}-{min(hi, 1)}: {100 * sel.float().mean():.0f}% токенов, ответ закрепляется "
              f"на слое: медиана {q[0]:.0f}, 90% токенов к слою {q[1]:.0f}")
    save("layers", res)


def part_heads(m, wins_stat, wins_abl):
    stats = {k: torch.zeros(L, H) for k in ("first", "prev", "self", "entropy")}
    n = 0

    def rec(name, l, a):
        nonlocal n
        if name != "attn_w":
            return
        a = a[0].float()  # [H, T, T]
        T = a.shape[-1]
        stats["first"][l] += a[:, 1:, 0].sum(-1).cpu()
        stats["prev"][l] += a[:, 1:, :].diagonal(offset=0, dim1=1, dim2=2).sum(-1).cpu()  # a[t+1, t]
        stats["self"][l] += a.diagonal(dim1=1, dim2=2)[:, 1:].sum(-1).cpu()
        stats["entropy"][l] += (-(a * (a + 1e-12).log()).sum(-1))[:, 1:].sum(-1).cpu()
        if l == 0:
            n += T - 1
    for i in range(wins_stat.shape[0]):
        m.forward(wins_stat[i:i + 1], {"rec": rec})
    for k in stats:
        stats[k] /= n
    base = evaluate(m, wins_abl)
    delta = torch.zeros(L, H)
    t0 = time.time()
    for l in range(L):
        for h in range(H):
            mult = torch.ones(H, device=DEV); mult[h] = 0
            delta[l, h] = evaluate(m, wins_abl, {"head_mult": {l: mult}})[0] - base[0]
        print(f"слой {l}: прирост потери при удалении головы: " + " ".join(f"{d:.3f}" for d in delta[l]),
              f"({time.time() - t0:.0f} с)", flush=True)
    res = {"base": base, "delta": delta.tolist(), **{k: v.tolist() for k, v in stats.items()}}
    d = delta.flatten()
    for thr in (0.001, 0.005, 0.01, 0.05):
        print(f"голов с приростом потери < {thr}: {(d < thr).sum().item()} из {L * H}")
    print(f"голов, где >50% внимания на первый токен: {(stats['first'] > 0.5).sum().item()}, "
          f">50% на предыдущий: {(stats['prev'] > 0.5).sum().item()}, >50% на себя: {(stats['self'] > 0.5).sum().item()}")
    save("heads", res)


if __name__ == "__main__":
    part = sys.argv[1]
    m = Qwen(DEV)
    ids = wiki_tokens("test").to(DEV)
    t0 = time.time()
    if part == "mlp":
        part_mlp(m, windows(ids, 8))
    elif part == "layers":
        part_layers(m, windows(ids, 8))
    elif part == "heads":
        part_heads(m, windows(ids, 8), windows(ids, 4))
    print(f"время {time.time() - t0:.0f} с")
