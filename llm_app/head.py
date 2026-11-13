"""Выбор следующего слова, точный как у полного fp32-словаря, без bf16-словаря в видеокарте.

1) Видеокарта: грубые оценки по int8-копии словаря + консервативная оценка ошибки (vocab8.vocab_a_k) -> для каждой
   строки верх оценки HI[v] и лучший низ LO.
2) Для greedy кандидаты HI[v] >= LO; при sampling пересчитывается полный словарь.
3) Процессор: точные логиты кандидатов по bf16-строкам словаря (в RAM), счёт fp32.
"""
import math

import numba as nb
import numpy as np
import torch
import triton

from .cpu_layer import bf
from .vocab8 import vocab_a_k


@nb.njit(parallel=True, fastmath=False, cache=True, nogil=True)
def exact_rows(W, toks, rows, H, out):
    for i in nb.prange(toks.shape[0]):
        s = np.float32(0.0)
        t, r = toks[i], rows[i]
        for d in range(H.shape[1]):
            s += bf(W[t, d]) * H[r, d]
        out[i] = s


class Head:
    def __init__(self, Wu16, TM=16):
        """Wu16 [V, D]: bf16-биты (uint16), обычно memmap файла модели — точные строки читаются оттуда."""
        V, D = Wu16.shape
        self.V, self.D = V, D
        self.Wc = np.asarray(Wu16)
        self.Q = torch.empty(V, D, dtype=torch.int8, device="cuda")
        self.S, self.C1, self.C2, self.AM = (torch.empty(V, device="cuda") for _ in range(4))
        for i in range(0, V, 8192):
            e = torch.from_numpy(np.ascontiguousarray(Wu16[i:i + 8192]).view(np.int16)).view(torch.bfloat16).to("cuda").double()
            am = e.abs().amax(1)
            s = (am / 127).float().double()
            s[s == 0] = 1.0
            q = torch.round(e / s[:, None]).clamp(-127, 127)
            d = e - q * s[:, None]
            self.Q[i:i + 8192] = q.to(torch.int8)
            self.S[i:i + 8192] = s.float()
            self.C1[i:i + 8192] = (d.abs().amax(1) * (1 + 1e-6)).float()
            self.C2[i:i + 8192] = (d.norm(dim=1) * (1 + 1e-6)).float()
            self.AM[i:i + 8192] = am.float()
        self.HI = torch.empty(TM, V, device="cuda")
        self.LO = torch.full((TM,), -float("inf"), device="cuda")

    def bounds(self, h, T):
        """Запуск на видеокарте (годится для графа): HI[:T], LO[:T]."""
        self.LO[:T].fill_(-float("inf"))
        V, K = self.V, self.D
        vocab_a_k[lambda m: (triton.cdiv(V, m["BN"]),)](h, self.Q, self.S, self.C1, self.C2, self.AM, self.HI, self.LO,
                                                        T, V, K, KP=triton.next_power_of_2(K))

    def pick(self, h, T, temperature=0.0, top_p=1.0, rng=None, run_bounds=True):
        """Слова для T строк h [T, D] (fp32 на видеокарте). temperature=0 — самое вероятное."""
        if run_bounds:
            self.bounds(h, T)
        delta = 0.0 if temperature <= 0 else 20.0 * temperature
        # Sampling использует весь словарь: отсечение по delta меняет распределение.
        mask = (torch.ones_like(self.HI[:T], dtype=torch.bool) if temperature > 0
                else self.HI[:T] >= self.LO[:T, None])
        rows, toks = mask.nonzero(as_tuple=True)
        rows, toks = rows.cpu().numpy(), toks.cpu().numpy()
        H = h[:T].float().cpu().numpy()
        lg = np.empty(len(toks), np.float32)
        exact_rows(self.Wc, toks, rows, H, lg)
        out = []
        for t in range(T):
            sel = rows == t
            c, l = toks[sel], lg[sel].astype(np.float64)
            if temperature <= 0:
                m = l.max()
                out.append(int(c[l == m].min()))
                continue
            z = (l - l.max()) / temperature
            p = np.exp(z)
            p /= p.sum()
            o = np.argsort(-p, kind="stable")
            cs = np.cumsum(p[o])
            keep = o[:int(np.searchsorted(cs, top_p) + 1)]
            pk = p[keep] / p[keep].sum()
            out.append(int(c[keep[(rng or np.random.default_rng()).choice(len(keep), p=pk)]]))
        return out
