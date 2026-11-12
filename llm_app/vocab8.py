"""Точный выбор слова по сжатому (int8) словарю — тот же ответ, что полный fp32-словарь, но читается 156 МБ вместо 311.

Строка словаря e_v ~ s_v * q_v (q — int8 по строке). Ошибка d_v = e_v - s_v q_v известна заранее, поэтому
|e_v.h - s_v q_v.h| <= B_v = min(||d_v||_2 ||h||_2, max|d_v| ||h||_1) + запас на округление fp32.
1) vocab_a_k: грубая оценка a_v = s_v q_v.h; пишет a_v + B_v и общий максимум a_v - B_v (атомик).
2) vocab_b_k: слова с a_v + B_v >= max(a - B) — кандидаты; для них точный счёт по bf16-строкам (маска: другие
   строки не читаются), лучший (значение, номер) — атомик над 64-битным ключом (при равенстве — меньший номер).
3) vocab_c_k: номер слова в выход, сброс накопителей.
Вариант для T слов: слова пачки идут по очереди внутри программы (веса из кэша)."""
import torch
import triton
import triton.language as tl

SLACK = tl.constexpr(2e-4)  # запас на округление fp32: относительно max|e_v| * ||h||_1 (с большим избытком)
CONF_A = [triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (16, 32, 64) for bk in (256, 512, 1024) for w in (4, 8)]
CONF_B = [triton.Config({"BS": bs}, num_warps=w) for bs in (128, 256, 512) for w in (4, 8)]


@triton.autotune(configs=CONF_A, key=["V", "K"])
@triton.jit
def vocab_a_k(H, Q, S, C1, C2, AM, HI, LO, T, V, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < V
    s = tl.load(S + rn, mask=mn, other=0.0)
    c1 = tl.load(C1 + rn, mask=mn, other=0.0)
    c2 = tl.load(C2 + rn, mask=mn, other=0.0)
    am = tl.load(AM + rn, mask=mn, other=0.0)
    for t in range(T):
        rk = tl.arange(0, KP)
        hv = tl.load(H + t * K + rk, mask=rk < K, other=0.0)
        n1 = tl.sum(tl.abs(hv), 0)
        n2 = tl.sqrt(tl.sum(hv * hv, 0))
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rb = k0 + tl.arange(0, BK)
            mb = rb < K
            x = tl.load(H + t * K + rb, mask=mb, other=0.0)
            q = tl.load(Q + rn[:, None] * K + rb[None, :], mask=mn[:, None] & mb[None, :], other=0).to(tl.float32)
            acc += tl.sum(q * x[None, :], axis=1)
        a = s * acc
        b = tl.minimum(c2 * n2, c1 * n1) + SLACK * am * n1
        tl.store(HI + t * V + rn, a + b, mask=mn)
        tl.atomic_max(LO + t, tl.max(tl.where(mn, a - b, -float("inf")), 0))


@triton.autotune(configs=CONF_B, key=["V", "K"])
@triton.jit
def vocab_b_k(H, E, HI, LO, BEST, T, V, K: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BS + tl.arange(0, BS)
    mn = rn < V
    for t in range(T):
        hi = tl.load(HI + t * V + rn, mask=mn, other=-float("inf"))
        c = hi >= tl.load(LO + t)
        if tl.sum(c.to(tl.int32), 0) > 0:
            acc = tl.zeros((BS,), dtype=tl.float32)
            for k0 in range(0, K, 64):
                rk = k0 + tl.arange(0, 64)
                x = tl.load(H + t * K + rk)
                w = tl.load(E + rn[:, None] * K + rk[None, :], mask=c[:, None], other=0.0).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)
            bits = acc.to(tl.int32, bitcast=True)
            key32 = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)  # порядок как у чисел fp32
            key = (key32.to(tl.int64) << 32) | (4294967295 - rn.to(tl.int64))
            key = tl.where(c, key, -9223372036854775807 - 1)
            tl.atomic_max(BEST + t, tl.max(key, 0))


@triton.jit
def vocab_c_k(BEST, LO, OUT, NCAND):
    t = tl.program_id(0)
    key = tl.load(BEST + t)
    tl.store(OUT + t, 4294967295 - (key & 4294967295))
    tl.store(BEST + t, -9223372036854775807 - 1)
    tl.store(LO + t, -float("inf"))


class Vocab8:
    def __init__(self, emb16, TM=16):
        V, K = emb16.shape
        self.V, self.K, self.E = V, K, emb16
        self.Q = torch.empty(V, K, dtype=torch.int8, device="cuda")
        self.S, self.C1, self.C2, self.AM = (torch.empty(V, device="cuda") for _ in range(4))
        for i in range(0, V, 8192):
            e = emb16[i:i + 8192].double()
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
        self.BEST = torch.full((TM,), -2 ** 63, dtype=torch.int64, device="cuda")
        self.out = torch.zeros(TM, dtype=torch.int64, device="cuda")

    def __call__(self, h, T):
        V, K = self.V, self.K
        vocab_a_k[lambda m: (triton.cdiv(V, m["BN"]),)](h, self.Q, self.S, self.C1, self.C2, self.AM, self.HI, self.LO, T, V, K, KP=triton.next_power_of_2(K))
        vocab_b_k[lambda m: (triton.cdiv(V, m["BS"]),)](h, self.E, self.HI, self.LO, self.BEST, T, V, K)
        vocab_c_k[(T,)](self.BEST, self.LO, self.out, 0)
        return self.out[:T]
