"""gemv с делением входа на S частей (split-K) для матриц с малым числом выходов (o, down: N=1024).
Каждая программа считает свой кусок входа для BN выходов и пишет частичную сумму в PART[s, n];
последняя закончившая программа данного блока выходов (счётчик-атомик) складывает части в фиксированном
порядке s=0..S-1 и прибавляет к Y — результат одинаков от запуска к запуску. Счётчик сбрасывается ей же.
MODE 1: Y += Wx; MODE 2: вход x_k = silu(X[k]) * X[K+k], Y += Wx."""
import torch
import triton
import triton.language as tl

CONF_SK = [triton.Config({"BN": bn, "BK": bk, "S": s}, num_warps=w)
           for bn in (8, 16, 32) for bk in (128, 256, 512) for s in (2, 4, 8) for w in (2, 4)]


@triton.autotune(configs=CONF_SK, key=["N", "K", "MODE"], restore_value=["Y"])
@triton.jit
def gemv_sk_k(X, W, Y, PART, CNT, N, K, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    KS = tl.cdiv(K, S)
    k_lo = s * KS
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, KS, BK):
        rk = k_lo + k0 + tl.arange(0, BK)
        mk = (rk < K) & (rk < k_lo + KS)
        if MODE == 2:
            g = tl.load(X + rk, mask=mk, other=0.0)
            u = tl.load(X + K + rk, mask=mk, other=0.0)
            x = g / (1.0 + tl.exp(-g)) * u
        else:
            x = tl.load(X + rk, mask=mk, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(PART + s * N + rn, acc, mask=rn < N)
    tl.debug_barrier()
    old = tl.atomic_add(CNT + pid, 1, sem="acq_rel")
    if old == S - 1:
        tot = tl.zeros((BN,), dtype=tl.float32)
        for j in range(S):
            tot += tl.load(PART + j * N + rn, mask=rn < N, other=0.0, cache_modifier=".cg")
        y = tl.load(Y + rn, mask=rn < N, other=0.0)
        tl.store(Y + rn, y + tot, mask=rn < N)
        tl.atomic_xchg(CNT + pid, 0)


class SplitK:
    """Буферы для частей и счётчиков (одни на все слои: ядра идут по очереди)."""

    def __init__(self, N=1024, smax=8, device="cuda"):
        self.part = torch.zeros(smax, N, device=device)
        self.cnt = torch.zeros(N, dtype=torch.int32, device=device)

    def __call__(self, x, W, y, mode):
        N, K = W.shape
        gemv_sk_k[lambda m: (triton.cdiv(N, m["BN"]), m["S"])](x, W, y, self.part, self.cnt, N, K, mode)
