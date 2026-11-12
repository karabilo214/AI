"""Ядра Triton: умножение на веса bf16 со счётом в fp32 (вес переводится в fp32 при загрузке, накопление fp32).

W хранится как [N, K] (строка = выход, непрерывна по входу K) — чтение весов сплошными кусками.
gemv1: одно слово (T=1). gemvT: до 16 слов, tl.dot в точном режиме (ieee, без TF32).
"""
import torch
import triton
import triton.language as tl

CONF1 = [triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (4, 8, 16, 32) for bk in (256, 512, 1024) for w in (2, 4, 8)]


@triton.autotune(configs=CONF1, key=["N", "K"])
@triton.jit
def _gemv1(X, W, Y, N, K, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    t = tl.program_id(1)  # слово пачки: веса одной матрицы после первого слова берутся из кэша L2
    X += t * K
    Y += t * N
    rn = pid * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        x = tl.load(X + rk, mask=mk, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + rn, acc, mask=rn < N)


CONFT = [triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (16, 32, 64) for bk in (64, 128, 256) for w in (2, 4, 8)]


@triton.autotune(configs=CONFT, key=["N", "K"])
@triton.jit
def _gemvT(X, W, Y, T, N, K, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, 16)
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & mk[None, :], other=0.0).to(tl.float32)
        xt = tl.load(X + rt[None, :] * K + rk[:, None], mask=(rt[None, :] < T) & mk[:, None], other=0.0)  # [BK, 16]
        acc += tl.dot(w, xt, input_precision="ieee")
    tl.store(Y + rt[None, :] * N + rn[:, None], acc, mask=(rn[:, None] < N) & (rt[None, :] < T))


def mm16(x, W):
    """x [T, K] fp32 (T <= 16), W [N, K] bf16 -> [T, N] fp32."""
    T, K = x.shape
    N = W.shape[0]
    y = torch.empty(T, N, device=x.device, dtype=torch.float32)
    x = x.contiguous()
    _gemv1[lambda m: (triton.cdiv(N, m["BN"]), T)](x, W, y, N, K)
    return y


def mm16_dot(x, W):
    """Вариант для пачки через tl.dot (ieee) — для сравнения."""
    T, K = x.shape
    N = W.shape[0]
    y = torch.empty(T, N, device=x.device, dtype=torch.float32)
    _gemvT[lambda m: (triton.cdiv(N, m["BN"]),)](x.contiguous(), W, y, T, N, K)
    return y
