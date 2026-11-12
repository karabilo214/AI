"""Сжатие с потерями до ~4.5 бит (для угадывателя): группы по 64 числа строки, w ~ q*s + m, q — 4 бита,
s и m — fp16 на группу. Раскладка [N, K/2] uint8: в куске из 128 чисел байт i хранит число i (младшие 4 бита,
группа 2c) и число 64+i (старшие, группа 2c+1)."""
import torch
import triton
import triton.language as tl


def quant4(W):
    N, K = W.shape
    assert K % 128 == 0
    w = W.float().view(N, K // 64, 64)
    mn, mx = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-8)
    q = torch.round((w - mn) / s).clamp(0, 15).to(torch.uint8).view(N, K // 128, 2, 64)
    qp = (q[:, :, 0] | (q[:, :, 1] << 4)).reshape(N, K // 2).contiguous()
    return {"q": qp, "s": s.view(N, K // 64).half().contiguous(), "m": mn.view(N, K // 64).half().contiguous(),
            "N": N, "K": K}


def dequant4(c):
    N, K = c["N"], c["K"]
    b = c["q"].view(N, K // 128, 64).int()
    q = torch.stack([b & 15, b >> 4], 2).reshape(N, K // 64, 64).float()
    return (q * c["s"].float()[..., None] + c["m"].float()[..., None]).view(N, K)


@triton.jit
def _xq(X, rk, K, MODE: tl.constexpr):
    if MODE == 2:
        g = tl.load(X + rk)
        u = tl.load(X + K + rk)
        return g / (1.0 + tl.exp(-g)) * u
    else:
        return tl.load(X + rk)


@triton.jit
def gemvq4_k(X, Q, S, M, Y, T, N, K, MODE: tl.constexpr, BN: tl.constexpr):
    """MODE 0: Y = Wx; 1: Y += Wx; 2: вход silu(g)*u, Y += Wx. Слова пачки по очереди (веса из кэша)."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    r64 = tl.arange(0, 64)
    XS = K * 2 if MODE == 2 else K
    G = K // 64
    for t in range(T):
        Xt = X + t * XS
        acc = tl.zeros((BN,), dtype=tl.float32)
        for c in range(0, K // 128):
            k0 = c * 128
            x0 = _xq(Xt, k0 + r64, K, MODE)
            x1 = _xq(Xt, k0 + 64 + r64, K, MODE)
            b = tl.load(Q + rn[:, None] * (K // 2) + (c * 64 + r64)[None, :]).to(tl.int32)
            s0 = tl.load(S + rn * G + 2 * c).to(tl.float32); m0 = tl.load(M + rn * G + 2 * c).to(tl.float32)
            s1 = tl.load(S + rn * G + 2 * c + 1).to(tl.float32); m1 = tl.load(M + rn * G + 2 * c + 1).to(tl.float32)
            w0 = (b & 15).to(tl.float32) * s0[:, None] + m0[:, None]
            w1 = (b >> 4).to(tl.float32) * s1[:, None] + m1[:, None]
            acc += tl.sum(w0 * x0[None, :], axis=1) + tl.sum(w1 * x1[None, :], axis=1)
        if MODE == 0:
            tl.store(Y + t * N + rn, acc)
        else:
            tl.store(Y + t * N + rn, tl.load(Y + t * N + rn) + acc)


CFGQ = {}


def gemvq4(x, c, y, mode, T, BN=None, w=None):
    N, K = c["N"], c["K"]
    if BN is None:
        BN, w = CFGQ.get((N, K, mode), (4, 2))
    gemvq4_k[(N // BN,)](x, c["q"], c["s"], c["m"], y, T, N, K, mode, BN=BN, num_warps=w)
