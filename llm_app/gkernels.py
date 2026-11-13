"""Веса GGUF на видеокарте: перекладка без потерь в удобный вид и ядра Triton (Q4_K, Q6_K, F16).

Перекладка (repack): 4/6-битные числа остаются как были, масштабы выносятся в отдельные массивы, чтобы ядро
читало веса сплошными кусками:
  Q4_K: Q [N, K/2] (байт l куска из 64 чисел: число l — младшие 4 бита, число 32+l — старшие; так и в GGUF),
        SC, MN [N, K/32] uint8 (6-битные масштаб и сдвиг подблока), D [N, K/256, 2] fp16 (d, dmin).
  Q6_K: QL [N, K/2] (младшие 4 бита, в куске из 128 байт i: числа i и 64+i), QH [N, K/4] (старшие 2 бита),
        SC [N, K/16] int8, D [N, K/256] fp16.
Ядра: Y[t] (+)= W x[t] для t < T <= 16, счёт fp32; блок распаковывается один раз на все слова.
MODE 0: Y = Wx, 1: Y += Wx. YS — шаг строк Y (можно писать в кусок общего буфера)."""
import numpy as np
import torch
import triton
import triton.language as tl


def repack(buf, N, K, t, dev="cuda"):
    """buf — байты тензора GGUF (numpy/memmap). Возвращает словарь тензоров на dev."""
    if N <= 0 or K <= 0 or K % 256:
        raise ValueError("GPU-матрица требует N > 0 и K, кратное 256")
    b = torch.from_numpy(np.array(buf)).to(dev)
    if t == "F16":
        return {"t": t, "N": N, "K": K, "W": b.view(torch.float16).view(N, K)}
    if t == "F32":
        return {"t": t, "N": N, "K": K, "W": b.view(torch.float32).view(N, K)}
    nb = K // 256
    if t == "Q4_K":
        blk = b.view(N, nb, 144)
        D = blk[:, :, :4].contiguous().view(torch.float16).view(N, nb, 2)
        s = blk[:, :, 4:16].to(torch.int32)
        sc = torch.empty(N, nb, 8, dtype=torch.int32, device=dev)
        mn = torch.empty_like(sc)
        sc[:, :, :4] = s[:, :, 0:4] & 63
        mn[:, :, :4] = s[:, :, 4:8] & 63
        sc[:, :, 4:] = (s[:, :, 8:12] & 15) | ((s[:, :, 0:4] >> 6) << 4)
        mn[:, :, 4:] = (s[:, :, 8:12] >> 4) | ((s[:, :, 4:8] >> 6) << 4)
        return {"t": t, "N": N, "K": K, "Q": blk[:, :, 16:].contiguous().view(N, K // 2),
                "SC": sc.to(torch.uint8).view(N, K // 32), "MN": mn.to(torch.uint8).view(N, K // 32), "D": D}
    if t == "Q6_K":
        blk = b.view(N, nb, 210)
        return {"t": t, "N": N, "K": K, "QL": blk[:, :, :128].contiguous().view(N, K // 2),
                "QH": blk[:, :, 128:192].contiguous().view(N, K // 4),
                "SC": blk[:, :, 192:208].contiguous().view(torch.int8).view(N, K // 16),
                "D": blk[:, :, 208:210].contiguous().view(torch.float16).view(N, nb)}
    raise NotImplementedError(t)


def nbytes(w):
    return sum(v.numel() * v.element_size() for v in w.values() if torch.is_tensor(v))


@triton.jit
def _put(acc, rt, t, c):
    return acc + tl.where(rt[None, :] == t, c[:, None], 0.0)


@triton.jit
def _store(Y, acc, rn, rt, mt, YS, MODE: tl.constexpr):
    p = Y + rt[None, :] * YS + rn[:, None]
    if MODE == 0:
        tl.store(p, acc, mask=mt[None, :])
    else:
        tl.store(p, tl.load(p, mask=mt[None, :], other=0.0) + acc, mask=mt[None, :])


@triton.jit
def q4k_k(X, Q, SC, MN, D, Y, T, N, K, YS, MODE: tl.constexpr, BN: tl.constexpr, NC: tl.constexpr, TT: tl.constexpr):
    """Кусок из NC*64 чисел строки за шаг."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, TT)
    mt = rt < T
    c = tl.arange(0, NC)
    l = tl.arange(0, 32)
    acc = tl.zeros((BN, TT), dtype=tl.float32)
    for k0 in range(0, K, NC * 64):
        q = tl.load(Q + rn[:, None, None] * (K // 2) + k0 // 2 + c[None, :, None] * 32 + l[None, None, :]).to(tl.int32)
        qlo = (q & 15).to(tl.float32)
        qhi = (q >> 4).to(tl.float32)
        sub = k0 // 32 + 2 * c
        so = rn[:, None] * (K // 32) + sub[None, :]
        slo, shi = tl.load(SC + so).to(tl.float32), tl.load(SC + so + 1).to(tl.float32)
        mlo, mhi = tl.load(MN + so).to(tl.float32), tl.load(MN + so + 1).to(tl.float32)
        do = (rn[:, None] * (K // 256) + k0 // 256 + c[None, :] // 4) * 2
        d = tl.load(D + do).to(tl.float32)
        dm = tl.load(D + do + 1).to(tl.float32)
        for t in range(T):
            xo = X + t * K + k0 + c[:, None] * 64 + l[None, :]
            xlo = tl.load(xo)
            xhi = tl.load(xo + 32)
            dl = tl.sum(qlo * xlo[None, :, :], axis=2)
            dh = tl.sum(qhi * xhi[None, :, :], axis=2)
            sl = tl.sum(xlo, axis=1)
            sh = tl.sum(xhi, axis=1)
            cc = tl.sum(d * (slo * dl + shi * dh) - dm * (mlo * sl[None, :] + mhi * sh[None, :]), axis=1)
            acc = _put(acc, rt, t, cc)
    _store(Y, acc, rn, rt, mt, YS, MODE)


@triton.jit
def q6k_k(X, QL, QH, SC, D, Y, T, N, K, YS, MODE: tl.constexpr, BN: tl.constexpr, TT: tl.constexpr):
    """Блок из 256 чисел за шаг. Числа блока: [hc=4][s=2][i=16]: h = hc//2 (половина по 128), c = hc%2,
    l = 16s + i; младшая тетрада — число h*128 + c*32 + l, старшая — +64. Масштаб — один на 16 чисел."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, TT)
    mt = rt < T
    hc = tl.arange(0, 4)[:, None, None]
    h = hc // 2
    c = hc % 2
    sI = tl.arange(0, 2)[None, :, None]
    i = tl.arange(0, 16)[None, None, :]
    l = sI * 16 + i  # [1, 2, 16]
    r4 = rn[:, None, None, None]
    acc = tl.zeros((BN, TT), dtype=tl.float32)
    for k0 in range(0, K, 256):
        L = tl.load(QL + r4 * (K // 2) + k0 // 2 + (h * 64 + c * 32 + l)[None]).to(tl.int32)
        H = tl.load(QH + r4 * (K // 4) + k0 // 4 + (h * 32 + l)[None]).to(tl.int32)
        sh = (2 * c)[None]
        qlo = ((L & 15) | (((H >> sh) & 3) << 4)) - 32
        qhi = ((L >> 4) | (((H >> (sh + 4)) & 3) << 4)) - 32
        so = r4 * (K // 16) + k0 // 16 + (h * 8 + 2 * c + sI)[None]  # [BN, 4, 2, 1]
        wlo = tl.load(SC + so).to(tl.float32) * qlo.to(tl.float32)
        whi = tl.load(SC + so + 4).to(tl.float32) * qhi.to(tl.float32)
        d = tl.load(D + rn * (K // 256) + k0 // 256).to(tl.float32)
        for t in range(T):
            xo = X + t * K + k0 + h * 128 + c * 32 + l  # [4, 2, 16]
            xlo = tl.load(xo)
            xhi = tl.load(xo + 64)
            v = wlo * xlo[None] + whi * xhi[None]
            s = tl.sum(tl.sum(tl.sum(v, axis=3), axis=2), axis=1)
            acc = _put(acc, rt, t, d * s)
    _store(Y, acc, rn, rt, mt, YS, MODE)


@triton.jit
def f16_k(X, W, Y, T, N, K, YS, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TT: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, TT)
    mt = rt < T
    r = tl.arange(0, BK)
    acc = tl.zeros((BN, TT), dtype=tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(W + rn[:, None] * K + k0 + r[None, :]).to(tl.float32)
        for t in range(T):
            acc = _put(acc, rt, t, tl.sum(w * tl.load(X + t * K + k0 + r)[None, :], axis=1))
    _store(Y, acc, rn, rt, mt, YS, MODE)


@triton.jit
def silu_mul_k(GU, M, FF, BF: tl.constexpr):
    t = tl.program_id(0)
    r = tl.program_id(1) * BF + tl.arange(0, BF)
    m = r < FF
    g = tl.load(GU + t * 2 * FF + r, mask=m, other=0.0)
    u = tl.load(GU + t * 2 * FF + FF + r, mask=m, other=0.0)
    tl.store(M + t * FF + r, g / (1.0 + tl.exp(-g)) * u, mask=m)


# ---------------- пачка 2..16 слов на тензорных блоках ----------------
# Вход раскладывается один раз (xprep_k) на 3 части bf16 (~fp32) + суммы по подблокам 32. Вес подблока
# sc*q — целое (Q4_K: 0..945, Q6_K: -4096..4064); оно точно делится на две части bf16: a*S + b. Умножения
# на тензорных блоках длиной 128, масштаб d (и сдвиги Q4_K) — после, в fp32.
@triton.jit
def xprep_k(X, X3, SX, T, K):
    t = tl.program_id(0)
    j = tl.program_id(1)
    r = j * 32 + tl.arange(0, 32)
    x = tl.load(X + t * K + r)
    x1 = x.to(tl.bfloat16)
    e = x - x1.to(tl.float32)
    x2 = e.to(tl.bfloat16)
    x3 = (e - x2.to(tl.float32)).to(tl.bfloat16)
    tl.store(X3 + t * K + r, x1)
    tl.store(X3 + 16 * K + t * K + r, x2)
    tl.store(X3 + 32 * K + t * K + r, x3)
    tl.store(SX + t * (K // 32) + j, tl.sum(x, 0))


@triton.jit
def _dx(X3, k, T, K, a, b, S: tl.constexpr):
    """(S*a + b) @ x^T, x [len(k), 16] из трёх частей bf16."""
    rt = tl.arange(0, 16)
    p = X3 + rt[None, :] * K + k[:, None]
    m = rt[None, :] < T
    r = tl.zeros((a.shape[0], 16), dtype=tl.float32)
    for i in tl.static_range(3):
        x = tl.load(p + i * 16 * K, mask=m, other=0.0)
        r += tl.dot(a, x) * S + tl.dot(b, x)
    return r


@triton.jit
def q4k_tc_k(X3, SX, Q, SC, MN, D, Y, T, N, K, YS, MODE: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, 16)
    c = tl.arange(0, 4)
    l = tl.arange(0, 32)
    kk = tl.reshape(c[:, None] * 64 + l[None, :], (128,))
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    for k0 in range(0, K, 256):
        do = (rn * (K // 256) + k0 // 256) * 2
        d = tl.load(D + do).to(tl.float32)
        dm = tl.load(D + do + 1).to(tl.float32)
        q = tl.load(Q + rn[:, None, None] * (K // 2) + k0 // 2 + c[None, :, None] * 32 + l[None, None, :]).to(tl.int32)
        so = rn[:, None] * (K // 32) + k0 // 32 + 2 * c[None, :]
        wl = tl.reshape(tl.load(SC + so).to(tl.int32)[:, :, None] * (q & 15), (BN, 128))
        wh = tl.reshape(tl.load(SC + so + 1).to(tl.int32)[:, :, None] * (q >> 4), (BN, 128))
        blk = _dx(X3, k0 + kk, T, K, (wl >> 5).to(tl.bfloat16), (wl & 31).to(tl.bfloat16), 32)
        blk += _dx(X3, k0 + kk + 32, T, K, (wh >> 5).to(tl.bfloat16), (wh & 31).to(tl.bfloat16), 32)
        # сдвиги: dm * sum_sub m_sub * sum(x по подблоку)
        sxo = SX + rt[None, :] * (K // 32) + k0 // 32 + 2 * c[:, None]  # [4, 16]
        sxl = tl.load(sxo, mask=rt[None, :] < T, other=0.0)
        sxh = tl.load(sxo + 1, mask=rt[None, :] < T, other=0.0)
        ml = tl.load(MN + so).to(tl.float32)
        mh = tl.load(MN + so + 1).to(tl.float32)
        mins = tl.sum(ml[:, :, None] * sxl[None, :, :], axis=1) + tl.sum(mh[:, :, None] * sxh[None, :, :], axis=1)
        acc += d[:, None] * blk - dm[:, None] * mins
    _store(Y, acc, rn, rt, rt < T, YS, MODE)


@triton.jit
def q6k_tc_k(X3, QL, QH, SC, D, Y, T, N, K, YS, MODE: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, 16)
    hc = tl.arange(0, 4)[:, None, None]
    h = hc // 2
    c = hc % 2
    sI = tl.arange(0, 2)[None, :, None]
    i = tl.arange(0, 16)[None, None, :]
    l = sI * 16 + i
    r4 = rn[:, None, None, None]
    kk = tl.reshape(h * 128 + c * 32 + l, (128,))
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    for k0 in range(0, K, 256):
        L = tl.load(QL + r4 * (K // 2) + k0 // 2 + (h * 64 + c * 32 + l)[None]).to(tl.int32)
        H = tl.load(QH + r4 * (K // 4) + k0 // 4 + (h * 32 + l)[None]).to(tl.int32)
        sh = (2 * c)[None]
        qlo = ((L & 15) | (((H >> sh) & 3) << 4)) - 32
        qhi = ((L >> 4) | (((H >> (sh + 4)) & 3) << 4)) - 32
        so = r4 * (K // 16) + k0 // 16 + (h * 8 + 2 * c + sI)[None]
        wl = tl.reshape(tl.load(SC + so).to(tl.int32) * qlo, (BN, 128))
        wh = tl.reshape(tl.load(SC + so + 4).to(tl.int32) * qhi, (BN, 128))
        blk = _dx(X3, k0 + kk, T, K, (wl >> 6).to(tl.bfloat16), (wl & 63).to(tl.bfloat16), 64)
        blk += _dx(X3, k0 + kk + 64, T, K, (wh >> 6).to(tl.bfloat16), (wh & 63).to(tl.bfloat16), 64)
        acc += tl.load(D + rn * (K // 256) + k0 // 256).to(tl.float32)[:, None] * blk
    _store(Y, acc, rn, rt, rt < T, YS, MODE)


_XP = {}


def xprep(x, T, K):
    key = (K, x.device.index)
    if key not in _XP:
        _XP[key] = (torch.zeros(3, 16, K, dtype=torch.bfloat16, device=x.device), torch.zeros(16, K // 32, device=x.device))
    X3, SX = _XP[key]
    xprep_k[(T, K // 32)](x, X3, SX, T, K)
    return X3, SX


TC_FROM = 2  # с какого размера пачки — тензорные блоки
CFGTC = {}


def ggemv_tc(x, w, y, mode, T, cfg=None, ys=None):
    t, N, K = w["t"], w["N"], w["K"]
    ys = ys or N
    if t == "Q4_K":
        X3, SX = xprep(x, T, K)
        BN, nw = cfg or CFGTC.get((t, N, K), (32, 4))
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        q4k_tc_k[(N // BN,)](X3, SX, w["Q"], w["SC"], w["MN"], w["D"], y, T, N, K, ys, mode, BN=BN, num_warps=nw)
    elif t == "Q6_K":
        X3, SX = xprep(x, T, K)
        BN, nw = cfg or CFGTC.get((t, N, K), (32, 4))
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        q6k_tc_k[(N // BN,)](X3, w["QL"], w["QH"], w["SC"], w["D"], y, T, N, K, ys, mode, BN=BN, num_warps=nw)
    else:  # F16 — обычные вычислители (небольшие матрицы, счёт fp32)
        TT = _tt(T)
        BN, BK, nw = cfg or CFG.get((t, N, K, TT), (4, 256, 1))
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        if K % BK:
            raise ValueError("длина строки не кратна блоку GPU-ядра")
        f16_k[(N // BN,)](x, w["W"], y, T, N, K, ys, mode, BN=BN, BK=BK, TT=TT, num_warps=nw)


def cfg_space_tc(t):
    if t in ("Q4_K", "Q6_K"):
        return [(bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8)]
    return [(bn, bk, nw) for bn in (2, 4, 8) for bk in (128, 256, 512) for nw in (1, 2, 4)]


CFG = {}  # (тип, N, K, TT) -> настройки


def _tt(T):
    return 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8 if T <= 8 else 16


def ggemv(x, w, y, mode, T, cfg=None, ys=None):
    """y[:T, :N] (+)= x[:T] @ W^T; x — [T, K] fp32 подряд, y — буфер с шагом строк ys (по умолчанию N)."""
    t, N, K = w["t"], w["N"], w["K"]
    if T >= TC_FROM and cfg is None:
        return ggemv_tc(x, w, y, mode, T, ys=ys)
    TT = _tt(T)
    ys = ys or N
    if t == "Q4_K":
        BN, NC, nw = cfg or CFG.get((t, N, K, TT), (8, 8, 1))
        while K % (NC * 64):  # шаг ядра должен делить длину строки (у экспертов MoE бывает 768)
            NC //= 2
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        q4k_k[(N // BN,)](x, w["Q"], w["SC"], w["MN"], w["D"], y, T, N, K, ys, mode, BN=BN, NC=NC, TT=TT, num_warps=nw)
    elif t == "Q6_K":
        BN, nw = cfg or CFG.get((t, N, K, TT), (4, 1))
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        q6k_k[(N // BN,)](x, w["QL"], w["QH"], w["SC"], w["D"], y, T, N, K, ys, mode, BN=BN, TT=TT, num_warps=nw)
    elif t in ("F16", "F32"):
        BN, BK, nw = cfg or CFG.get((t, N, K, TT), (4, 256, 1))
        if N % BN:
            raise ValueError("число строк не кратно блоку GPU-ядра")
        if K % BK:
            raise ValueError("длина строки не кратна блоку GPU-ядра")
        f16_k[(N // BN,)](x, w["W"], y, T, N, K, ys, mode, BN=BN, BK=BK, TT=TT, num_warps=nw)
    else:
        raise NotImplementedError(t)


def silu_mul(gu, m, T, FF):
    silu_mul_k[(T, triton.cdiv(FF, 1024))](gu, m, FF, BF=1024)


def cfg_space(t):
    if t == "Q4_K":
        return [(bn, nc, nw) for bn in (1, 2, 4, 8) for nc in (2, 4, 8) for nw in (1, 2, 4)]
    if t == "Q6_K":
        return [(bn, nw) for bn in (1, 2, 4, 8) for nw in (1, 2, 4)]
    return [(bn, bk, nw) for bn in (1, 2, 4, 8) for bk in (128, 256, 512) for nw in (1, 2, 4)]


# ---------------- MoE: групповые ядра (все пары «слово-эксперт» слоя за один запуск) ----------------
# Эксперты в видеокарте сложены подряд (слот s — строки s*N .. s*N+N-1 общих массивов). Программа: пара p и
# блок строк; эксперт = SLOT[p], вход = строка TOK[p] матрицы X, выход — строка p матрицы Y [P, N].
@triton.jit
def q4k_moe_k(X, Q, SC, MN, D, Y, SLOT, TOK, N, K, BN: tl.constexpr, NC: tl.constexpr):
    p = tl.program_id(0)
    s = tl.load(SLOT + p)
    t = tl.load(TOK + p)
    rl = tl.program_id(1) * BN + tl.arange(0, BN)
    rn = s * N + rl
    c = tl.arange(0, NC)
    l = tl.arange(0, 32)
    acc = tl.zeros((BN,), dtype=tl.float32)
    Xt = X + t * K
    for k0 in range(0, K, NC * 64):
        q = tl.load(Q + rn[:, None, None] * (K // 2) + k0 // 2 + c[None, :, None] * 32 + l[None, None, :]).to(tl.int32)
        so = rn[:, None] * (K // 32) + k0 // 32 + 2 * c[None, :]
        do = (rn[:, None] * (K // 256) + k0 // 256 + c[None, :] // 4) * 2
        d = tl.load(D + do).to(tl.float32)
        dm = tl.load(D + do + 1).to(tl.float32)
        xo = Xt + k0 + c[:, None] * 64 + l[None, :]
        xlo = tl.load(xo)
        xhi = tl.load(xo + 32)
        dl = tl.sum((q & 15).to(tl.float32) * xlo[None, :, :], axis=2)
        dh = tl.sum((q >> 4).to(tl.float32) * xhi[None, :, :], axis=2)
        acc += tl.sum(d * (tl.load(SC + so).to(tl.float32) * dl + tl.load(SC + so + 1).to(tl.float32) * dh)
                      - dm * (tl.load(MN + so).to(tl.float32) * tl.sum(xlo, axis=1)[None, :]
                              + tl.load(MN + so + 1).to(tl.float32) * tl.sum(xhi, axis=1)[None, :]), axis=1)
    tl.store(Y + p * N + rl, acc)


@triton.jit
def q6k_moe_k(X, QL, QH, SC, D, Y, SLOT, TOK, N, K, BN: tl.constexpr):
    p = tl.program_id(0)
    s = tl.load(SLOT + p)
    t = tl.load(TOK + p)
    rl = tl.program_id(1) * BN + tl.arange(0, BN)
    rn = s * N + rl
    hc = tl.arange(0, 4)[:, None, None]
    h = hc // 2
    c = hc % 2
    sI = tl.arange(0, 2)[None, :, None]
    i = tl.arange(0, 16)[None, None, :]
    l = sI * 16 + i
    r4 = rn[:, None, None, None]
    Xt = X + t * K
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, 256):
        L = tl.load(QL + r4 * (K // 2) + k0 // 2 + (h * 64 + c * 32 + l)[None]).to(tl.int32)
        H = tl.load(QH + r4 * (K // 4) + k0 // 4 + (h * 32 + l)[None]).to(tl.int32)
        sh = (2 * c)[None]
        qlo = ((L & 15) | (((H >> sh) & 3) << 4)) - 32
        qhi = ((L >> 4) | (((H >> (sh + 4)) & 3) << 4)) - 32
        so = r4 * (K // 16) + k0 // 16 + (h * 8 + 2 * c + sI)[None]
        wlo = tl.load(SC + so).to(tl.float32) * qlo.to(tl.float32)
        whi = tl.load(SC + so + 4).to(tl.float32) * qhi.to(tl.float32)
        d = tl.load(D + rn * (K // 256) + k0 // 256).to(tl.float32)
        xo = Xt + k0 + h * 128 + c * 32 + l
        v = wlo * tl.load(xo)[None] + whi * tl.load(xo + 64)[None]
        acc += d * tl.sum(tl.sum(tl.sum(v, axis=3), axis=2), axis=1)
    tl.store(Y + p * N + rl, acc)


def moe_gemv(x, w, y, slot, tok, P):
    """y[p] = W[slot[p]] x[tok[p]] для p < P; w — сложенные эксперты (repack, N — строк одного эксперта)."""
    t, N, K = w["t"], w["N"], w["K"]
    if t == "Q4_K":
        NC = 8
        while K % (NC * 64):
            NC //= 2
        q4k_moe_k[(P, N // 8)](x, w["Q"], w["SC"], w["MN"], w["D"], y, slot, tok, N, K, BN=8, NC=NC, num_warps=1)
    elif t == "Q6_K":
        q6k_moe_k[(P, N // 4)](x, w["QL"], w["QH"], w["SC"], w["D"], y, slot, tok, N, K, BN=4, num_warps=1)
    else:
        raise NotImplementedError(t)
