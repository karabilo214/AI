"""Сжатие весов bf16 без потерь до 12 бит: знак+мантисса (8 бит) отдельно, порядок — 4-битный код в окне из
15 порядков (base..base+14), код 15 — исключение (вес вне окна; таких ~0.03%), его точное значение — в списке
исключений (строка, столбец, значение), прибавляется в той же программе.

Раскладка матрицы [N, K] (K кратно 256): M8[N, K] uint8 = знак<<7 | мантисса;
E4[N, K/2] uint8: в куске из 256 чисел байт i хранит код числа i (младшие 4 бита) и числа 128+i (старшие).
Восстановление: биты fp32 = знак<<31 | (base+код)<<23 | мантисса<<16 (у bf16 и fp32 одинаковый сдвиг порядка)."""
import torch
import triton
import triton.language as tl


def compress(W, CH=256):
    """W [N, K] bf16 -> словарь тензоров сжатой матрицы. CH — длина куска (байт i куска: числа i и CH/2+i)."""
    N, K = W.shape
    assert K % CH == 0
    b = W.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    cnt = torch.bincount(e.flatten(), minlength=256)
    cs = torch.cat([torch.zeros(1, dtype=torch.long, device=W.device), cnt.cumsum(0)])
    base = int((cs[15:] - cs[:-15]).argmax())
    code = e - base
    exc = (code < 0) | (code > 14)
    code = torch.where(exc, 15, code)
    m8 = (((b >> 15) << 7) | (b & 0x7F)).to(torch.uint8)
    c = code.view(N, K // CH, 2, CH // 2)
    e4 = (c[:, :, 0] | (c[:, :, 1] << 4)).to(torch.uint8).reshape(N, K // 2)
    r, col = exc.nonzero(as_tuple=True)  # по строкам по порядку
    rp = torch.zeros(N + 1, dtype=torch.int32, device=W.device)
    rp[1:] = torch.bincount(r, minlength=N).cumsum(0)
    return {"m8": m8.contiguous(), "e4": e4.contiguous(), "base": base, "rp": rp, "er": r.int().contiguous(),
            "ec": col.int().contiguous(), "ev": W[r, col].float().contiguous(), "N": N, "K": K, "CH": CH}


def decompress(c):
    """Для проверки: обратно в bf16."""
    N, K = c["N"], c["K"]
    e4 = c["e4"].int().view(N, K // c["CH"], c["CH"] // 2)
    code = torch.stack([e4 & 15, e4 >> 4], 2).reshape(N, K)
    m = c["m8"].int()
    bits = ((m >> 7) << 31) | ((code + c["base"]) << 23) | ((m & 0x7F) << 16)
    bits = torch.where(code == 15, 0, bits)
    W = bits.view(torch.float32)
    W[c["er"].long(), c["ec"].long()] = c["ev"]
    return W.bfloat16()


@triton.jit
def _dec(m, c, base):
    m = m.to(tl.int32)
    c = c.to(tl.int32)
    bits = ((m & 0x80) << 24) | ((c + base) << 23) | ((m & 0x7F) << 16)
    bits = tl.where(c == 15, 0, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _xin(X, rk, K, MODE: tl.constexpr):
    if MODE == 2:
        g = tl.load(X + rk)
        u = tl.load(X + K + rk)
        return g / (1.0 + tl.exp(-g)) * u
    else:
        return tl.load(X + rk)


@triton.jit
def gemvc1_k(X, M8, E4, base, RP, ER, EC, EV, Y, LNW, eps, N, K, MODE: tl.constexpr, BN: tl.constexpr,
             HALF: tl.constexpr, RMS: tl.constexpr, KR: tl.constexpr):
    """Одно слово: как gemv_k (MODE 0: Y = Wx; 1: Y += Wx; 2: вход silu(g)*u, Y += Wx), веса — сжатые.
    RMS: вход — нормировка X (длина KR) с весами LNW, считается в каждой программе (как rms_k)."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rh = tl.arange(0, HALF)
    if RMS:
        rr = tl.arange(0, KR)
        xr = tl.load(X + rr, mask=rr < K, other=0.0)
        rs = tl.rsqrt(tl.sum(xr * xr, 0) / K + eps)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, 2 * HALF):
        rk = k0 + rh
        x0 = _xin(X, rk, K, MODE)
        x1 = _xin(X, rk + HALF, K, MODE)
        if RMS:
            x0 = tl.load(LNW + rk).to(tl.float32) * (x0 * rs)
            x1 = tl.load(LNW + rk + HALF).to(tl.float32) * (x1 * rs)
        e = tl.load(E4 + rn[:, None] * (K // 2) + (k0 // 2 + rh)[None, :])
        m0 = tl.load(M8 + rn[:, None] * K + rk[None, :])
        m1 = tl.load(M8 + rn[:, None] * K + (rk + HALF)[None, :])
        acc += tl.sum(_dec(m0, e & 15, base) * x0[None, :], axis=1)
        acc += tl.sum(_dec(m1, e >> 4, base) * x1[None, :], axis=1)
    e0 = tl.load(RP + pid * BN)
    e1 = tl.load(RP + tl.minimum(pid * BN + BN, N))
    for j0 in range(e0, e1, 16):
        rj = j0 + tl.arange(0, 16)
        mj = rj < e1
        r = tl.load(ER + rj, mask=mj, other=-1)
        cc = tl.load(EC + rj, mask=mj, other=0)
        v = tl.load(EV + rj, mask=mj, other=0.0)
        if MODE == 2:
            g = tl.load(X + cc, mask=mj, other=0.0)
            u = tl.load(X + K + cc, mask=mj, other=0.0)
            xg = g / (1.0 + tl.exp(-g)) * u
        else:
            xg = tl.load(X + cc, mask=mj, other=0.0)
        if RMS:
            xg = tl.load(LNW + cc, mask=mj, other=0.0).to(tl.float32) * (xg * rs)
        acc += tl.sum(tl.where(r[None, :] == rn[:, None], (v * xg)[None, :], 0.0), axis=1)
    if MODE == 0:
        tl.store(Y + rn, acc)
    else:
        tl.store(Y + rn, tl.load(Y + rn) + acc)


CFG1 = {}  # (N, K, MODE) -> (BN, num_warps); заполняет tune.py


def gemvc1(x, c, y, mode, BN=None, w=None, lnw=None, eps=1e-6):
    N, K = c["N"], c["K"]
    if BN is None:
        BN, w = CFG1.get((N, K, mode), (8, 4))
    gemvc1_k[(N // BN,)](x, c["m8"], c["e4"], c["base"], c["rp"], c["er"], c["ec"], c["ev"], y,
                        lnw if lnw is not None else x, eps, N, K, mode,
                        BN=BN, HALF=c["CH"] // 2, RMS=lnw is not None, KR=triton.next_power_of_2(K), num_warps=w)


# ---------------- пачка до 16 слов на сжатых 12-битных весах (тензорные блоки) ----------------
@triton.jit
def _x3(x, w, acc):
    """acc += w @ x, x fp32 [BK, 16] раскладывается на 3 части bf16 (точность как у fp32), w bf16 [BN, BK]."""
    x1 = x.to(tl.bfloat16)
    r1 = x - x1.to(tl.float32)
    x2 = r1.to(tl.bfloat16)
    x3 = (r1 - x2.to(tl.float32)).to(tl.bfloat16)
    acc += tl.dot(w, x1)
    acc += tl.dot(w, x2)
    acc += tl.dot(w, x3)
    return acc


@triton.jit
def _xt(X, rk, rt, mt, XS, K, MODE: tl.constexpr):
    """Вход для пачки: [len(rk), 16] (столбец = слово)."""
    p = X + rt[None, :] * XS + rk[:, None]
    if MODE == 2:
        g = tl.load(p, mask=mt[None, :], other=0.0)
        u = tl.load(p + K, mask=mt[None, :], other=0.0)
        return g / (1.0 + tl.exp(-g)) * u
    else:
        return tl.load(p, mask=mt[None, :], other=0.0)


@triton.jit
def gemvc_tc_k(X, M8, E4, base, RP, ER, EC, EV, Y, PART, CNT, T, N, K, MODE: tl.constexpr, BN: tl.constexpr,
               S: tl.constexpr):
    """Y[t] (+)= W x[t] для t < T <= 16; MODE как в gemvc1_k. Кусок весов распаковывается один раз и идёт на все слова.
    S > 1: вход делится на S частей (кусками по 256); части пишутся в PART, последняя закончившая программа блока
    складывает их в порядке 0..S-1 (одинаковый результат от запуска к запуску)."""
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, 16)
    mt = rt < T
    XS = K * 2 if MODE == 2 else K
    r128 = tl.arange(0, 128)
    nch = K // 256
    c_lo = sp * nch // S
    c_hi = (sp + 1) * nch // S
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    for ci in range(c_lo, c_hi):
        k0 = ci * 256
        rk = k0 + r128
        e = tl.load(E4 + rn[:, None] * (K // 2) + (k0 // 2 + r128)[None, :])
        m0 = tl.load(M8 + rn[:, None] * K + rk[None, :])
        m1 = tl.load(M8 + rn[:, None] * K + (rk + 128)[None, :])
        acc = _x3(_xt(X, rk, rt, mt, XS, K, MODE), _dec(m0, e & 15, base).to(tl.bfloat16), acc)
        acc = _x3(_xt(X, rk + 128, rt, mt, XS, K, MODE), _dec(m1, e >> 4, base).to(tl.bfloat16), acc)
    if sp == 0:  # исключения — в части 0
        e0 = tl.load(RP + pid * BN)
        e1 = tl.load(RP + tl.minimum(pid * BN + BN, N))
        for j0 in range(e0, e1, 16):
            rj = j0 + tl.arange(0, 16)
            mj = rj < e1
            r = tl.load(ER + rj, mask=mj, other=-1)
            cc = tl.load(EC + rj, mask=mj, other=0)
            v = tl.load(EV + rj, mask=mj, other=0.0)
            p = X + rt[None, :] * XS + cc[:, None]  # [16 исключений, 16 слов]
            mm = mj[:, None] & mt[None, :]
            if MODE == 2:
                g = tl.load(p, mask=mm, other=0.0)
                u = tl.load(p + K, mask=mm, other=0.0)
                xg = g / (1.0 + tl.exp(-g)) * u
            else:
                xg = tl.load(p, mask=mm, other=0.0)
            pr = v[:, None] * xg
            acc += tl.sum(tl.where(r[None, :, None] == rn[:, None, None], pr[None, :, :], 0.0), axis=1)
    yp = Y + rt[None, :] * N + rn[:, None]
    my = mt[None, :] & (rn[:, None] < N)
    if S > 1:
        pp = PART + (sp * 16 + rt[None, :]) * N + rn[:, None]
        tl.store(pp, acc, mask=my)
        tl.debug_barrier()
        old = tl.atomic_add(CNT + pid, 1, sem="acq_rel")
        if old == S - 1:
            acc = tl.zeros((BN, 16), dtype=tl.float32)
            for j in range(S):
                acc += tl.load(PART + (j * 16 + rt[None, :]) * N + rn[:, None], mask=my, other=0.0, cache_modifier=".cg")
            if MODE == 0:
                tl.store(yp, acc, mask=my)
            else:
                tl.store(yp, tl.load(yp, mask=my, other=0.0) + acc, mask=my)
            tl.atomic_xchg(CNT + pid, 0)
    else:
        if MODE == 0:
            tl.store(yp, acc, mask=my)
        else:
            tl.store(yp, tl.load(yp, mask=my, other=0.0) + acc, mask=my)


CFGT = {}  # (N, K, MODE) -> (BN, num_warps, S); заполняет tune.py
_SK = {}


def _sk_buf(N, S, dev):
    k = (N, str(dev))
    if k not in _SK or _SK[k][0].shape[0] < S * 16:
        _SK[k] = (torch.zeros(S * 16, N, device=dev), torch.zeros(N, dtype=torch.int32, device=dev))
    return _SK[k]


def gemvc_tc(x, c, y, mode, T, BN=None, w=None, S=None):
    N, K = c["N"], c["K"]
    if BN is None:
        BN, w, S = CFGT.get((N, K, mode), (32, 4, 1))
    S = S or 1
    part, cnt = _sk_buf(N, max(S, 8), x.device)
    gemvc_tc_k[(triton.cdiv(N, BN), S)](x, c["m8"], c["e4"], c["base"], c["rp"], c["er"], c["ec"], c["ev"], y, part, cnt,
                                       T, N, K, mode, BN=BN, S=S, num_warps=w)
