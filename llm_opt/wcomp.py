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


CFG1 = {}  # (N, K, MODE) -> (BN, num_warps), подобрано в графе (sweep_wcomp.py)


def gemvc1(x, c, y, mode, BN=None, w=None, lnw=None, eps=1e-6):
    N, K = c["N"], c["K"]
    if BN is None:
        BN, w = CFG1.get((N, K, mode), (8, 4))
    gemvc1_k[(N // BN,)](x, c["m8"], c["e4"], c["base"], c["rp"], c["er"], c["ec"], c["ev"], y,
                        lnw if lnw is not None else x, eps, N, K, mode,
                        BN=BN, HALF=c["CH"] // 2, RMS=lnw is not None, KR=triton.next_power_of_2(K), num_warps=w)


CONF_C = [triton.Config({"BN": bn}, num_warps=w) for bn in (4, 8, 16, 32) for w in (2, 4, 8)]


@triton.autotune(configs=CONF_C, key=["N", "K", "MODE", "TT"], restore_value=["Y"], use_cuda_graph=True)
@triton.jit
def gemvc_k(X, M8, E4, base, RP, ER, EC, EV, Y, T, N, K, MODE: tl.constexpr, TT: tl.constexpr, BN: tl.constexpr):
    """Как gemv_k (MODE 0: Y = Wx; 1: Y += Wx; 2: вход silu(g)*u, Y += Wx), веса — сжатые.
    До TT слов пачки сразу: кусок весов распаковывается один раз и умножается на все слова."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    XS = K * 2 if MODE == 2 else K
    r128 = tl.arange(0, 128)
    rt = tl.arange(0, TT)
    mt = rt < T
    acc = tl.zeros((BN, TT), dtype=tl.float32)
    for k0 in range(0, K, 256):
        rk = k0 + r128
        xp = X + rt[:, None] * XS + rk[None, :]
        if MODE == 2:
            g0 = tl.load(xp, mask=mt[:, None], other=0.0); u0 = tl.load(xp + K, mask=mt[:, None], other=0.0)
            g1 = tl.load(xp + 128, mask=mt[:, None], other=0.0); u1 = tl.load(xp + 128 + K, mask=mt[:, None], other=0.0)
            x0 = g0 / (1.0 + tl.exp(-g0)) * u0
            x1 = g1 / (1.0 + tl.exp(-g1)) * u1
        else:
            x0 = tl.load(xp, mask=mt[:, None], other=0.0)
            x1 = tl.load(xp + 128, mask=mt[:, None], other=0.0)
        e = tl.load(E4 + rn[:, None] * (K // 2) + (k0 // 2 + r128)[None, :])
        m0 = tl.load(M8 + rn[:, None] * K + rk[None, :])
        m1 = tl.load(M8 + rn[:, None] * K + (rk + 128)[None, :])
        w0 = _dec(m0, e & 15, base)
        w1 = _dec(m1, e >> 4, base)
        acc += tl.sum(w0[:, None, :] * x0[None, :, :], axis=2)
        acc += tl.sum(w1[:, None, :] * x1[None, :, :], axis=2)
    e0 = tl.load(RP + pid * BN)
    e1 = tl.load(RP + tl.minimum(pid * BN + BN, N))
    for j0 in range(e0, e1, 16):
        rj = j0 + tl.arange(0, 16)
        mj = rj < e1
        r = tl.load(ER + rj, mask=mj, other=-1)
        cc = tl.load(EC + rj, mask=mj, other=0)
        v = tl.load(EV + rj, mask=mj, other=0.0)
        xp = X + rt[:, None] * XS + cc[None, :]
        mm = mt[:, None] & mj[None, :]
        if MODE == 2:
            g = tl.load(xp, mask=mm, other=0.0)
            u = tl.load(xp + K, mask=mm, other=0.0)
            xg = g / (1.0 + tl.exp(-g)) * u
        else:
            xg = tl.load(xp, mask=mm, other=0.0)
        pr = v[None, :] * xg  # [TT, 16]
        acc += tl.sum(tl.where(r[None, None, :] == rn[:, None, None], pr[None, :, :], 0.0), axis=2)
    yp = Y + rt[None, :] * N + rn[:, None]
    my = mt[None, :] & (rn[:, None] < N)
    if MODE == 0:
        tl.store(yp, acc, mask=my)
    else:
        tl.store(yp, tl.load(yp, mask=my, other=0.0) + acc, mask=my)


def gemvc(x, c, y, mode, T):
    N, K = c["N"], c["K"]
    TT = 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8 if T <= 8 else 16
    gemvc_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, c["m8"], c["e4"], c["base"], c["rp"], c["er"], c["ec"], c["ev"],
                                                 y, T, N, K, mode, TT)


# ---------------- вариант ~11 бит: порядки тройками ----------------
# Порядок каждого веса сначала — 4-битный код окна (как выше, 15 = исключение из списка). Строка делится на три
# части длины H = ceil(K/3); тройка (j, H+j, 2H+j) кодов -> 1 байт: номер одной из 255 частых троек матрицы
# (таблица LUT), 255 = редкая тройка: её три кода лежат в запасном потоке ESC (uint16), место = смещение блока из
# 128 троек (OFF) + номер среди редких в блоке. Итого ~8 + 8/3 + немного бит на вес.
BH = 128


def compress11(W):
    N, K = W.shape
    c = compress(W)  # m8, base, исключения — общие с 12-битным вариантом
    e4 = c["e4"].int().view(N, K // c["CH"], c["CH"] // 2)
    code = torch.stack([e4 & 15, e4 >> 4], 2).reshape(N, K)
    H = -(-K // 3)
    pad = torch.full((N, 3 * H), int(code.flatten().mode().values), dtype=torch.int32, device=W.device)
    pad[:, :K] = code
    t = pad.view(N, 3, H)
    key = t[:, 0] | (t[:, 1] << 4) | (t[:, 2] << 8)  # [N, H], 12 бит
    cnt = torch.bincount(key.flatten(), minlength=4096)
    top = cnt.argsort(descending=True)[:255]
    tab = torch.full((4096,), 255, dtype=torch.int32, device=W.device)
    tab[top] = torch.arange(255, dtype=torch.int32, device=W.device)
    e3 = tab[key.long()]
    esc = e3 == 255
    nb = -(-H // BH)
    escp = torch.zeros(N, nb * BH, dtype=torch.int32, device=W.device)
    escp[:, :H] = esc.int()
    per_block = escp.view(N, nb, BH).sum(-1).flatten()
    off = torch.zeros_like(per_block)
    off[1:] = per_block.cumsum(0)[:-1]
    lut = torch.zeros(256, dtype=torch.int32, device=W.device)
    lut[:255] = top.int()
    c.update({"e3": e3.to(torch.uint8).contiguous(), "lut": lut, "esc": key[esc].to(torch.int16).contiguous(),
              "off": off.int().view(N, nb).contiguous(), "H": H})
    del c["e4"]
    return c


def decompress11(c):
    N, K, H = c["N"], c["K"], c["H"]
    e3 = c["e3"].long()
    key = c["lut"][e3]
    esc = e3 == 255
    key[esc] = c["esc"].int() & 0xFFF
    code = torch.cat([key & 15, (key >> 4) & 15, (key >> 8) & 15], 1)[:, :K]  # [N, 3H] -> [N, K]
    m = c["m8"].int()
    bits = ((m >> 7) << 31) | ((code + c["base"]) << 23) | ((m & 0x7F) << 16)
    bits = torch.where(code == 15, 0, bits)
    Wf = bits.view(torch.float32)
    Wf[c["er"].long(), c["ec"].long()] = c["ev"]
    return Wf.bfloat16()


def bits11(c):
    n = c["N"] * c["K"]
    by = c["m8"].numel() + c["e3"].numel() + c["esc"].numel() * 2 + c["off"].numel() * 4 + c["ev"].numel() * 12 + c["rp"].numel() * 4
    return 8 * by / n


@triton.jit
def _part(X, M8, LNW, rs, rn, k, mk, K, cd, base, acc, MODE: tl.constexpr, RMS: tl.constexpr):
    if MODE == 2:
        g = tl.load(X + k, mask=mk, other=0.0)
        u = tl.load(X + K + k, mask=mk, other=0.0)
        x = g / (1.0 + tl.exp(-g)) * u
    else:
        x = tl.load(X + k, mask=mk, other=0.0)
    if RMS:
        x = tl.load(LNW + k, mask=mk, other=0.0).to(tl.float32) * (x * rs)
    m = tl.load(M8 + rn[:, None] * K + k[None, :], mask=mk[None, :], other=0)
    return acc + tl.sum(_dec(m, cd, base) * x[None, :], axis=1)


@triton.jit
def gemvc11_k(X, M8, E3, LUT, ESC, OFF, base, RP, ER, EC, EV, Y, LNW, eps, N, K, H, NB, MODE: tl.constexpr,
              BN: tl.constexpr, RMS: tl.constexpr, KR: tl.constexpr):
    """Одно слово, веса ~11 бит (порядки тройками). MODE/RMS как в gemvc1_k."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rs = 1.0
    if RMS:
        rr = tl.arange(0, KR)
        xr = tl.load(X + rr, mask=rr < K, other=0.0)
        rs = tl.rsqrt(tl.sum(xr * xr, 0) / K + eps)
    acc = tl.zeros((BN,), dtype=tl.float32)
    rb = tl.arange(0, 128)
    for j0 in range(0, H, 128):
        rj = j0 + rb
        mj = rj < H
        e = tl.load(E3 + rn[:, None] * H + rj[None, :], mask=mj[None, :], other=0).to(tl.int32)
        lk = tl.load(LUT + e)
        es = (e == 255) & mj[None, :]
        pos = tl.load(OFF + rn * NB + j0 // 128)[:, None] + tl.cumsum(es.to(tl.int32), axis=1) - 1
        sk = tl.load(ESC + pos, mask=es, other=0).to(tl.int32)
        key = tl.where(es, sk, lk)
        acc = _part(X, M8, LNW, rs, rn, rj, mj, K, key & 15, base, acc, MODE, RMS)
        acc = _part(X, M8, LNW, rs, rn, H + rj, mj, K, (key >> 4) & 15, base, acc, MODE, RMS)
        acc = _part(X, M8, LNW, rs, rn, 2 * H + rj, mj & (2 * H + rj < K), K, (key >> 8) & 15, base, acc, MODE, RMS)
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


CFG11 = {}


def gemvc11(x, c, y, mode, BN=None, w=None, lnw=None, eps=1e-6):
    N, K = c["N"], c["K"]
    if BN is None:
        BN, w = CFG11.get((N, K, mode), (4, 2))
    gemvc11_k[(N // BN,)](x, c["m8"], c["e3"], c["lut"], c["esc"], c["off"], c["base"], c["rp"], c["er"], c["ec"],
                         c["ev"], y, lnw if lnw is not None else x, eps, N, K, c["H"], c["off"].shape[1], mode,
                         BN=BN, RMS=lnw is not None, KR=triton.next_power_of_2(K), num_warps=w)


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


CFGT = {}  # (N, K, MODE) -> (BN, num_warps, S)
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
