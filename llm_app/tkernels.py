"""Ядра Triton для движка: gemv на bf16 (одно слово и пачка до 16 на тензорных блоках), нормировка,
поворот+кэш, внимание. Перенесены из llm_opt (fused_decode, fused2, big_engine)."""
import triton
import triton.language as tl

CONF = [triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (4, 8, 16, 32) for bk in (256, 512, 1024) for w in (2, 4, 8)]


@triton.autotune(configs=CONF, key=["N", "K", "MODE"], restore_value=["Y"])
@triton.jit
def gemv_k(X, W, Y, T, N, K, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """MODE 0: Y = Wx; 1: Y += Wx; 2: вход x_k = silu(X[k]) * X[K+k], Y += Wx.
    Слова пачки идут по очереди внутри программы: та же арифметика, что для одного слова, веса из кэша."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    XS = K * 2 if MODE == 2 else K
    for t in range(T):
        _gemv_row(X + t * XS, W, Y + t * N, rn, N, K, MODE, BN, BK)


@triton.jit
def _gemv_row(X, W, Y, rn, N, K, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        if MODE == 2:
            g = tl.load(X + rk, mask=mk, other=0.0)
            u = tl.load(X + K + rk, mask=mk, other=0.0)
            x = g / (1.0 + tl.exp(-g)) * u
        else:
            x = tl.load(X + rk, mask=mk, other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    if MODE == 0:
        tl.store(Y + rn, acc, mask=rn < N)
    else:
        y = tl.load(Y + rn, mask=rn < N, other=0.0)
        tl.store(Y + rn, y + acc, mask=rn < N)


CONF_TC = [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
           for bn in (16, 32, 64) for bk in (64, 128, 256) for w in (4, 8) for st in (2, 3)]


@triton.autotune(configs=CONF_TC, key=["N", "K", "MODE"], restore_value=["Y"])
@triton.jit
def gemv_tc_k(X, W, Y, T, N, K, MODE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Пачка до 16 слов на тензорных блоках: вход раскладывается на 3 части bf16 (~24 бита, как fp32),
    вес bf16, накопление fp32. MODE как в gemv_k."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, 16)
    XS = K * 2 if MODE == 2 else K
    acc = tl.zeros((BN, 16), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        mx = (rt[None, :] < T) & mk[:, None]
        if MODE == 2:
            g = tl.load(X + rt[None, :] * XS + rk[:, None], mask=mx, other=0.0)
            u = tl.load(X + rt[None, :] * XS + K + rk[:, None], mask=mx, other=0.0)
            x = g / (1.0 + tl.exp(-g)) * u
        else:
            x = tl.load(X + rt[None, :] * XS + rk[:, None], mask=mx, other=0.0)  # [BK, 16]
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & mk[None, :], other=0.0)  # bf16
        x1 = x.to(tl.bfloat16)
        r1 = x - x1.to(tl.float32)
        x2 = r1.to(tl.bfloat16)
        x3 = (r1 - x2.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(w, x1)
        acc += tl.dot(w, x2)
        acc += tl.dot(w, x3)
    my = (rn[:, None] < N) & (rt[None, :] < T)
    yp = Y + rt[None, :] * N + rn[:, None]
    if MODE == 0:
        tl.store(yp, acc, mask=my)
    else:
        tl.store(yp, tl.load(yp, mask=my, other=0.0) + acc, mask=my)


@triton.jit
def qk_rope_cache_k(QKV, QN, KN, INV, POS, QOUT, KC, VC, maxt, eps,
                    NH: tl.constexpr, NKV: tl.constexpr, HDIM: tl.constexpr, half: tl.constexpr):
    """Программа на (голову, слово): q — нормировка+поворот в QOUT; k — в кэш; v — копия в кэш."""
    h = tl.program_id(0)
    t = tl.program_id(1)
    QKV += t * (NH + 2 * NKV) * HDIM
    QOUT += t * NH * HDIM
    r = tl.arange(0, half)
    pos = tl.load(POS) + t
    ang = pos.to(tl.float32) * tl.load(INV + r)
    c, s = tl.cos(ang), tl.sin(ang)
    if h < NH + NKV:
        base = h * HDIM
        a = tl.load(QKV + base + r)
        b = tl.load(QKV + base + half + r)
        rs = tl.rsqrt((tl.sum(a * a, 0) + tl.sum(b * b, 0)) / HDIM + eps)
        if h < NH:
            wa = tl.load(QN + r).to(tl.float32); wb = tl.load(QN + half + r).to(tl.float32)
        else:
            wa = tl.load(KN + r).to(tl.float32); wb = tl.load(KN + half + r).to(tl.float32)
        a = wa * (a * rs)
        b = wb * (b * rs)
        oa = a * c - b * s
        ob = b * c + a * s
        if h < NH:
            tl.store(QOUT + h * HDIM + r, oa)
            tl.store(QOUT + h * HDIM + half + r, ob)
        else:
            g = h - NH
            dst = KC + (g * maxt + pos) * HDIM
            tl.store(dst + r, oa)
            tl.store(dst + half + r, ob)
    else:
        g = h - NH - NKV
        src = QKV + (NH + NKV + g) * HDIM
        dst = VC + (g * maxt + pos) * HDIM
        rr = tl.arange(0, HDIM)
        tl.store(dst + rr, tl.load(src + rr))


@triton.jit
def attn_k(Q, KC, VC, POS, OUT, maxt, scale, G: tl.constexpr, HDIM: tl.constexpr, BJ: tl.constexpr, NHQ: tl.constexpr):
    """Программа на (голову, слово t): внимание по позициям 0..pos+t с онлайн softmax."""
    h = tl.program_id(0)
    t = tl.program_id(1)
    Q += t * NHQ * HDIM
    OUT += t * NHQ * HDIM
    g = h // G
    rd = tl.arange(0, HDIM)
    q = tl.load(Q + h * HDIM + rd)
    n = tl.load(POS) + t + 1
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HDIM,), dtype=tl.float32)
    for j0 in range(0, n, BJ):
        rj = j0 + tl.arange(0, BJ)
        mj = rj < n
        k = tl.load(KC + (g * maxt + rj[:, None]) * HDIM + rd[None, :], mask=mj[:, None], other=0.0)
        sc = tl.sum(k * q[None, :], axis=1) * scale
        sc = tl.where(mj, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        p = tl.exp(sc - m_new)
        alpha = tl.exp(m_i - m_new)
        v = tl.load(VC + (g * maxt + rj[:, None]) * HDIM + rd[None, :], mask=mj[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    tl.store(OUT + h * HDIM + rd, acc / l_i)


TC_FROM = 4


def gemv(x, W, y, mode, T):
    N, K = W.shape
    if T >= TC_FROM:
        gemv_tc_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, W, y, T, N, K, mode)
    else:
        gemv_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, W, y, T, N, K, mode)


@triton.jit
def rope_attn_k(QKV, QN, KN, INV, POS, KC, VC, OUT, maxt, eps, scale, NH: tl.constexpr, NKV: tl.constexpr,
                HDIM: tl.constexpr, half: tl.constexpr, BJ: tl.constexpr):
    """Одно слово, программа на голову q: нормировка+поворот q и k своей группы, запись k, v в кэш (одна голова
    группы), внимание: текущая позиция — из регистров, прошлые 0..pos-1 — из кэша (онлайн softmax)."""
    h = tl.program_id(0)
    G = NH // NKV
    g = h // G
    r = tl.arange(0, half)
    pos = tl.load(POS)
    ang = pos.to(tl.float32) * tl.load(INV + r)
    c, s = tl.cos(ang), tl.sin(ang)
    qa = tl.load(QKV + h * HDIM + r); qb = tl.load(QKV + h * HDIM + half + r)
    rs = tl.rsqrt((tl.sum(qa * qa, 0) + tl.sum(qb * qb, 0)) / HDIM + eps)
    qa = tl.load(QN + r).to(tl.float32) * (qa * rs); qb = tl.load(QN + half + r).to(tl.float32) * (qb * rs)
    qa, qb = qa * c - qb * s, qb * c + qa * s
    kb0 = (NH + g) * HDIM
    ka = tl.load(QKV + kb0 + r); kb = tl.load(QKV + kb0 + half + r)
    rs = tl.rsqrt((tl.sum(ka * ka, 0) + tl.sum(kb * kb, 0)) / HDIM + eps)
    ka = tl.load(KN + r).to(tl.float32) * (ka * rs); kb = tl.load(KN + half + r).to(tl.float32) * (kb * rs)
    ka, kb = ka * c - kb * s, kb * c + ka * s
    rd = tl.arange(0, HDIM)
    v = tl.load(QKV + (NH + NKV + g) * HDIM + rd)
    if h % G == 0:
        dst = (g * maxt + pos) * HDIM
        tl.store(KC + dst + r, ka); tl.store(KC + dst + half + r, kb)
        tl.store(VC + dst + rd, v)
    m_i = (tl.sum(qa * ka, 0) + tl.sum(qb * kb, 0)) * scale
    l_i = 1.0
    acc = v
    for j0 in range(0, pos, BJ):
        rj = j0 + tl.arange(0, BJ)
        mj = rj < pos
        base = (g * maxt + rj[:, None]) * HDIM
        kA = tl.load(KC + base + r[None, :], mask=mj[:, None], other=0.0)
        kB = tl.load(KC + base + half + r[None, :], mask=mj[:, None], other=0.0)
        sc = (tl.sum(kA * qa[None, :], axis=1) + tl.sum(kB * qb[None, :], axis=1)) * scale
        sc = tl.where(mj, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 0))
        p = tl.exp(sc - m_new)
        alpha = tl.exp(m_i - m_new)
        vv = tl.load(VC + base + rd[None, :], mask=mj[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    tl.store(OUT + h * HDIM + rd, acc / l_i)


@triton.jit
def rmsg_k(X, W, O, eps, D, DP: tl.constexpr):
    t = tl.program_id(0)
    X += t * D
    O += t * D
    r = tl.arange(0, DP)
    m = r < D
    x = tl.load(X + r, mask=m, other=0.0)
    v = tl.sum(x * x, 0) / D
    tl.store(O + r, tl.load(W + r, mask=m, other=0.0).to(tl.float32) * (x * tl.rsqrt(v + eps)), mask=m)


