"""Своё ядро для одного слова (Triton): 8 программ видеокарты на слой вместо ~30 операций torch.

Веса bf16 [N, K], счёт fp32. На слой: rms -> gemv(qkv) -> нормировка голов + поворот + запись k, v в кэш ->
внимание (онлайн softmax) -> gemv(o) с прибавлением к x -> rms -> gemv(gate|up) -> gemv(down) с SiLU на входе
и прибавлением к x. Шаг записывается в CUDA-граф. Пачка (чтение запроса, проверка угадываний) — путь GpuEngine
(mode w16s2, cuBLAS), кэш общий.
"""
import torch
import triton
import triton.language as tl

from core import EPS, HD, H, KV, L
from gpu_engine import GpuEngine

D, FF = 1024, 3072
CONF = [triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (4, 8, 16, 32) for bk in (256, 512, 1024) for w in (2, 4, 8)]


@triton.jit
def rms_k(X, W, O, eps, DD: tl.constexpr):
    t = tl.program_id(0)
    X += t * DD
    O += t * DD
    r = tl.arange(0, DD)
    x = tl.load(X + r)
    v = tl.sum(x * x, 0) / DD
    tl.store(O + r, tl.load(W + r).to(tl.float32) * (x * tl.rsqrt(v + eps)))


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


TC_FROM = 4  # с какого размера пачки — тензорные блоки
ATTN = {"BJ": 64, "num_warps": 4}  # настройки attn_k для шага


def gemv(x, W, y, mode, T):
    N, K = W.shape
    if T >= TC_FROM:
        gemv_tc_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, W, y, T, N, K, mode)
    else:
        gemv_k[lambda m: (triton.cdiv(N, m["BN"]),)](x, W, y, T, N, K, mode)


class FusedEngine(GpuEngine):
    """Чтение запроса — как в GpuEngine (w16s2, без графа); слово и пачка до 16 — свои ядра в CUDA-графах."""

    def __init__(self, maxt=1024, Ts=(1, 2, 4, 8, 16)):
        super().__init__(mode="w16s2", maxt=maxt, Ts=())
        m = self.m
        self.Wn = [{n: w[n].T.contiguous() for n in ("qkv", "o", "gu", "down")} for w in m.layers]  # [N, K]
        for w in m.layers:
            for n in ("qkv", "o", "gu", "down"):
                del w[n]
        torch.cuda.empty_cache()
        self.emb16 = m.emb  # [V, K] bf16
        TM = 16
        f = lambda *s: torch.zeros(*s, device="cuda")
        self.x, self.h, self.qkv = f(TM, D), f(TM, D), f(TM, (H + 2 * KV) * HD)
        self.q, self.att, self.gu, self.logits = f(TM, H * HD), f(TM, H * HD), f(TM, 2 * FF), f(TM, m.emb.shape[0])
        self.tok = torch.zeros(TM, dtype=torch.long, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.graphs, self.outs = {}, {}
        for T in Ts:
            self._capture(T)

    def prefill(self, tokens):
        """Чтение запроса: путь GpuEngine, но с нашими весами [N, K] (через транспонированный вид)."""
        m = self.m
        for l, w in enumerate(m.layers):
            for n in ("qkv", "o", "gu", "down"):
                w[n] = self.Wn[l][n].T  # вид без копии
        m.head = self.emb16.T
        try:
            return super().prefill(tokens)
        finally:
            for w in m.layers:
                for n in ("qkv", "o", "gu", "down"):
                    del w[n]

    def _step(self, T):
        m = self.m
        x, h, qkv, q, att, gu = self.x[:T], self.h[:T], self.qkv[:T], self.q[:T], self.att[:T], self.gu[:T]
        x.copy_(m.emb.index_select(0, self.tok[:T]).float())
        for l, w in enumerate(m.layers):
            Wn = self.Wn[l]
            rms_k[(T,)](x, w["ln1"], h, EPS, DD=D)
            self._mm(l, "qkv", h, qkv, 0, T)
            qk_rope_cache_k[(H + 2 * KV, T)](qkv, w["qn"], w["kn"], m.inv_freq, self.pos, q,
                                              self.K[l], self.V[l], self.maxt, EPS, NH=H, NKV=KV, HDIM=HD, half=HD // 2)
            self._attn(l, q, att, T)
            self._mm(l, "o", att, x, 1, T)
            rms_k[(T,)](x, w["ln2"], h, EPS, DD=D)
            self._mm(l, "gu", h, gu, 0, T)
            self._mm(l, "down", gu, x, 2, T)
        rms_k[(T,)](x, m.norm, h, EPS, DD=D)
        return self._head(h, T)

    def _attn(self, l, q, att, T):
        attn_k[(H, T)](q, self.K[l], self.V[l], self.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, NHQ=H, **ATTN)

    def _mm(self, l, n, x, y, mode, T):
        gemv(x, self.Wn[l][n], y, mode, T)

    def _head(self, h, T):
        lg = self.logits[:T]
        gemv(h, self.emb16, lg, 0, T)
        return lg.argmax(-1)

    def _capture(self, T):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._step(T)  # прогрев и подбор настроек ядер (до записи графа)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            self.outs[T] = self._step(T)
        self.graphs[T] = g
        self.K.zero_(); self.V.zero_()

    @torch.no_grad()
    def run(self, tokens, pos0):
        T = len(tokens)
        self.tok[:T].copy_(torch.as_tensor(tokens, device="cuda"))
        self.pos.fill_(pos0)
        self.graphs[T].replay()
        return self.outs[T]
