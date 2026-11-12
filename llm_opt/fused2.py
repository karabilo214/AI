"""FusedEngine + точный выбор слова по словарю int8 (vocab8): читается 156 МБ словаря вместо 311.
generate(): обычная жадная генерация без ожидания процессора на каждом слове — отдельный граф для T=1 сам
кладёт выбранное слово на вход следующего шага, пишет его в историю и сдвигает позицию; все шаги ставятся
в очередь сразу, текст читается в конце."""
import torch
import torch.nn.functional as F

import fused_decode
from core import EPS, HD, H, KV
from fused_decode import D, FF, attn_k, qk_rope_cache_k, rms_k
import triton
import triton.language as tl
from vocab8 import Vocab8
from wcomp import CFG1, CFG11, compress, compress11, gemvc1, gemvc11

PB = (32, 64, 128, 256, 512)  # длины запроса, для которых пишется граф чтения (запрос дополняется до ближайшей)

V8_T = (1, 2)  # для каких размеров пачки словарь int8 (на больших пачках он медленнее полного)


@triton.jit
def split2_k(X, OUT, T, K, MODE: tl.constexpr, BK: tl.constexpr):
    """Строка t входа -> две части bf16: OUT[t] = bf16(x), OUT[T + t] = bf16(x - bf16(x)).
    MODE 2: x_k = silu(X[t, k]) * X[t, K + k] (вход down)."""
    t = tl.program_id(0)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    mk = rk < K
    if MODE == 2:
        g = tl.load(X + t * 2 * K + rk, mask=mk, other=0.0)
        u = tl.load(X + t * 2 * K + K + rk, mask=mk, other=0.0)
        x = g / (1.0 + tl.exp(-g)) * u
    else:
        x = tl.load(X + t * K + rk, mask=mk, other=0.0)
    p1 = x.to(tl.bfloat16)
    p2 = (x - p1.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + t * K + rk, p1, mask=mk)
    tl.store(OUT + (T + t) * K + rk, p2, mask=mk)


@triton.jit
def comb2_k(Y, OUT, T, N, ADD: tl.constexpr, BN: tl.constexpr):
    """OUT[t] = Y[t] + Y[T + t] (ADD: OUT[t] += ...)."""
    t = tl.program_id(0)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mn = rn < N
    y = tl.load(Y + t * N + rn, mask=mn, other=0.0) + tl.load(Y + (T + t) * N + rn, mask=mn, other=0.0)
    if ADD:
        y += tl.load(OUT + t * N + rn, mask=mn, other=0.0)
    tl.store(OUT + t * N + rn, y, mask=mn)


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


class FusedEngine(fused_decode.FusedEngine):
    def __init__(self, maxt=1024, Ts=(1, 2, 4, 8, 16), wbits=12):
        """wbits: 12 — быстрее; 11 — меньше памяти (порядки тройками), медленнее распаковка."""
        self.v8 = self.Wc = None
        self.wbits = wbits
        fused_decode.FusedEngine.__init__(self, maxt, Ts=())
        self.v8 = Vocab8(self.emb16)
        comp = compress11 if wbits == 11 else compress
        self.Wc = [{n: comp(W[n]) for n in W} for W in self.Wn]  # сжатые без потерь — для T=1
        self._gc = gemvc11 if wbits == 11 else gemvc1
        CFG11.update({(4096, 1024, 0): (8, 4), (1024, 2048, 1): (4, 4), (6144, 1024, 0): (8, 4), (1024, 3072, 2): (4, 2)})
        CFG1.update({(4096, 1024, 0): (8, 1), (1024, 2048, 1): (2, 1), (6144, 1024, 0): (2, 1), (1024, 3072, 2): (2, 1)})
        self.hist = torch.zeros(maxt + 1, dtype=torch.long, device="cuda")
        m = self.m
        for l, w in enumerate(m.layers):  # виды [K, N] на наши веса для пути GpuEngine (без копий)
            for n in ("qkv", "o", "gu", "down"):
                w[n] = self.Wn[l][n].T
        m.head = self.emb16.T
        self.pgraphs = {}
        self.pbuf = torch.empty(2 * max(PB) * 2 * FF, dtype=torch.bfloat16, device="cuda")
        for T in Ts:
            self._capture(T)
        self._capture_auto()

    def _step(self, T):
        """T=1: 5 программ на слой (нормировка внутри умножений, поворот+кэш внутри внимания)."""
        if T != 1 or self.Wc is None:
            return super()._step(T)
        m = self.m
        x, qkv, att, gu = self.x[:1], self.qkv[:1], self.att[:1], self.gu[:1]
        x.copy_(m.emb.index_select(0, self.tok[:1]).float())
        for l, w in enumerate(m.layers):
            Wc = self.Wc[l]
            self._gc(x, Wc["qkv"], qkv, 0, lnw=w["ln1"])
            rope_attn_k[(H,)](qkv, w["qn"], w["kn"], m.inv_freq, self.pos, self.K[l], self.V[l], att, self.maxt, EPS,
                              HD ** -0.5, NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)
            self._gc(att, Wc["o"], x, 1)
            self._gc(x, Wc["gu"], gu, 0, lnw=w["ln2"])
            self._gc(gu, Wc["down"], x, 2)
        h = self.h[:1]
        rms_k[(1,)](x, m.norm, h, EPS, DD=D)
        return self._head(h, 1)

    def _mm(self, l, n, x, y, mode, T):
        if T == 1 and getattr(self, "Wc", None) is not None:
            self._gc(x, self.Wc[l][n], y, mode)
        else:
            super()._mm(l, n, x, y, mode, T)

    def _head(self, h, T):
        if self.v8 is not None and T in V8_T:
            return self.v8(h, T)
        return super()._head(h, T)

    def _capture_auto(self):
        pos0 = self.pos.clone(); tok0 = self.tok.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self._step(1)
            self.hist.index_copy_(0, self.pos + 1, out)
            self.tok[:1].copy_(out)
            self.pos.add_(1)
        self.auto = g
        self.pos.copy_(pos0); self.tok.copy_(tok0)
        self.K.zero_(); self.V.zero_()

    def _mm2(self, x, W, out, T, add=False, mode=0):
        """out[T, N] (+)= x @ W^T, W [N, K] bf16; вход — 2 части bf16 (как w16s2), умножение cuBLAS."""
        N, K = W.shape
        parts = self.pbuf[:2 * T * K].view(2 * T, K)
        split2_k[(T, triton.cdiv(K, 1024))](x, parts, T, K, MODE=mode, BK=1024)
        Y = torch.mm(parts, W.T, out_dtype=torch.float32)
        comb2_k[(T, triton.cdiv(N, 1024))](Y, out, T, N, ADD=add, BN=1024)

    def _prefill_fwd(self, tok, pos0, last):
        """Чтение запроса своими ядрами (нормировка, поворот+кэш, внимание по 0..t), умножения — cuBLAS с
        входом из 2 частей bf16; словарь — только для слова last, через vocab8."""
        m = self.m
        T = tok.shape[0]
        f = lambda n: torch.empty(T, n, device="cuda")
        x, h, qkv, q, att, gu = f(D), f(D), f((H + 2 * KV) * HD), f(H * HD), f(H * HD), f(2 * FF)
        x.copy_(m.emb.index_select(0, tok).float())
        for l, w in enumerate(m.layers):
            Wn = self.Wn[l]
            rms_k[(T,)](x, w["ln1"], h, EPS, DD=D)
            self._mm2(h, Wn["qkv"], qkv, T)
            qk_rope_cache_k[(H + 2 * KV, T)](qkv, w["qn"], w["kn"], m.inv_freq, pos0, q,
                                              self.K[l], self.V[l], self.maxt, EPS, NH=H, NKV=KV, HDIM=HD, half=HD // 2)
            attn_k[(H, T)](q, self.K[l], self.V[l], pos0, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)
            self._mm2(att, Wn["o"], x, T, add=True)
            rms_k[(T,)](x, w["ln2"], h, EPS, DD=D)
            self._mm2(h, Wn["gu"], gu, T)
            self._mm2(gu, Wn["down"], x, T, add=True, mode=2)
        xl = x.index_select(0, last).contiguous()
        hl = self.h[:1]
        rms_k[(1,)](xl, m.norm, hl, EPS, DD=D)
        return self.v8(hl, 1)

    def _prefill_fwd_torch(self, tok, pos0, last):
        """Как GpuEngine._forward (w16s2), но словарь — только для слова last, через vocab8."""
        m = self.m
        T = tok.shape[0]
        G = H // KV
        pos = pos0 + torch.arange(T, device="cuda")
        fr = pos[:, None].float() * m.inv_freq[None]
        e = torch.cat((fr, fr), -1)
        cos, sin = e.cos(), e.sin()
        visible = torch.arange(self.maxt, device="cuda")[None, :] <= pos[:, None]
        bias = torch.zeros(T, self.maxt, device="cuda").masked_fill(~visible, float("-inf"))
        x = m.emb[tok].float()
        for l, w in enumerate(m.layers):
            h = self._rms(x, w["ln1"])
            qkv = self.mm(h, w["qkv"])
            q = self._rms(qkv[:, :H * HD].reshape(T, H, HD), w["qn"])
            k = self._rms(qkv[:, H * HD:(H + KV) * HD].reshape(T, KV, HD), w["kn"])
            v = qkv[:, (H + KV) * HD:].reshape(T, KV, HD)
            q = q * cos[:, None] + self._rot(q) * sin[:, None]
            k = k * cos[:, None] + self._rot(k) * sin[:, None]
            self.K[l].index_copy_(1, pos, k.transpose(0, 1))
            self.V[l].index_copy_(1, pos, v.transpose(0, 1))
            qg = q.view(T, KV, G, HD).permute(1, 2, 0, 3).reshape(KV, G * T, HD)
            s = (qg @ self.K[l].transpose(1, 2)) * HD ** -0.5
            s = s.view(KV, G, T, self.maxt) + bias
            a = F.softmax(s, dim=-1).view(KV, G * T, self.maxt)
            o = (a @ self.V[l]).view(KV, G, T, HD).permute(2, 0, 1, 3).reshape(T, H * HD)
            x = x + self.mm(o, w["o"])
            h = self._rms(x, w["ln2"])
            gu = self.mm(h, w["gu"])
            x = x + self.mm(F.silu(gu[:, :3072]) * gu[:, 3072:], w["down"])
        xl = x.index_select(0, last).contiguous()
        hl = self.h[:1]
        rms_k[(1,)](xl, m.norm, hl, EPS, DD=D)
        return self.v8(hl, 1)

    def _capture_prefill(self, T):
        tok = torch.zeros(T, dtype=torch.long, device="cuda")
        pos0 = torch.zeros(1, dtype=torch.long, device="cuda")
        last = torch.zeros(1, dtype=torch.long, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self._prefill_fwd(tok, pos0, last)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self._prefill_fwd(tok, pos0, last)
        self.pgraphs[T] = (g, tok, last, out, pos0)  # pos0 хранить: иначе память освободится, а граф читает её

    @torch.no_grad()
    def prefill(self, tokens):
        """Чтение запроса графом (дополнение до длины из PB). Возвращает [1] — слово после запроса."""
        P = len(tokens)
        T = next((b for b in PB if b >= P), None)
        if T is None:
            return fused_decode.GpuEngine.prefill(self, tokens)[-1:]
        if T not in self.pgraphs:
            self._capture_prefill(T)
        g, tok, last, out, _ = self.pgraphs[T]
        tok.zero_()
        tok[:P].copy_(torch.as_tensor(tokens, device="cuda"))
        last.fill_(P - 1)
        g.replay()
        return out

    @torch.no_grad()
    def generate(self, prompt, n):
        """Жадно n слов после prompt; первое — из чтения запроса, остальные n-1 шагов без синхронизации."""
        out = self.prefill(prompt)
        P = len(prompt)
        self.tok[:1].copy_(out[-1:])
        self.hist[P].copy_(out[-1])
        self.pos.fill_(P)
        for _ in range(n - 1):
            self.auto.replay()
        return self.hist[P:P + n].tolist()
