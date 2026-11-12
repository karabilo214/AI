"""Слой Qwen3 на процессоре (Numba): веса bf16 (биты uint16) из RAM, счёт fp32 — как на видеокарте (точно).
Пачка T слов: строка веса читается из памяти один раз на все слова. Кэш k, v слоя — в RAM (fp32)."""
import math

import numba as nb
import numpy as np
import torch
from numba.core import types
from numba.extending import intrinsic


@intrinsic
def _u32_as_f32(typingctx, x):
    sig = types.float32(types.uint32)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.float32))
    return sig, codegen


@nb.njit(inline="always")
def bf(u):
    return _u32_as_f32(np.uint32(u) << np.uint32(16))


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def matmat16(W, X, Y, add):
    """Y[t] (+)= W X[t]; W [out, in] uint16 (bf16)."""
    n_out, n_in = W.shape
    T = X.shape[0]
    for o in nb.prange(n_out):
        for t in range(T):
            s = np.float32(0.0)
            for i in range(n_in):
                s += bf(W[o, i]) * X[t, i]
            if add:
                Y[t, o] += s
            else:
                Y[t, o] = s


@nb.njit(fastmath=False, cache=True, nogil=True)
def rms_rows(X, w, out, eps):
    T, D = X.shape
    for t in range(T):
        s = np.float32(0.0)
        for i in range(D):
            s += X[t, i] * X[t, i]
        r = np.float32(1.0) / np.float32(math.sqrt(s / np.float32(D) + np.float32(eps)))
        for i in range(D):
            out[t, i] = w[i] * (X[t, i] * r)


@nb.njit(parallel=True, fastmath=False, cache=True, nogil=True)
def qk_rope_attn(QKV, qn, kn, inv, pos0, Kc, Vc, out, H, KV, HD, eps):
    """Нормировка голов q, k, поворот, запись k, v в кэш (позиции pos0..pos0+T-1), внимание по 0..pos0+t."""
    T = QKV.shape[0]
    half = HD // 2
    G = H // KV
    # k, v в кэш
    for t in range(T):
        p = pos0 + t
        for g in range(KV):
            b = (H + g) * HD
            s = np.float32(0.0)
            for d in range(HD):
                s += QKV[t, b + d] * QKV[t, b + d]
            r = np.float32(1.0) / np.float32(math.sqrt(s / np.float32(HD) + np.float32(eps)))
            for d in range(half):
                ang = np.float32(p) * inv[d]
                c, sn = np.float32(math.cos(ang)), np.float32(math.sin(ang))
                a = kn[d] * (QKV[t, b + d] * r)
                bb = kn[d + half] * (QKV[t, b + d + half] * r)
                Kc[g, p, d] = a * c - bb * sn
                Kc[g, p, d + half] = bb * c + a * sn
            vb = (H + KV + g) * HD
            for d in range(HD):
                Vc[g, p, d] = QKV[t, vb + d]
    for th in nb.prange(T * H):
        t = th // H
        h = th % H
        g = h // G
        p = pos0 + t
        q = np.empty(HD, np.float32)
        b = h * HD
        s = np.float32(0.0)
        for d in range(HD):
            s += QKV[t, b + d] * QKV[t, b + d]
        r = np.float32(1.0) / np.float32(math.sqrt(s / np.float32(HD) + np.float32(eps)))
        for d in range(half):
            ang = np.float32(p) * inv[d]
            c, sn = np.float32(math.cos(ang)), np.float32(math.sin(ang))
            a = qn[d] * (QKV[t, b + d] * r)
            bb = qn[d + half] * (QKV[t, b + d + half] * r)
            q[d] = a * c - bb * sn
            q[d + half] = bb * c + a * sn
        scale = np.float32(1.0) / np.float32(math.sqrt(HD))
        n = p + 1
        sc = np.empty(n, np.float32)
        m = np.float32(-1e30)
        for j in range(n):
            s = np.float32(0.0)
            for d in range(HD):
                s += q[d] * Kc[g, j, d]
            s *= scale
            sc[j] = s
            if s > m:
                m = s
        l = np.float32(0.0)
        for j in range(n):
            sc[j] = np.float32(math.exp(sc[j] - m))
            l += sc[j]
        for d in range(HD):
            out[t, h * HD + d] = 0.0
        for j in range(n):
            w = sc[j] / l
            for d in range(HD):
                out[t, h * HD + d] += w * Vc[g, j, d]


@nb.njit(parallel=True, fastmath=False, cache=True, nogil=True)
def silu_mul(GU, M, FF):
    T = GU.shape[0]
    for t in nb.prange(T):
        for i in range(FF):
            g = GU[t, i]
            M[t, i] = g / (np.float32(1.0) + np.float32(math.exp(-g))) * GU[t, FF + i]


class CpuLayer:
    """Слой на процессоре. mats: q, k, v, o, gate, up, down — bf16 как uint16 (массивы или memmap файла модели).
    small: ln1, ln2, qn, kn (torch bf16)."""

    def __init__(self, mats, small, ck, maxt):
        self.W = {k: np.asarray(v) for k, v in mats.items()}  # memmap -> ndarray (без копии)
        self.s = {k: small[k].float().numpy() for k in ("ln1", "ln2", "qn", "kn")}
        self.H, self.KV, self.HD, self.FF, self.eps, self.D = ck.H, ck.KV, ck.HD, ck.FF, ck.eps, ck.D
        self.K = np.zeros((self.KV, maxt, self.HD), np.float32)
        self.V = np.zeros_like(self.K)

    def nbytes(self):
        return sum(m.size * 2 for m in self.W.values())

    def __call__(self, X, pos0, inv):
        """X [T, D] fp32 numpy — меняется на месте."""
        T = X.shape[0]
        H, KV, HD, FF = self.H, self.KV, self.HD, self.FF
        W = self.W
        h = np.empty_like(X)
        rms_rows(X, self.s["ln1"], h, self.eps)
        qkv = np.empty((T, (H + 2 * KV) * HD), np.float32)
        matmat16(W["q"], h, qkv[:, :H * HD], False)
        matmat16(W["k"], h, qkv[:, H * HD:(H + KV) * HD], False)
        matmat16(W["v"], h, qkv[:, (H + KV) * HD:], False)
        att = np.empty((T, H * HD), np.float32)
        qk_rope_attn(qkv, self.s["qn"], self.s["kn"], inv, pos0, self.K, self.V, att, H, KV, HD, self.eps)
        matmat16(W["o"], att, X, True)
        rms_rows(X, self.s["ln2"], h, self.eps)
        gu = np.empty((T, 2 * FF), np.float32)
        matmat16(W["gate"], h, gu[:, :FF], False)
        matmat16(W["up"], h, gu[:, FF:], False)
        m = np.empty((T, FF), np.float32)
        silu_mul(gu, m, FF)
        matmat16(W["down"], m, X, True)
