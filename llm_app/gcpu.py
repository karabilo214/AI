"""Слой GGUF на процессоре (Numba): веса читаются прямо из файла модели (memmap), счёт fp32.
Блок весов распаковывается один раз и умножается на все слова пачки. Масштабы Q4_K/Q6_K раскладываются при
загрузке в небольшие массивы float32; сами 4/6-битные числа — из файла без копии."""
import ctypes
import os
import sys

import numba as nb
import numpy as np

from .cpu_layer import qk_rope_attn, rms_rows, silu_mul
from .gguf import _scale_min_k4


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def sub_sums(X, SX):
    """SX[t, j] = сумма X[t, 32j : 32j+32]."""
    T, K = X.shape
    for j in nb.prange(K // 32):
        for t in range(T):
            s = np.float32(0.0)
            for l in range(32):
                s += X[t, j * 32 + l]
            SX[t, j] = s


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def q4k_mm(QS, DSC, DMN, X, SX, Y, add):
    """QS [N, nb, 128] uint8 (из файла); DSC = d*sc, DMN = dmin*m — [N, nb, 8] float32. Y[t] (+)= W X[t].
    Слово пачки — снаружи: строка весов (~3 КБ) остаётся в кэше, внутренний цикл по 32 числам векторизуется.
    (Вариант «распаковать подблок один раз на все слова» на Numba оказался медленнее: 15.6/75.6 мс против 11.5/57.)"""
    N, NB = QS.shape[0], QS.shape[1]
    T = X.shape[0]
    for r in nb.prange(N):
        for t in range(T):
            acc = np.float32(0.0)
            for b in range(NB):
                for j in range(4):
                    k0 = b * 256 + j * 64
                    sl = np.float32(0.0)
                    sh = np.float32(0.0)
                    for l in range(32):
                        q = QS[r, b, j * 32 + l]
                        sl += np.float32(q & 15) * X[t, k0 + l]
                        sh += np.float32(q >> 4) * X[t, k0 + 32 + l]
                    s0 = 2 * j
                    acc += DSC[r, b, s0] * sl - DMN[r, b, s0] * SX[t, b * 8 + s0] \
                        + DSC[r, b, s0 + 1] * sh - DMN[r, b, s0 + 1] * SX[t, b * 8 + s0 + 1]
            if add:
                Y[t, r] += acc
            else:
                Y[t, r] = acc


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def q6k_mm(QL, QH, DS, X, Y, add):
    """QL [N, nb, 128], QH [N, nb, 64] uint8 (из файла); DS = d*sc — [N, nb, 16] float32."""
    N, NB = QL.shape[0], QL.shape[1]
    T = X.shape[0]
    for r in nb.prange(N):
        for t in range(T):
            acc = np.float32(0.0)
            for b in range(NB):
                for h in range(2):
                    base = b * 256 + h * 128
                    for s in range(2):
                        a0 = np.float32(0.0)
                        a1 = np.float32(0.0)
                        a2 = np.float32(0.0)
                        a3 = np.float32(0.0)
                        for i in range(16):
                            l = s * 16 + i
                            L0 = np.int32(QL[r, b, h * 64 + l])
                            L1 = np.int32(QL[r, b, h * 64 + 32 + l])
                            Hh = np.int32(QH[r, b, h * 32 + l])
                            a0 += np.float32(((L0 & 15) | ((Hh & 3) << 4)) - 32) * X[t, base + l]
                            a1 += np.float32(((L1 & 15) | (((Hh >> 2) & 3) << 4)) - 32) * X[t, base + 32 + l]
                            a2 += np.float32(((L0 >> 4) | (((Hh >> 4) & 3) << 4)) - 32) * X[t, base + 64 + l]
                            a3 += np.float32(((L1 >> 4) | (((Hh >> 6) & 3) << 4)) - 32) * X[t, base + 96 + l]
                        g = h * 8 + s
                        acc += DS[r, b, g] * a0 + DS[r, b, g + 2] * a1 + DS[r, b, g + 4] * a2 + DS[r, b, g + 6] * a3
            if add:
                Y[t, r] += acc
            else:
                Y[t, r] = acc


@nb.njit(parallel=True, fastmath=True, cache=True, nogil=True)
def f32_mm(W, X, Y, add):
    N, K = W.shape
    T = X.shape[0]
    for r in nb.prange(N):
        for t in range(T):
            s = np.float32(0.0)
            for k in range(K):
                s += W[r, k] * X[t, k]
            if add:
                Y[t, r] += s
            else:
                Y[t, r] = s


_DLL = None


def _dll():
    """Ядра на C++ (cpp/qmm.dll, AVX-512, свой пул потоков); None — тогда Numba."""
    global _DLL
    if _DLL is None and sys.platform == "win32" and os.environ.get("LLM_NO_CPP") != "1":
        p = os.path.join(os.path.dirname(__file__), "cpp", "qmm.dll")
        if os.path.exists(p):
            d = ctypes.CDLL(p)
            for f in ("q4k_mm", "q6k_mm", "f16_mm"):
                getattr(d, f).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            d.set_threads(int(os.environ.get("LLM_CPP_THREADS", "16")))
            _DLL = d
    return _DLL or None


class CpuMat:
    """Матрица GGUF для процессора."""

    def __init__(self, g, name):
        buf, N, K, t = g.raw(name)
        self.t, self.N, self.K = t, N, K
        a = np.asarray(buf)
        self.raw = a  # байты из файла (memmap) — для ядер C++
        nb_ = K // 256
        if t == "Q4_K":
            blk = a.reshape(N, nb_, 144)
            d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)
            dm = blk[:, :, 2:4].copy().view(np.float16).astype(np.float32)
            sc, mn = _scale_min_k4(np.ascontiguousarray(blk[:, :, 4:16]).reshape(-1, 12))
            self.DSC = (d * sc.reshape(N, nb_, 8)).astype(np.float32)
            self.DMN = (dm * mn.reshape(N, nb_, 8)).astype(np.float32)
            self.QS = blk[:, :, 16:]
        elif t == "Q6_K":
            blk = a.reshape(N, nb_, 210)
            d = blk[:, :, 208:210].copy().view(np.float16).astype(np.float32)
            sc = np.ascontiguousarray(blk[:, :, 192:208]).view(np.int8).astype(np.float32)
            self.DS = (d * sc).astype(np.float32)
            self.QL, self.QH = blk[:, :, :128], blk[:, :, 128:192]
        else:  # F16 / F32 — в float32 (такие матрицы небольшие)
            from .gguf import dequant
            self.W = dequant(a, t, N * K).reshape(N, K).astype(np.float32)

    def mm(self, X, Y, SX, add=False):
        d = _dll()
        if d is not None and self.t in ("Q4_K", "Q6_K", "F16"):
            f = {"Q4_K": d.q4k_mm, "Q6_K": d.q6k_mm, "F16": d.f16_mm}[self.t]
            assert X.flags.c_contiguous and X.dtype == np.float32 and Y.dtype == np.float32
            f(self.raw.ctypes.data, self.N, self.K, X.ctypes.data, X.shape[0], Y.ctypes.data, Y.strides[0] // 4, int(add))
            return
        if self.t == "Q4_K":
            q4k_mm(self.QS, self.DSC, self.DMN, X, SX, Y, add)
        elif self.t == "Q6_K":
            q6k_mm(self.QL, self.QH, self.DS, X, Y, add)
        else:
            f32_mm(self.W, X, Y, add)


class GCpuLayer:
    """Слой Qwen3 из GGUF на процессоре. Кэш k, v — в RAM (fp32)."""

    def __init__(self, g, i, small, H, KV, HD, FF, eps, maxt):
        n = lambda s: f"blk.{i}.{s}.weight"
        self.m = {k: CpuMat(g, n(v)) for k, v in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"),
                                                   ("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down"))}
        self.s = {k: v.float().cpu().numpy() for k, v in small.items()}
        self.H, self.KV, self.HD, self.FF, self.eps = H, KV, HD, FF, eps
        self.K = np.zeros((KV, maxt, HD), np.float32)
        self.V = np.zeros_like(self.K)

    def __call__(self, X, pos0, inv):
        T, D = X.shape
        H, KV, HD, FF, m = self.H, self.KV, self.HD, self.FF, self.m
        h = np.empty_like(X)
        rms_rows(X, self.s["ln1"], h, self.eps)
        sx = np.empty((T, D // 32), np.float32)
        sub_sums(h, sx)
        qkv = np.empty((T, (H + 2 * KV) * HD), np.float32)
        m["q"].mm(h, qkv[:, :H * HD], sx)
        m["k"].mm(h, qkv[:, H * HD:(H + KV) * HD], sx)
        m["v"].mm(h, qkv[:, (H + KV) * HD:], sx)
        att = np.empty((T, H * HD), np.float32)
        qk_rope_attn(qkv, self.s["qn"], self.s["kn"], inv, pos0, self.K, self.V, att, H, KV, HD, self.eps)
        sa = np.empty((T, att.shape[1] // 32), np.float32)
        sub_sums(att, sa)
        m["o"].mm(att, X, sa, True)
        rms_rows(X, self.s["ln2"], h, self.eps)
        sub_sums(h, sx)
        gu = np.empty((T, 2 * FF), np.float32)
        m["gate"].mm(h, gu[:, :FF], sx)
        m["up"].mm(h, gu[:, FF:], sx)
        mm_ = np.empty((T, FF), np.float32)
        silu_mul(gu, mm_, FF)
        sm = np.empty((T, FF // 32), np.float32)
        sub_sums(mm_, sm)
        m["down"].mm(mm_, X, sm, True)
