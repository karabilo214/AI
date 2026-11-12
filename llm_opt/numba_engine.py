"""Генерация Qwen3-0.6B на Numba: один проход слова целиком в скомпилированном коде, без torch и Python в цикле.

Веса в исходной раскладке [выход, вход]: каждая строка — скалярное произведение, потоки делят строки.
q/k/v и gate/up склеены, чтобы реже запускать потоки. Кэш внимания выделен заранее.
Словарь: полный (argmax по 151 936 словам) или угадыватель r=64 (step3_vocab.py).
"""
import glob
import math
import os

import numba as nb
import numpy as np
from safetensors.numpy import load_file

from core import SNAP

L, D, H, KV, HD, FF, V = 28, 1024, 16, 8, 128, 3072, 151936
EPS, THETA = 1e-6, 1_000_000.0


def load_weights():
    import torch
    from safetensors.torch import load_file as lt
    sd = {}
    for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
        sd.update(lt(f))
    g = lambda k: sd[k].float().numpy()
    p = lambda l, n: g(f"model.layers.{l}.{n}")
    w = {
        "emb": np.ascontiguousarray(g("model.embed_tokens.weight")),
        "norm": g("model.norm.weight"),
        "qkv": np.ascontiguousarray(np.stack([np.concatenate([p(l, "self_attn.q_proj.weight"), p(l, "self_attn.k_proj.weight"),
                                                              p(l, "self_attn.v_proj.weight")]) for l in range(L)])),
        "o": np.ascontiguousarray(np.stack([p(l, "self_attn.o_proj.weight") for l in range(L)])),
        "gu": np.ascontiguousarray(np.stack([np.concatenate([p(l, "mlp.gate_proj.weight"), p(l, "mlp.up_proj.weight")]) for l in range(L)])),
        "down": np.ascontiguousarray(np.stack([p(l, "mlp.down_proj.weight") for l in range(L)])),
        "ln1": np.stack([p(l, "input_layernorm.weight") for l in range(L)]),
        "ln2": np.stack([p(l, "post_attention_layernorm.weight") for l in range(L)]),
        "qn": np.stack([p(l, "self_attn.q_norm.weight") for l in range(L)]),
        "kn": np.stack([p(l, "self_attn.k_norm.weight") for l in range(L)]),
    }
    inv = (1.0 / THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD)).astype(np.float32)
    w["inv_freq"] = inv
    return w


@nb.njit(parallel=True, fastmath=True, cache=True)
def matvec(W, x, y):
    """y = W x, W [out, in]."""
    n_out, n_in = W.shape
    for o in nb.prange(n_out):
        s = np.float32(0.0)
        for i in range(n_in):
            s += W[o, i] * x[i]
        y[o] = s


@nb.njit(fastmath=True, cache=True)
def rmsnorm(x, w, out):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    r = 1.0 / math.sqrt(s / x.shape[0] + EPS)
    for i in range(x.shape[0]):
        out[i] = np.float32(x[i] * r) * w[i]


@nb.njit(fastmath=True, cache=True)
def norm_rope(v, w, cos, sin):
    """RMS-нормировка вектора головы и поворот (RoPE) на месте."""
    s = 0.0
    for i in range(HD):
        s += v[i] * v[i]
    r = 1.0 / math.sqrt(s / HD + EPS)
    tmp = np.empty(HD, np.float32)
    for i in range(HD):
        tmp[i] = np.float32(v[i] * r) * w[i]
    half = HD // 2
    for i in range(half):
        a, b = tmp[i], tmp[i + half]
        v[i] = a * cos[i] - b * sin[i]
        v[i + half] = b * cos[i + half] + a * sin[i + half]


@nb.njit(parallel=True, fastmath=True, cache=True)
def attention(q, Kc, Vc, n, out):
    """q [H*HD], Kc/Vc [KV, maxT, HD], n — сколько позиций в кэше. out [H*HD]."""
    scale = np.float32(1.0 / math.sqrt(HD))
    for h in nb.prange(H):
        g = h // (H // KV)
        sc = np.empty(n, np.float32)
        mx = np.float32(-1e30)
        for t in range(n):
            s = np.float32(0.0)
            for i in range(HD):
                s += q[h * HD + i] * Kc[g, t, i]
            s *= scale
            sc[t] = s
            if s > mx:
                mx = s
        tot = np.float32(0.0)
        for t in range(n):
            sc[t] = math.exp(sc[t] - mx)
            tot += sc[t]
        for i in range(HD):
            out[h * HD + i] = 0.0
        for t in range(n):
            p = sc[t] / tot
            for i in range(HD):
                out[h * HD + i] += p * Vc[g, t, i]


@nb.njit(fastmath=True, cache=True)
def layers_step(tok, pos, emb, qkv_w, o_w, gu_w, down_w, ln1, ln2, qn, kn, inv_freq, Kc, Vc, x):
    """Прогоняет одно слово через 28 слоёв, пишет k, v в кэш. x [D] — выход (до конечной нормировки)."""
    x[:] = emb[tok]
    h = np.empty(D, np.float32)
    qkv = np.empty(H * HD + 2 * KV * HD, np.float32)
    att = np.empty(H * HD, np.float32)
    tmp = np.empty(D, np.float32)
    gu = np.empty(2 * FF, np.float32)
    mm = np.empty(FF, np.float32)
    cos = np.empty(HD, np.float32)
    sin = np.empty(HD, np.float32)
    for i in range(HD // 2):
        a = np.float32(pos) * inv_freq[i]
        cos[i] = math.cos(a); cos[i + HD // 2] = cos[i]
        sin[i] = math.sin(a); sin[i + HD // 2] = sin[i]
    for l in range(L):
        rmsnorm(x, ln1[l], h)
        matvec(qkv_w[l], h, qkv)
        for hh in range(H):
            norm_rope(qkv[hh * HD:(hh + 1) * HD], qn[l], cos, sin)
        ko = H * HD
        vo = H * HD + KV * HD
        for g in range(KV):
            norm_rope(qkv[ko + g * HD: ko + (g + 1) * HD], kn[l], cos, sin)
            Kc[l, g, pos, :] = qkv[ko + g * HD: ko + (g + 1) * HD]
            Vc[l, g, pos, :] = qkv[vo + g * HD: vo + (g + 1) * HD]
        attention(qkv[:H * HD], Kc[l], Vc[l], pos + 1, att)
        matvec(o_w[l], att, tmp)
        x += tmp
        rmsnorm(x, ln2[l], h)
        matvec(gu_w[l], h, gu)
        for i in range(FF):
            g_ = gu[i]
            mm[i] = g_ / (1.0 + math.exp(-g_)) * gu[FF + i]
        matvec(down_w[l], mm, tmp)
        x += tmp


@nb.njit(parallel=True, fastmath=True, cache=True)
def argmax_rows(W, x):
    """Номер строки W с наибольшим W[row]·x (полный словарь). Потоки по кускам строк."""
    n = W.shape[0]
    nt = 64
    best_v = np.full(nt, -1e30, np.float32)
    best_i = np.zeros(nt, np.int64)
    chunk = (n + nt - 1) // nt
    for c in nb.prange(nt):
        for o in range(c * chunk, min(n, (c + 1) * chunk)):
            s = np.float32(0.0)
            for i in range(x.shape[0]):
                s += W[o, i] * x[i]
            if s > best_v[c]:
                best_v[c] = s
                best_i[c] = o
    return best_i[np.argmax(best_v)]


@nb.njit(parallel=True, fastmath=True, cache=True)
def predictor_argmax(emb, At, Bt, x, S):
    """Угадыватель: z = A^T x (r), грубые оценки Bt z по всем словам, S лучших, точная оценка среди них."""
    r = At.shape[0]
    z = np.empty(r, np.float32)
    matvec(At, x, z)
    ap = np.empty(Bt.shape[0], np.float32)
    matvec(Bt, z, ap)
    thr = np.partition(ap, ap.shape[0] - S)[ap.shape[0] - S]  # S-е по величине
    cand = np.nonzero(ap >= thr)[0]
    ex = np.empty(cand.shape[0], np.float32)
    for j in nb.prange(cand.shape[0]):
        s = np.float32(0.0)
        for i in range(x.shape[0]):
            s += emb[cand[j], i] * x[i]
        ex[j] = s
    return cand[np.argmax(ex)]


class Engine:
    def __init__(self, max_t=1024, predictor=None):
        self.w = load_weights()
        self.K = np.zeros((L, KV, max_t, HD), np.float32)
        self.V = np.zeros((L, KV, max_t, HD), np.float32)
        self.x = np.empty(D, np.float32)
        self.hn = np.empty(D, np.float32)
        if predictor is not None:
            import torch
            ab = torch.load(predictor)
            self.At = np.ascontiguousarray(ab["A"].numpy().T)  # [r, D]
            self.Bt = np.ascontiguousarray(ab["B"].numpy().T)  # [V, r]

    def layers(self, tok, pos):
        w = self.w
        layers_step(tok, pos, w["emb"], w["qkv"], w["o"], w["gu"], w["down"], w["ln1"], w["ln2"], w["qn"], w["kn"],
                    w["inv_freq"], self.K, self.V, self.x)
        rmsnorm(self.x, w["norm"], self.hn)
        return self.hn

    def step(self, tok, pos, S=0):
        h = self.layers(tok, pos)
        if S == 0:
            return int(argmax_rows(self.w["emb"], h))
        return int(predictor_argmax(self.w["emb"], self.At, self.Bt, h, S))
