"""Версия 2: веса в исходном bf16 (перевод в fp32 при умножении, точный), внимание с локальными накопителями,
по 4 строки матрицы за проход.

Генерация Qwen3-0.6B на Numba: один проход слова целиком в скомпилированном коде, без torch и Python в цикле.

Веса в исходной раскладке [выход, вход]: каждая строка — скалярное произведение, потоки делят строки.
q/k/v и gate/up склеены, чтобы реже запускать потоки. Кэш внимания выделен заранее.
Словарь: полный (argmax по 151 936 словам) или угадыватель r=64 (step3_vocab.py).
"""
import glob
import math
import os

import llvmlite.ir as lir
import numba as nb
import numpy as np
from numba import types
from numba.extending import intrinsic

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
    gb = lambda k: sd[k].view(torch.int16).numpy().view(np.uint16)  # биты bf16 как есть
    p = lambda l, n: g(f"model.layers.{l}.{n}")
    pb = lambda l, n: gb(f"model.layers.{l}.{n}")
    w = {
        "emb": np.ascontiguousarray(g("model.embed_tokens.weight")),
        "emb16": np.ascontiguousarray(gb("model.embed_tokens.weight")),
        "norm": g("model.norm.weight"),
        "qkv": np.ascontiguousarray(np.stack([np.concatenate([pb(l, "self_attn.q_proj.weight"), pb(l, "self_attn.k_proj.weight"),
                                                              pb(l, "self_attn.v_proj.weight")]) for l in range(L)])),
        "o": np.ascontiguousarray(np.stack([pb(l, "self_attn.o_proj.weight") for l in range(L)])),
        "gu": np.ascontiguousarray(np.stack([np.concatenate([pb(l, "mlp.gate_proj.weight"), pb(l, "mlp.up_proj.weight")]) for l in range(L)])),
        "down": np.ascontiguousarray(np.stack([pb(l, "mlp.down_proj.weight") for l in range(L)])),
        "ln1": np.stack([p(l, "input_layernorm.weight") for l in range(L)]),
        "ln2": np.stack([p(l, "post_attention_layernorm.weight") for l in range(L)]),
        "qn": np.stack([p(l, "self_attn.q_norm.weight") for l in range(L)]),
        "kn": np.stack([p(l, "self_attn.k_norm.weight") for l in range(L)]),
    }
    inv = (1.0 / THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD)).astype(np.float32)
    w["inv_freq"] = inv
    return w


@intrinsic
def u32_as_f32(typingctx, x):
    """Переосмыслить биты uint32 как float32 (без преобразования значения)."""
    sig = types.float32(types.uint32)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], lir.FloatType())
    return sig, codegen


@nb.njit(inline="always")
def bf(u):
    return u32_as_f32(np.uint32(u) << np.uint32(16))


@nb.njit(parallel=True, fastmath=True, cache=True)
def matvec16(W, x, y):
    """y = W x, W [out, in] в битах bf16; по 4 строки за проход (x читается один раз на 4 строки)."""
    n_out, n_in = W.shape
    nb4 = n_out // 4
    for ob in nb.prange(nb4):
        o = ob * 4
        s0 = np.float32(0.0); s1 = np.float32(0.0); s2 = np.float32(0.0); s3 = np.float32(0.0)
        for i in range(n_in):
            xi = x[i]
            s0 += bf(W[o, i]) * xi
            s1 += bf(W[o + 1, i]) * xi
            s2 += bf(W[o + 2, i]) * xi
            s3 += bf(W[o + 3, i]) * xi
        y[o] = s0; y[o + 1] = s1; y[o + 2] = s2; y[o + 3] = s3


@nb.njit(parallel=True, fastmath=True, cache=True)
def matvec16_rows(W, x, y):
    """Вариант: по одной строке."""
    n_out, n_in = W.shape
    for o in nb.prange(n_out):
        s = np.float32(0.0)
        for i in range(n_in):
            s += bf(W[o, i]) * x[i]
        y[o] = s


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
        qh = q[h * HD:(h + 1) * HD].copy()
        sc = np.empty(n, np.float32)
        mx = np.float32(-1e30)
        for t in range(n):
            kt = Kc[g, t]
            s = np.float32(0.0)
            for i in range(HD):
                s += qh[i] * kt[i]
            s *= scale
            sc[t] = s
            if s > mx:
                mx = s
        tot = np.float32(0.0)
        for t in range(n):
            e = np.float32(math.exp(sc[t] - mx))
            sc[t] = e
            tot += e
        acc = np.zeros(HD, np.float32)
        for t in range(n):
            p = sc[t] / tot
            vt = Vc[g, t]
            for i in range(HD):
                acc[i] += p * vt[i]
        out[h * HD:(h + 1) * HD] = acc


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
        matvec16_rows(qkv_w[l], h, qkv)
        for hh in range(H):
            norm_rope(qkv[hh * HD:(hh + 1) * HD], qn[l], cos, sin)
        ko = H * HD
        vo = H * HD + KV * HD
        for g in range(KV):
            norm_rope(qkv[ko + g * HD: ko + (g + 1) * HD], kn[l], cos, sin)
            Kc[l, g, pos, :] = qkv[ko + g * HD: ko + (g + 1) * HD]
            Vc[l, g, pos, :] = qkv[vo + g * HD: vo + (g + 1) * HD]
        attention(qkv[:H * HD], Kc[l], Vc[l], pos + 1, att)
        matvec16_rows(o_w[l], att, tmp)
        x += tmp
        rmsnorm(x, ln2[l], h)
        matvec16_rows(gu_w[l], h, gu)
        for i in range(FF):
            g_ = gu[i]
            mm[i] = g_ / (1.0 + math.exp(-g_)) * gu[FF + i]
        matvec16_rows(down_w[l], mm, tmp)
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
def argmax_rows16(W, x):
    n = W.shape[0]
    nt = 64
    best_v = np.full(nt, -1e30, np.float32)
    best_i = np.zeros(nt, np.int64)
    chunk = (n + nt - 1) // nt
    for c in nb.prange(nt):
        for o in range(c * chunk, min(n, (c + 1) * chunk)):
            s = np.float32(0.0)
            for i in range(x.shape[0]):
                s += bf(W[o, i]) * x[i]
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


@nb.njit(parallel=True, fastmath=True, cache=True)
def predictor_argmax16(emb16, At, Bt, x, S):
    r = At.shape[0]
    z = np.empty(r, np.float32)
    matvec(At, x, z)
    ap = np.empty(Bt.shape[0], np.float32)
    matvec(Bt, z, ap)
    thr = np.partition(ap, ap.shape[0] - S)[ap.shape[0] - S]
    cand = np.nonzero(ap >= thr)[0]
    ex = np.empty(cand.shape[0], np.float32)
    for j in nb.prange(cand.shape[0]):
        s = np.float32(0.0)
        for i in range(x.shape[0]):
            s += bf(emb16[cand[j], i]) * x[i]
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
            return int(argmax_rows16(self.w["emb16"], h))
        return int(predictor_argmax16(self.w["emb16"], self.At, self.Bt, h, S))


# ================= чтение текста пачкой (prefill) =================
# Большие умножения — numpy (Apple Accelerate, ~1.7 TFLOPS fp32), мелкие операции — Numba.

@nb.njit(parallel=True, fastmath=True, cache=True)
def rms_rows(X, w, out):
    for t in nb.prange(X.shape[0]):
        rmsnorm(X[t], w, out[t])


@nb.njit(parallel=True, fastmath=True, cache=True)
def qk_norm_rope_rows(QKV, qn, kn, pos0, inv_freq):
    """Нормировка голов q, k и поворот по позиции pos0 + t, на месте."""
    for t in nb.prange(QKV.shape[0]):
        cos = np.empty(HD, np.float32)
        sin = np.empty(HD, np.float32)
        for i in range(HD // 2):
            a = np.float32(pos0 + t) * inv_freq[i]
            cos[i] = math.cos(a); cos[i + HD // 2] = cos[i]
            sin[i] = math.sin(a); sin[i + HD // 2] = sin[i]
        row = QKV[t]
        for hh in range(H):
            norm_rope(row[hh * HD:(hh + 1) * HD], qn, cos, sin)
        for g in range(KV):
            norm_rope(row[H * HD + g * HD: H * HD + (g + 1) * HD], kn, cos, sin)


@nb.njit(parallel=True, fastmath=True, cache=True)
def causal_softmax(S, pos0, scale):
    """S [T, N]: строка t видит позиции 0..pos0+t; на месте превращается в вероятности."""
    T, N = S.shape
    for t in nb.prange(T):
        n = pos0 + t + 1
        mx = np.float32(-1e30)
        for j in range(n):
            v = S[t, j] * scale
            S[t, j] = v
            if v > mx:
                mx = v
        tot = np.float32(0.0)
        for j in range(n):
            e = np.float32(math.exp(S[t, j] - mx))
            S[t, j] = e
            tot += e
        inv = np.float32(1.0) / tot
        for j in range(n):
            S[t, j] *= inv
        for j in range(n, N):
            S[t, j] = 0.0


@nb.njit(parallel=True, fastmath=True, cache=True)
def silu_mul(GU, M):
    for t in nb.prange(GU.shape[0]):
        for i in range(FF):
            g = GU[t, i]
            M[t, i] = g / (np.float32(1.0) + np.float32(math.exp(-g))) * GU[t, FF + i]


@nb.njit(parallel=True, fastmath=True, cache=True)
def causal_softmax_g(S, T, pos0, scale):
    """S [G*T, N]: строка r — слово t = r % T, видит позиции 0..pos0+t; на месте в вероятности."""
    R, N = S.shape
    for r in nb.prange(R):
        n = pos0 + (r % T) + 1
        mx = np.float32(-1e30)
        for j in range(n):
            v = S[r, j] * scale
            S[r, j] = v
            if v > mx:
                mx = v
        tot = np.float32(0.0)
        for j in range(n):
            e = np.float32(math.exp(S[r, j] - mx))
            S[r, j] = e
            tot += e
        inv = np.float32(1.0) / tot
        for j in range(n):
            S[r, j] *= inv
        for j in range(n, N):
            S[r, j] = 0.0


def _ensure_fp32_T(self):
    """Копии матриц слоёв в fp32 в виде W^T [вход, выход] для Accelerate (≈2.4 ГБ)."""
    if hasattr(self, "wT"):
        return
    import torch
    conv = lambda a: np.ascontiguousarray(torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float().numpy().T)
    self.wT = {n: [conv(self.w[n][l]) for l in range(L)] for n in ("qkv", "o", "gu", "down")}


def _omp_blocktime(ms):
    """Сколько мс потоки OpenMP ждут работу активно. Для пачки 0 (иначе мешают Accelerate), для генерации 200."""
    try:
        import ctypes
        import os
        lib = ctypes.CDLL(os.path.join(os.environ.get("DYLD_LIBRARY_PATH", ""), "libomp.dylib"))
        lib.kmp_set_blocktime(ctypes.c_int(ms))
    except OSError:
        pass


def prefill(self, tokens, pos0=0):
    """Прогоняет пачку слов через все слои, заполняет кэш k, v. Возвращает конечное состояние последнего слова."""
    _ensure_fp32_T(self)
    if os.environ.get("NUMBA_THREADING_LAYER") == "omp":
        _omp_blocktime(0)
    w, wT = self.w, self.wT
    T = len(tokens)
    N = pos0 + T
    X = w["emb"][np.asarray(tokens)].astype(np.float32)
    Hn = np.empty_like(X)
    att = np.empty((T, H * HD), np.float32)
    M = np.empty((T, FF), np.float32)
    scale = np.float32(1.0 / math.sqrt(HD))
    ko, vo = H * HD, H * HD + KV * HD
    G = H // KV
    for l in range(L):
        rms_rows(X, w["ln1"][l], Hn)
        QKV = Hn @ wT["qkv"][l]
        qk_norm_rope_rows(QKV, w["qn"][l], w["kn"][l], pos0, w["inv_freq"])
        for g in range(KV):
            self.K[l, g, pos0:N] = QKV[:, ko + g * HD: ko + (g + 1) * HD]
            self.V[l, g, pos0:N] = QKV[:, vo + g * HD: vo + (g + 1) * HD]
        for g in range(KV):
            # две головы группы одним умножением: [G*T, HD]
            Qg = QKV[:, g * G * HD:(g + 1) * G * HD].reshape(T, G, HD).transpose(1, 0, 2).reshape(G * T, HD)
            S = Qg @ self.K[l, g, :N].T
            causal_softmax_g(S, T, pos0, scale)
            out = S @ self.V[l, g, :N]  # [G*T, HD]
            att[:, g * G * HD:(g + 1) * G * HD] = out.reshape(G, T, HD).transpose(1, 0, 2).reshape(T, G * HD)
        X += att @ wT["o"][l]
        rms_rows(X, w["ln2"][l], Hn)
        GU = Hn @ wT["gu"][l]
        silu_mul(GU, M)
        X += M @ wT["down"][l]
    self.x[:] = X[-1]
    rmsnorm(self.x, w["norm"], self.hn)
    if os.environ.get("NUMBA_THREADING_LAYER") == "omp":
        _omp_blocktime(200)
    return self.hn


Engine.prefill = prefill


# ================= проверка нескольких слов одним проходом (для угадывания вперёд) =================

@nb.njit(parallel=True, fastmath=True, cache=True)
def matmat16(W, X, Y):
    """Y[t] = W X[t] для всех t; W [out, in] в bf16. Строка W читается из памяти один раз на все T слов."""
    n_out, n_in = W.shape
    T = X.shape[0]
    for o in nb.prange(n_out):
        for t in range(T):
            s = np.float32(0.0)
            for i in range(n_in):
                s += bf(W[o, i]) * X[t, i]
            Y[t, o] = s


@nb.njit(parallel=True, fastmath=True, cache=True)
def argmax_rows16_multi(W, X):
    """Для каждого t номер строки W с наибольшим W[row]·X[t] (полный словарь, одно чтение на все t)."""
    n = W.shape[0]
    T = X.shape[0]
    nt = 64
    best_v = np.full((nt, T), -1e30, np.float32)
    best_i = np.zeros((nt, T), np.int64)
    chunk = (n + nt - 1) // nt
    for c in nb.prange(nt):
        for o in range(c * chunk, min(n, (c + 1) * chunk)):
            for t in range(T):
                s = np.float32(0.0)
                for i in range(X.shape[1]):
                    s += bf(W[o, i]) * X[t, i]
                if s > best_v[c, t]:
                    best_v[c, t] = s
                    best_i[c, t] = o
    out = np.empty(T, np.int64)
    for t in range(T):
        out[t] = best_i[np.argmax(best_v[:, t]), t]
    return out


def verify(self, tokens, pos0):
    """Прогоняет T слов (позиции pos0..pos0+T-1), пишет кэш, возвращает самое вероятное следующее слово после каждого."""
    w = self.w
    T = len(tokens)
    N = pos0 + T
    X = w["emb"][np.asarray(tokens)].astype(np.float32)
    Hn = np.empty_like(X)
    QKV = np.empty((T, H * HD + 2 * KV * HD), np.float32)
    att = np.empty((T, H * HD), np.float32)
    tmp = np.empty((T, D), np.float32)
    GU = np.empty((T, 2 * FF), np.float32)
    M = np.empty((T, FF), np.float32)
    scale = np.float32(1.0 / math.sqrt(HD))
    ko, vo = H * HD, H * HD + KV * HD
    G = H // KV
    for l in range(L):
        rms_rows(X, w["ln1"][l], Hn)
        matmat16(w["qkv"][l], Hn, QKV)
        qk_norm_rope_rows(QKV, w["qn"][l], w["kn"][l], pos0, w["inv_freq"])
        for g in range(KV):
            self.K[l, g, pos0:N] = QKV[:, ko + g * HD: ko + (g + 1) * HD]
            self.V[l, g, pos0:N] = QKV[:, vo + g * HD: vo + (g + 1) * HD]
        attention_all(QKV, self.K[l], self.V[l], pos0, att)
        matmat16(w["o"][l], att, tmp)
        X += tmp
        rms_rows(X, w["ln2"][l], Hn)
        matmat16(w["gu"][l], Hn, GU)
        silu_mul(GU, M)
        matmat16(w["down"][l], M, tmp)
        X += tmp
    Xn = np.empty_like(X)
    rms_rows(X, w["norm"], Xn)
    return argmax_rows16_multi(w["emb16"], Xn)


@nb.njit(parallel=True, fastmath=True, cache=True)
def attention_rows(Q, Kc, Vc, pos0, out):
    """Внимание одной головы для T слов: слово t видит позиции 0..pos0+t."""
    T = Q.shape[0]
    scale = np.float32(1.0 / math.sqrt(HD))
    for t in nb.prange(T):
        n = pos0 + t + 1
        sc = np.empty(n, np.float32)
        mx = np.float32(-1e30)
        for j in range(n):
            s = np.float32(0.0)
            for i in range(HD):
                s += Q[t, i] * Kc[j, i]
            s *= scale
            sc[j] = s
            if s > mx:
                mx = s
        tot = np.float32(0.0)
        for j in range(n):
            e = np.float32(math.exp(sc[j] - mx))
            sc[j] = e
            tot += e
        acc = np.zeros(HD, np.float32)
        for j in range(n):
            p = sc[j] / tot
            for i in range(HD):
                acc[i] += p * Vc[j, i]
        out[t] = acc


@nb.njit(parallel=True, fastmath=True, cache=True)
def attention_all(QKV, Kc, Vc, pos0, out):
    """Внимание всех 16 голов для T слов одним запуском: слово t видит позиции 0..pos0+t."""
    T = QKV.shape[0]
    scale = np.float32(1.0 / math.sqrt(HD))
    for job in nb.prange(T * H):
        t = job // H
        h = job % H
        g = h // (H // KV)
        n = pos0 + t + 1
        q = QKV[t, h * HD:(h + 1) * HD].copy()
        sc = np.empty(n, np.float32)
        mx = np.float32(-1e30)
        for j in range(n):
            kj = Kc[g, j]
            s = np.float32(0.0)
            for i in range(HD):
                s += q[i] * kj[i]
            s *= scale
            sc[j] = s
            if s > mx:
                mx = s
        tot = np.float32(0.0)
        for j in range(n):
            e = np.float32(math.exp(sc[j] - mx))
            sc[j] = e
            tot += e
        acc = np.zeros(HD, np.float32)
        for j in range(n):
            p = sc[j] / tot
            vj = Vc[g, j]
            for i in range(HD):
                acc[i] += p * vj[i]
        out[t, h * HD:(h + 1) * HD] = acc


Engine.verify = verify
