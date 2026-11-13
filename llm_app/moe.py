"""Qwen3-MoE/GGUF: первые слои целиком CPU (C++/Numba), последние целиком GPU.
Один переход CPU -> GPU на проход. Общая генерация и JSON — engine/gengine.
"""
import ctypes
import time
import types

import numpy as np
import torch
import triton

from .gcpu import _dll
from .gengine import GGUFEngine
from .gguf import GGUF, find_ollama, validate_qwen, eos_ids
from .gkernels import ggemv, moe_gemv, nbytes, repack, silu_mul
from .plan import GiB
from .tkernels import attn_k, qk_rope_cache_k, rmsg_k, rope_attn_k

TYP = {"Q4_K": 0, "Q6_K": 1}


class CpuMoELayer:
    """Слой MoE целиком на процессоре: внимание (матрицы — C++, нормировки/поворот/внимание — Numba),
    маршрутизатор (numpy), эксперты (moe_ffn, C++). Кэш k, v — в RAM (fp32)."""

    def __init__(self, g, i, eng):
        from .gcpu import CpuMat
        n = lambda s: f"blk.{i}.{s}.weight"
        self.m = {k: CpuMat(g, n(s)) for k, s in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"))}
        self.s = {k: g.full(n(s)).astype(np.float32) for k, s in
                  (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))}
        self.R = np.ascontiguousarray(g.full(n("ffn_gate_inp")).astype(np.float32))  # [E, D]
        self.exps = []
        for s_ in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            b, _, _, tt = g.raw(n(s_))
            self.exps.append((np.asarray(b), TYP[tt]))
        self.e = eng
        self.Kc = np.zeros((eng.KV, eng.maxt, eng.HD), np.float32)
        self.Vc = np.zeros_like(self.Kc)

    def __call__(self, X, pos0, inv):
        from .cpu_layer import qk_rope_attn, rms_rows
        e = self.e
        T, D = X.shape
        H, KV, HD, Kx, F = e.H, e.KV, e.HD, e.K, e.F
        h = np.empty_like(X)
        rms_rows(X, self.s["ln1"], h, e.eps)
        qkv = np.empty((T, (H + 2 * KV) * HD), np.float32)
        self.m["q"].mm(h, qkv[:, :H * HD], None)
        self.m["k"].mm(h, qkv[:, H * HD:(H + KV) * HD], None)
        self.m["v"].mm(h, qkv[:, (H + KV) * HD:], None)
        att = np.empty((T, H * HD), np.float32)
        qk_rope_attn(qkv, self.s["qn"], self.s["kn"], inv, pos0, self.Kc, self.Vc, att, H, KV, HD, e.eps, True)
        self.m["o"].mm(att, X, None, True)
        rms_rows(X, self.s["ln2"], h, e.eps)
        lg = h @ self.R.T
        lg -= lg.max(-1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(-1, keepdims=True)
        ids = np.argsort(-p, axis=-1, kind="stable")[:, :Kx].astype(np.int32)
        w = np.take_along_axis(p, ids, -1)
        w = (w / w.sum(-1, keepdims=True)).astype(np.float32)
        ids, w = np.ascontiguousarray(ids), np.ascontiguousarray(w)
        Hs, Zs = e.Hs[:T * Kx], e.Zs[:T * Kx]
        (G, tg), (U, tu), (Dn, td) = self.exps
        t0 = time.perf_counter()
        rc = e.dll.moe_ffn(G.ctypes.data, U.ctypes.data, Dn.ctypes.data, tg, tu, td, D, F, T, Kx,
                           ids.ctypes.data, w.ctypes.data, h.ctypes.data, Hs.ctypes.data, Zs.ctypes.data)
        if rc != 0 or e.dll.take_failed():
            raise RuntimeError(f"moe_ffn: ошибка (код {rc})")
        e.moe_s += time.perf_counter() - t0
        X += Zs.reshape(T, Kx, D).sum(1)


class MoEEngine(GGUFEngine):
    supports_mask = True

    def __init__(self, name, maxt=2048, log=print, reserve=None, vram_reserve=None, extra_vram=0, resident=None, cpu=True):
        if not isinstance(maxt, int) or isinstance(maxt, bool) or maxt < 1:
            raise ValueError("maxt должно быть положительным целым")
        t0 = time.time()
        g = self.g = GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)
        a = g.meta["general.architecture"]
        if a != "qwen3moe":
            raise NotImplementedError(f"MoE: архитектура {a} не поддержана")
        validate_qwen(g)
        m = lambda k: g.meta[f"{a}.{k}"]
        self.L, self.D, self.H, self.KV = m("block_count"), m("embedding_length"), m("attention.head_count"), m("attention.head_count_kv")
        self.HD, self.eps = m("attention.key_length"), m("attention.layer_norm_rms_epsilon")
        self.E, self.K, self.F = m("expert_count"), m("expert_used_count"), m("expert_feed_forward_length")
        self.maxt, self.log, dev = maxt, log, "cuda"
        self.ck = types.SimpleNamespace(eos=eos_ids(g))
        self.inv = (1.0 / (m("rope.freq_base") ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))).to(dev)
        self.dll = _dll()
        if self.dll is None:
            raise RuntimeError("для MoE нужны ядра C++ (cpp/qmm.dll)")
        buf, N, K, t = g.raw("output.weight" if "output.weight" in g.tensors else "token_embd.weight")
        self.out, self.V = repack(buf, N, K, t), N
        self._norm = torch.from_numpy(g.full("output_norm.weight")).to(dev)
        n = lambda i, s: f"blk.{i}.{s}.weight"
        # деление по слоям: первые слои целиком на процессоре, последние целиком в видеокарте (один переход)
        def gpu_bytes(i):
            return sum(g.raw(n(i, s))[0].size for s in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_exps",
                                                         "ffn_up_exps", "ffn_down_exps")) * 1.03 + 2 * self.KV * maxt * self.HD * 2
        import os
        reserve = vram_reserve if vram_reserve is not None else reserve
        if reserve is None:
            reserve = float(os.environ.get("LLM_VRAM_RESERVE_GIB", "0.9")) * GiB
        free = torch.cuda.mem_get_info()[0] - reserve - extra_vram
        R = 0
        while R < self.L and sum(gpu_bytes(self.L - 1 - j) for j in range(R + 1)) <= free:
            R += 1
        if resident is not None:
            if not isinstance(resident, int) or not 0 <= resident <= self.L:
                raise ValueError("resident вне диапазона слоёв")
            R = resident
        if cpu is False and R < self.L:
            raise NotImplementedError("MoE без CPU требует размещения всех слоёв в GPU; подкачка MoE пока не поддерживается")
        self.Rg = R
        self.small, self.attn, self.router, self.gst, self.cpul, self.Kc, self.Vc = {}, {}, {}, {}, {}, {}, {}
        for i in range(self.L):
            if i < self.L - R:
                self.cpul[i] = CpuMoELayer(g, i, self)
                continue
            self.small[i] = {k: torch.from_numpy(g.full(n(i, s))).to(dev) for k, s in
                             (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))}
            self.attn[i] = {k: repack(*g.raw(n(i, s))) for k, s in
                            (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"))}
            self.router[i] = torch.from_numpy(g.full(n(i, "ffn_gate_inp"))).to(dev)  # [E, D]
            st = {}
            for k, s_ in (("g", "ffn_gate_exps"), ("u", "ffn_up_exps"), ("d", "ffn_down_exps")):
                b, N, K, tt = g.raw(n(i, s_))
                w = repack(b, N, K, tt)
                w["N"] = N // self.E  # строк одного эксперта (слот = номер эксперта)
                st[k] = w
            self.gst[i] = st
            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev, dtype=torch.float16)
            self.Vc[i] = torch.zeros_like(self.Kc[i])
            torch.cuda.empty_cache()
        self.place = ["cpu"] * (self.L - R) + ["gpu"] * R
        self._ids()
        f = lambda c: torch.zeros(16, c, device=dev)
        H, KV, HD = self.H, self.KV, self.HD
        self.QKV = (H + 2 * KV) * HD
        self.xb, self.hb, self.qkvb, self.qb, self.attb, self.logits = f(self.D), f(self.D), f(self.QKV), f(H * HD), f(H * HD), f(self.V)
        self.hX = torch.empty(16, self.D).pin_memory()
        self.hY = torch.empty(16, self.D).pin_memory()
        self.Hs = np.empty((16 * self.K, self.F), np.float32)
        self.Zs = np.empty((16 * self.K, self.D), np.float32)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.graphs, self.ctx = {}, []
        self.moe_s = 0.0
        self.pg = torch.zeros(16 * self.K, self.F, device=dev)
        self.pu = torch.zeros(16 * self.K, self.F, device=dev)
        self.pz = torch.zeros(16 * self.K, self.D, device=dev)
        self.inv_np = self.inv.cpu().numpy()
        self.load_s = time.time() - t0
        log(f"MoE {name}: {self.L} слоёв, экспертов {self.E} (на слово {self.K}); слоёв целиком в видеокарте {self.Rg}, "
            f"на процессоре {self.L - self.Rg}; видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ; загружено за {self.load_s:.0f} с")

    def process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None, mask=None, last_only=False):
        """MoE не поддерживает вытеснение через плотный GGUFEngine._demote."""
        try:
            return self._process(tokens, pos0, temperature, top_p, rng, mask, last_only)
        except torch.OutOfMemoryError:
            self.ctx = []  # ни один частично записанный префикс не переиспользуем
            self.graphs.clear()
            raise

    def autofit(self, *a, **k):
        pass  # размещение решается при загрузке (слои целиком)

    # ---------------- слой ----------------
    def _attn(self, l, x, T):
        H, KV, HD, D, s, W = self.H, self.KV, self.HD, self.D, self.small[l], self.attn[l]
        h, qkv, q, att = self.hb[:T], self.qkvb[:T], self.qb[:T], self.attb[:T]
        K, V = self.Kc[l], self.Vc[l]
        rmsg_k[(T,)](x, s["ln1"], h, self.eps, D, DP=triton.next_power_of_2(D))
        a, b = H * HD, (H + KV) * HD
        ggemv(h, W["q"], qkv, 0, T, ys=self.QKV)
        ggemv(h, W["k"], qkv[:, a:], 0, T, ys=self.QKV)
        ggemv(h, W["v"], qkv[:, b:], 0, T, ys=self.QKV)
        if T == 1:
            rope_attn_k[(H,)](qkv, s["qn"], s["kn"], self.inv, self.pos, K, V, att, self.maxt, self.eps, HD ** -0.5,
                              NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)
        else:
            qk_rope_cache_k[(H + 2 * KV, T)](qkv, s["qn"], s["kn"], self.inv, self.pos, q, K, V, self.maxt, self.eps,
                                              NH=H, NKV=KV, HDIM=HD, half=HD // 2)
            attn_k[(H, T)](q, K, V, self.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)
        ggemv(att, W["o"], x, 1, T)

    def _moe(self, l, x, T):
        """Эксперты слоя в видеокарте: выбор, все пары «слово-эксперт» групповыми ядрами — без синхронизации."""
        D, Kx = self.D, self.K
        h = self.hb[:T]
        rmsg_k[(T,)](x, self.small[l]["ln2"], h, self.eps, D, DP=triton.next_power_of_2(D))
        p = torch.softmax(h @ self.router[l].T, -1)
        ids = torch.argsort(p, dim=-1, descending=True, stable=True)[:, :Kx]
        w = p.gather(-1, ids)
        w = (w / w.sum(-1, keepdim=True)).reshape(-1)
        P = T * Kx
        sl = ids.reshape(-1).to(torch.int32)
        tk = torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(Kx)
        st = self.gst[l]
        pg, pu, pz = self.pg[:P], self.pu[:P], self.pz[:P]
        moe_gemv(h, st["g"], pg, sl, tk, P)
        moe_gemv(h, st["u"], pu, sl, tk, P)
        mm = (torch.nn.functional.silu(pg) * pu).contiguous()
        moe_gemv(mm, st["d"], pz, sl, torch.arange(P, device="cuda", dtype=torch.int32), P)
        x.add_((pz * w[:, None]).reshape(T, Kx, D).sum(dim=1))

    def _run(self, tokens, pos0):
        T = len(tokens)
        Lc = self.L - self.Rg
        xn = np.ascontiguousarray(self._emb(tokens).numpy())
        for l in range(Lc):  # слои процессора
            self.cpul[l](xn, int(pos0), self.inv_np)
        x = self.xb[:T]
        x.copy_(torch.from_numpy(xn))  # один переход процессор -> видеокарта
        self.pos.fill_(pos0)
        for l in range(Lc, self.L):
            self._attn(l, x, T)
            self._moe(l, x, T)
        return x

    def _decode(self, tokens, pos0):
        T = len(tokens)
        x = self._run(tokens, pos0)
        rmsg_k[(T,)](x, self._norm, self.hb[:T], self.eps, self.D, DP=triton.next_power_of_2(self.D))
        self._head(T)
        return T

    def _prefill(self, tokens, start):
        n = len(tokens)
        for c in range(0, n, 16):
            x = self._run(tokens[c:c + 16], start + c)
        T = min(16, n - (n - 1) // 16 * 16)
        rmsg_k[(1,)](x[T - 1:T], self._norm, self.hb[:1], self.eps, self.D, DP=triton.next_power_of_2(self.D))
        self._head(1)
