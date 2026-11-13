"""Модели MoE из GGUF (архитектура qwen3moe, например qwen3:30b = Qwen3-30B-A3B).
Видеокарта: внимание (наши ядра Q4_K/Q6_K), маршрутизатор (F32), словарь, кэш k, v (fp16).
Процессор: эксперты (cpp/qmm.dll, moe_ffn) прямо из файла модели — слова группируются по эксперту, строка
эксперта распаковывается один раз на все выбравшие его слова. Выбор экспертов как в Qwen3-MoE: softmax по
всем, лучшие k, веса перенормируются. Генерация, угадывание, JSON-ограничение — общие (engine/gengine)."""
import ctypes
import time
import types

import numpy as np
import torch
import triton

from .gcpu import _dll
from .gengine import GGUFEngine
from .gguf import GGUF, find_ollama
from .gkernels import ggemv, repack
from .plan import GiB
from .tkernels import attn_k, qk_rope_cache_k, rmsg_k, rope_attn_k

TYP = {"Q4_K": 0, "Q6_K": 1}


class MoEEngine(GGUFEngine):
    supports_mask = True

    def __init__(self, name, maxt=2048, log=print, **_):
        t0 = time.time()
        g = self.g = GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)
        a = g.meta["general.architecture"]
        if a != "qwen3moe":
            raise NotImplementedError(f"MoE: архитектура {a} не поддержана")
        m = lambda k: g.meta[f"{a}.{k}"]
        self.L, self.D, self.H, self.KV = m("block_count"), m("embedding_length"), m("attention.head_count"), m("attention.head_count_kv")
        self.HD, self.eps = m("attention.key_length"), m("attention.layer_norm_rms_epsilon")
        self.E, self.K, self.F = m("expert_count"), m("expert_used_count"), m("expert_feed_forward_length")
        self.maxt, self.log, dev = maxt, log, "cuda"
        self.ck = types.SimpleNamespace(eos={151645, 151643})
        self.inv = (1.0 / (m("rope.freq_base") ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))).to(dev)
        self.dll = _dll()
        if self.dll is None:
            raise RuntimeError("для MoE нужны ядра C++ (cpp/qmm.dll)")
        self.dll.moe_ffn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 7 + [ctypes.c_void_p] * 5
        buf, N, K, t = g.raw("output.weight" if "output.weight" in g.tensors else "token_embd.weight")
        self.out, self.V = repack(buf, N, K, t), N
        self._norm = torch.from_numpy(g.full("output_norm.weight")).to(dev)
        n = lambda i, s: f"blk.{i}.{s}.weight"
        self.small, self.attn, self.router, self.exps, self.Kc, self.Vc = [], [], [], [], {}, {}
        for i in range(self.L):
            self.small.append({k: torch.from_numpy(g.full(n(i, s))).to(dev) for k, s in
                               (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))})
            self.attn.append({k: repack(*g.raw(n(i, s))) for k, s in
                              (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output"))})
            self.router.append(torch.from_numpy(g.full(n(i, "ffn_gate_inp"))).to(dev))  # [E, D]
            ex = []
            for s in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                b, _, _, tt = g.raw(n(i, s))
                if tt not in TYP:
                    raise NotImplementedError(f"эксперты в {tt} пока не поддержаны")
                ex.append((np.asarray(b), TYP[tt]))
            self.exps.append(ex)
            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev, dtype=torch.float16)
            self.Vc[i] = torch.zeros_like(self.Kc[i])
        self.place = ["gpu"] * self.L
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
        self.load_s = time.time() - t0
        exb = sum(e[0].size for ex in self.exps for e in ex)
        log(f"MoE {name}: {self.L} слоёв, экспертов {self.E} (на слово {self.K}), эксперты {exb / GiB:.1f} ГиБ — на процессоре "
            f"из файла; видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ; загружено за {self.load_s:.0f} с")

    def autofit(self, *a, **k):
        pass  # всё внимание и так в видеокарте; эксперты — на процессоре

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
        D, Kx = self.D, self.K
        h = self.hb[:T]
        rmsg_k[(T,)](x, self.small[l]["ln2"], h, self.eps, D, DP=triton.next_power_of_2(D))
        p = torch.softmax(h @ self.router[l].T, -1)  # [T, E]
        w, ids = p.topk(Kx, -1)
        w = w / w.sum(-1, keepdim=True)
        hx = self.hX[:T]
        hx.copy_(h)
        ids_c = ids.to(torch.int32).cpu().numpy().copy()
        w_c = w.float().cpu().numpy().copy()
        t0 = time.perf_counter()
        (G, tg), (U, tu), (Dn, td) = self.exps[l]
        Hs, Zs = self.Hs[:T * Kx], self.Zs[:T * Kx]
        self.dll.moe_ffn(G.ctypes.data, U.ctypes.data, Dn.ctypes.data, tg, tu, td, D, self.F, T, Kx,
                         ids_c.ctypes.data, w_c.ctypes.data, hx.data_ptr(), Hs.ctypes.data, Zs.ctypes.data)
        y = Zs.reshape(T, Kx, D).sum(1)
        self.moe_s += time.perf_counter() - t0
        self.hY[:T].copy_(torch.from_numpy(y))
        x += self.hY[:T].to("cuda", non_blocking=True)

    def _run(self, tokens, pos0):
        T = len(tokens)
        x = self.xb[:T]
        x.copy_(self._emb(tokens))
        self.pos.fill_(pos0)
        for l in range(self.L):
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
