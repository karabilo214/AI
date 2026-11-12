"""Движок для моделей GGUF (те же файлы, что у Ollama/llama.cpp): python -m llm_app serve --model qwen3:14b
Веса перекладываются без потерь (gkernels.repack) и считаются своими ядрами (Q4_K, Q6_K, F16), счёт fp32.
План: сколько влезет слоёв в видеокарте, остальные подкачиваются по шине (поток-помощник, как в engine.py).
Кэш k, v — fp16 (для квантованной модели точность fp32 не нужна, память дороже). Выходной словарь (обычно
Q6_K) — в видеокарте, выбор слова там же. Чтение запроса — послойно, кусками по 16; угадывание — как в engine."""
import time
import types

import numpy as np
import torch
import triton

from .engine import BUCKETS, Arena, Engine, Loader, _pack, _unpack
from .gguf import GGUF, find_ollama
from .gkernels import ggemv, nbytes, repack, silu_mul
from .plan import GiB, ram_info
from .tkernels import attn_k, qk_rope_cache_k, rmsg_k, rope_attn_k

GM = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}


class GGUFEngine(Engine):
    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True):
        """cpu: слои вне видеокарты через один считает процессор (из файла модели), пока следующий едет по шине."""
        t0 = time.time()
        import os
        if vram_reserve is None:
            vram_reserve = float(os.environ.get("LLM_VRAM_RESERVE_GIB", "0.9")) * GiB
        g = self.g = GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)
        a = g.meta["general.architecture"]
        if a != "qwen3":
            raise NotImplementedError(f"архитектура {a} пока не поддержана")
        m = lambda k: g.meta[f"{a}.{k}"]
        self.L, self.D, self.H, self.KV = m("block_count"), m("embedding_length"), m("attention.head_count"), m("attention.head_count_kv")
        self.HD, self.FF, self.eps = m("attention.key_length"), m("feed_forward_length"), m("attention.layer_norm_rms_epsilon")
        self.maxt, self.log, dev = maxt, log, "cuda"
        self.ck = types.SimpleNamespace(eos={151645, 151643})
        self.inv = (1.0 / (m("rope.freq_base") ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))).to(dev)
        # выход (словарь) и нормировки — всегда в видеокарте
        buf, N, K, t = g.raw("output.weight" if "output.weight" in g.tensors else "token_embd.weight")
        self.out = repack(buf, N, K, t)
        self.V = N
        self._norm = torch.from_numpy(g.full("output_norm.weight")).to(dev)
        self.small = [{k: torch.from_numpy(g.full(f"blk.{i}.{n}.weight")).to(dev)
                       for k, n in (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))}
                      for i in range(self.L)]
        # план памяти: байты слоёв
        lb = [sum(g.raw(f"blk.{i}.{GM[n]}.weight")[0].size for n in GM) for i in range(self.L)]
        kvb = 2 * self.KV * maxt * self.HD * 2
        free, _ = torch.cuda.mem_get_info()
        budget = free - vram_reserve - extra_vram - 16 * self.V * 4  # логиты
        if resident is None:
            R = 0
            while R < self.L and sum(lb[:R + 1]) + self.L * kvb + (2 * max(lb) if R + 1 < self.L else 0) <= budget:
                R += 1
        else:
            R = resident
        self.place = ["gpu"] * R + [("cpu" if cpu and (i - R) % 2 == 1 else "stream") for i in range(R, self.L)]
        tot, avail = ram_info()
        log(f"GGUF {name}: {self.L} слоёв, {sum(lb) / GiB:.2f} ГиБ весов слоёв; видеокарта свободно {free / GiB:.1f} ГиБ -> "
            f"в видеокарте {R}, по шине {self.place.count('stream')}, на процессоре {self.place.count('cpu')}; "
            f"RAM доступно {avail / GiB:.1f} ГиБ")
        arena = Arena(int(0.4 * tot))
        self.W, self.Kc, self.Vc = [], {}, {}
        for i in range(self.L):
            if self.place[i] == "cpu":
                from .gcpu import GCpuLayer
                self.W.append(GCpuLayer(g, i, self.small[i], self.H, self.KV, self.HD, self.FF, self.eps, maxt))
                continue
            ws = {n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight")) for n in GM}
            if self.place[i] == "gpu":
                self.W.append(ws)
            else:
                pk = {}
                for n, w in ws.items():
                    b, lay = _pack(w)
                    pk[n] = (arena.put(b.cpu()), lay)
                self.W.append(pk)
                del ws
            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev, dtype=torch.float16)
            self.Vc[i] = torch.zeros_like(self.Kc[i])
            torch.cuda.empty_cache()
        self.arena = arena
        self.stream_ids = [i for i in range(self.L) if self.place[i] == "stream"]
        self.cpu_ids = [i for i in range(self.L) if self.place[i] == "cpu"]
        self.all_gpu = not self.stream_ids and not self.cpu_ids
        self.inv_np = self.inv.cpu().numpy()
        if self.cpu_ids:
            self.hx = torch.empty(16, self.D).pin_memory()
            self.xs = torch.cuda.Stream()
        if not self.all_gpu:
            self.loader = Loader()
            import collections
            self.pending, self.last_rel, self.cslot, self.order, self.phase = collections.deque(), {}, 0, None, None
            if self.stream_ids:
                mx = {n: max(self.W[l][n][0].numel() for l in self.stream_ids) for n in GM}
                self.cbuf = [{n: torch.empty(mx[n], dtype=torch.uint8, device=dev) for n in GM} for _ in range(2)]
        f = lambda n: torch.zeros(16, n, device=dev)
        H, KV, HD = self.H, self.KV, self.HD
        self.QKV = (H + 2 * KV) * HD
        self.xb, self.hb, self.qkvb, self.qb, self.attb = f(self.D), f(self.D), f(self.QKV), f(H * HD), f(H * HD)
        self.gub, self.mb, self.logits = f(2 * self.FF), f(self.FF), f(self.V)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.graphs, self.ctx = {}, []
        self.load_s = time.time() - t0
        log(f"загружено за {self.load_s:.0f} с: видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ, "
            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")

    # ---- подкачка: только сжатые слои ----
    def _request(self, l):
        if l is None:
            return
        s = self.cslot
        self.cslot ^= 1
        pairs = [(self.W[l][n][0], self.cbuf[s][n][:self.W[l][n][0].numel()]) for n in GM]
        import threading
        rel = {"done": threading.Event(), "ev": None}
        job = {"l": l, "kind": "c", "s": s, "pairs": pairs, "wait": self.last_rel.get(("c", s)),
               "done": threading.Event(), "ev": None, "rel": rel}
        self.last_rel[("c", s)] = rel
        self.pending.append(job)
        self.loader.q.put(job)

    def _acquire(self, l):
        job = self.pending.popleft()
        assert job["l"] == l
        job["done"].wait()
        torch.cuda.current_stream().wait_event(job["ev"])
        self.cur = job
        return {n: _unpack(self.cbuf[job["s"]][n], self.W[l][n][1]) for n in GM}

    def _set_phase(self, phase):
        if not self.all_gpu and phase == "prefill":
            phase = "decode"  # здесь чтение запроса тоже идёт по кругу подкачиваемых слоёв
        super()._set_phase(phase)

    # ---- слой ----
    def _gpu_layer(self, l, W, x, T):
        self._batch_layer(l, x, T, W, self.Kc[l], self.Vc[l], None)

    def _batch_layer(self, l, x, T, W, K, V, mm):
        H, KV, HD, D, s = self.H, self.KV, self.HD, self.D, self.small[l]
        h, qkv, q, att, gu, mb = self.hb[:T], self.qkvb[:T], self.qb[:T], self.attb[:T], self.gub[:T], self.mb[:T]
        DP = triton.next_power_of_2(D)
        rmsg_k[(T,)](x, s["ln1"], h, self.eps, D, DP=DP)
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
        rmsg_k[(T,)](x, s["ln2"], h, self.eps, D, DP=DP)
        ggemv(h, W["gate"], gu, 0, T, ys=2 * self.FF)
        ggemv(h, W["up"], gu[:, self.FF:], 0, T, ys=2 * self.FF)
        silu_mul(gu, mb, T, self.FF)
        ggemv(mb, W["down"], x, 1, T)

    @property
    def ck_norm(self):
        return self._norm

    def _emb(self, tokens):
        return torch.from_numpy(self.g.rows("token_embd.weight", list(tokens)))

    def _head(self, T):
        ggemv(self.hb[:T], self.out, self.logits, 0, T)

    def _decode(self, tokens, pos0):
        T = len(tokens)
        self._set_phase("decode")
        if self.all_gpu:
            B = next(b for b in BUCKETS if b >= T)
            if B not in self.graphs:
                self._capture(B)
            self.xb[:B].zero_()
            self.xb[:T].copy_(self._emb(tokens))
            self.pos.fill_(pos0)
            self.graphs[B].replay()
            return T
        self.xb[:T].copy_(self._emb(tokens))
        self.pos.fill_(pos0)
        self._layers(self.xb[:T], T, pos0)
        self._head(T)
        return T

    def _capture(self, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._layers(self.xb[:B], B, 0)
                self._head(B)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._layers(self.xb[:B], B, 0)
            self._head(B)
        self.graphs[B] = g

    def _prefill(self, tokens, start):
        """Послойно: каждый слой (в т.ч. подкачанный) сразу на весь запрос кусками по 16."""
        n = len(tokens)
        self._set_phase("decode")
        X = self._emb(tokens).cuda()
        for l in range(self.L):
            if self.place[l] == "cpu":
                for c in range(0, n, 16):
                    self._cpu_layer(l, X[c:c + 16], start + c)
                continue
            W = self.W[l] if self.place[l] == "gpu" else self._acquire(l)
            for c in range(0, n, 16):
                T = min(16, n - c)
                self.pos.fill_(start + c)
                self._batch_layer(l, X[c:c + T], T, W, self.Kc[l], self.Vc[l], None)
            if self.place[l] != "gpu":
                self._release()
        rmsg_k[(1,)](X[n - 1:n], self._norm, self.hb[:1], self.eps, self.D, DP=triton.next_power_of_2(self.D))
        self._head(1)

    def _pick(self, T, temperature, top_p, rng):
        lg = self.logits[:T]
        if temperature <= 0:
            return lg.argmax(-1).tolist()
        p = torch.softmax(lg.double() / temperature, -1)
        sp, si = p.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p
        sp = sp * keep
        out = []
        for t in range(T):
            pr = (sp[t] / sp[t].sum()).cpu().numpy()
            out.append(int(si[t, (rng or np.random.default_rng()).choice(len(pr), p=pr)]))
        return out

    def process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None):
        if pos0 + len(tokens) > self.maxt:
            raise ValueError(f"текст длиннее {self.maxt} слов")
        self.ctx = self.ctx[:pos0] + list(tokens)
        if len(tokens) > 16:
            self._prefill(tokens, pos0)
            return self._pick(1, temperature, top_p, rng)
        T = self._decode(tokens, pos0)
        return self._pick(T, temperature, top_p, rng)
