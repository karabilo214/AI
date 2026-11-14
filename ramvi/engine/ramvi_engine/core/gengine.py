"""Движок для моделей GGUF (те же файлы, что у Ollama/llama.cpp): python -m llm_app serve --model qwen3:14b
Веса перекладываются без потерь (gkernels.repack) и считаются своими ядрами (Q4_K, Q6_K, F16), счёт fp32.
План: сколько влезет слоёв в видеокарте, остальные подкачиваются по шине (поток-помощник, как в engine.py).
Кэш k, v — fp16 (для квантованной модели точность fp32 не нужна, память дороже). Выходной словарь (обычно
Q6_K) — в видеокарте, выбор слова там же. Чтение запроса — послойно, кусками по 16; угадывание — как в engine."""
from ramvi_engine.config import env
import time
import types

import numpy as np
import torch
import triton

from ramvi_engine.core.engine import BUCKETS, Arena, Engine, Loader, _pack, _unpack
from ramvi_engine.formats.gguf import GGUF, find_ollama, validate_qwen, eos_ids
from ramvi_engine.kernels.gkernels import ggemv, nbytes, repack, silu_mul
from ramvi_engine.planning.plan import GiB, ram_info
from ramvi_engine.kernels.tkernels import attn_k, qk_rope_cache_k, rmsg_k, rope_attn_k

GM = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}


class GGUFEngine(Engine):
    supports_mask = True
    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True):
        """cpu: True — слои вне видеокарты через один считает процессор (из файла модели), пока следующий едет по
        шине; "hybrid" — такие слои («hyb») при одиночном шаге считает процессор, а при проходе пачкой они
        подкачиваются по шине и считаются в видеокарте (кэш k, v — в обоих местах, новые строки копируются)."""
        if not isinstance(maxt, int) or isinstance(maxt, bool) or maxt < 1:
            raise ValueError("maxt должно быть положительным целым")
        t0 = time.time()
        import os
        if vram_reserve is None:
            vram_reserve = float(env("RAMVI_VRAM_RESERVE_GIB", "0.9")) * GiB
        g = self.g = GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)
        a = g.meta["general.architecture"]
        if a != "qwen3":
            raise NotImplementedError(f"архитектура {a} пока не поддержана")
        validate_qwen(g)
        m = lambda k: g.meta[f"{a}.{k}"]
        self.L, self.D, self.H, self.KV = m("block_count"), m("embedding_length"), m("attention.head_count"), m("attention.head_count_kv")
        self.HD, self.FF, self.eps = m("attention.key_length"), m("feed_forward_length"), m("attention.layer_norm_rms_epsilon")
        if resident is not None and (not isinstance(resident, int) or isinstance(resident, bool) or not 0 <= resident <= self.L):
            raise ValueError("resident вне диапазона слоёв")
        self.maxt, self.log, dev = maxt, log, "cuda"
        self.ck = types.SimpleNamespace(eos=eos_ids(g))
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
        alt = "hyb" if cpu == "hybrid" else "cpu"
        self.cpu_frac = float(env("RAMVI_CPU_FRAC", "0.58"))  # доля слоёв вне видеокарты на процессоре
        self.place = ["gpu"] * R + self._spread(self.L - R, alt if cpu else None)
        tot, avail = ram_info()
        log(f"GGUF {name}: {self.L} слоёв, {sum(lb) / GiB:.2f} ГиБ весов слоёв; видеокарта свободно {free / GiB:.1f} ГиБ -> "
            f"в видеокарте {R}, по шине {self.place.count('stream')}, на процессоре {self.place.count('cpu')}, "
            f"гибридных {self.place.count('hyb')}; "
            f"RAM доступно {avail / GiB:.1f} ГиБ")
        arena = Arena(int(0.4 * tot))
        self.W, self.HW, self.Kc, self.Vc = [], {}, {}, {}
        for i in range(self.L):
            if self.place[i] in ("cpu", "hyb"):
                from ramvi_engine.cpu.gcpu import GCpuLayer
                self.W.append(GCpuLayer(g, i, self.small[i], self.H, self.KV, self.HD, self.FF, self.eps, maxt))
                if self.place[i] == "cpu":
                    continue
            ws = {n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight")) for n in GM}
            if self.place[i] == "gpu":
                self.W.append(ws)
            else:
                pk = {}
                for n, w in ws.items():
                    b, lay = _pack(w)
                    pk[n] = (arena.put(b.cpu()), lay)
                if self.place[i] == "hyb":
                    self.HW[i] = pk
                else:
                    self.W.append(pk)
                del ws
            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev, dtype=torch.float16)
            self.Vc[i] = torch.zeros_like(self.Kc[i])
            torch.cuda.empty_cache()
        self.arena = arena
        self._ids()
        self.inv_np = self.inv.cpu().numpy()
        if self.cpu_ids or self.hyb_ids:
            self.hx = torch.empty(16, self.D).pin_memory()
            self.xs = torch.cuda.Stream()
        if not self.all_gpu:
            self.loader = Loader()
            import collections
            self.pending, self.last_rel, self.cslot, self.order, self.phase = collections.deque(), {}, 0, None, None
            if self.stream_ids or self.hyb_ids:
                mx = {n: max(self._pk(l)[n][0].numel() for l in self.stream_ids + self.hyb_ids) for n in GM}
                self.cbuf = [{n: torch.empty(mx[n], dtype=torch.uint8, device=dev) for n in GM} for _ in range(2)]
        f = lambda n: torch.zeros(16, n, device=dev)
        H, KV, HD = self.H, self.KV, self.HD
        self.QKV = (H + 2 * KV) * HD
        self.xb, self.hb, self.qkvb, self.qb, self.attb = f(self.D), f(self.D), f(self.QKV), f(H * HD), f(H * HD)
        self.gub, self.mb, self.logits = f(2 * self.FF), f(self.FF), f(self.V)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.graphs, self.ctx = {}, []
        profile_path = env("RAMVI_LAYER_PROFILE")
        if profile_path:
            if resident is None:
                raise ValueError("RAMVI_LAYER_PROFILE требует явного resident; профиль применяют после autofit через configure_placement")
            import json
            with open(profile_path, encoding="utf-8") as handle:
                self.configure_placement(json.load(handle))
        self.load_s = time.time() - t0
        log(f"загружено за {self.load_s:.0f} с: видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ, "
            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")

    # ---- подкачка: только сжатые слои ----
    def _spread(self, n, alt):
        """n слоёв вне видеокарты: доля cpu_frac — процессору (равномерно вперемешку), остальные — по шине."""
        out, acc = [], 0.0
        for _ in range(n):
            acc += self.cpu_frac if alt else 0.0
            if acc >= 1.0 - 1e-9:
                out.append(alt)
                acc -= 1.0
            else:
                out.append("stream")
        return out

    def _pk(self, l):
        return self.HW[l] if self.place[l] == "hyb" else self.W[l]

    def _request(self, l):
        if l is None:
            return
        s = self.cslot
        self.cslot ^= 1
        pk = self._pk(l)
        pairs = [(pk[n][0], self.cbuf[s][n][:pk[n][0].numel()]) for n in GM]
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
        self.loader.wait(job)
        torch.cuda.current_stream().wait_event(job["ev"])
        self.cur = job
        pk = self._pk(l)
        return {n: _unpack(self.cbuf[job["s"]][n], pk[n][1]) for n in GM}

    def _set_phase(self, phase):
        """Фазы: "d1" — одиночный шаг (подкачиваются только stream), "dN" — пачка и чтение запроса
        (подкачиваются stream и hyb). Порядок подкачки — по кругу; при смене фазы начатая подкачка отпускается."""
        if self.all_gpu:
            return
        routes = getattr(self, "_active_routes", None)
        if routes is not None:
            ids = [l for l, mode in enumerate(routes) if mode == "stream"]
            phase = ("measured", tuple(ids))
        else:
            if phase != "d1":
                phase = "dN"
            if not self.hyb_ids:
                phase = "dN"
            ids = self.stream_ids if phase == "d1" else sorted(self.stream_ids + self.hyb_ids)
        if self.phase == phase:
            return
        while self.pending:
            job = self.pending.popleft()
            self.loader.wait(job)
            ev = torch.cuda.Event(); ev.record()
            job["rel"]["ev"] = ev
            job["rel"]["done"].set()
        import itertools
        self.order = itertools.cycle(ids) if ids else iter(())
        self.phase = phase
        self._request(next(self.order, None))
        self._request(next(self.order, None))

    def _layers(self, x, T, pos0):
        for l in range(self.L):
            pl = self.place[l]
            routes = getattr(self, "_active_routes", None)
            mode = routes[l] if routes is not None else ("stream" if pl == "hyb" and T > 1 else "cpu" if pl == "hyb" else pl)
            if mode == "gpu":
                self._gpu_layer(l, self.W[l], x, T)
            elif mode == "stream":
                self._gpu_layer(l, self._acquire(l), x, T)
                self._release()
            else:
                self._cpu_layer(l, x, pos0)
                if pl == "hyb":  # строка кэша — в видеокарту (в fp16, как там хранится)
                    cl = self.W[l]
                    for C, G in ((cl.K, self.Kc[l]), (cl.V, self.Vc[l])):
                        r = C[:, pos0:pos0 + T].astype(np.float16)
                        C[:, pos0:pos0 + T] = r
                        G[:, pos0:pos0 + T].copy_(torch.from_numpy(r))
        rmsg_k[(T,)](x, self._norm, self.hb[:T], self.eps, self.D, DP=triton.next_power_of_2(self.D))

    def _hyb_to_cpu(self, p0, n):
        """После прохода пачкой: новые строки кэша гибридных слоёв — из видеокарты в RAM."""
        for l in self.hyb_ids:
            cl = self.W[l]
            cl.K[:, p0:p0 + n] = self.Kc[l][:, p0:p0 + n].float().cpu().numpy()
            cl.V[:, p0:p0 + n] = self.Vc[l][:, p0:p0 + n].float().cpu().numpy()

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
        self._active_routes = (getattr(self, "_measured_routes", {}).get(T)
                               if pos0 // 256 == getattr(self, "_measured_context_bucket", 0) else None)
        self._set_phase("d1" if T == 1 else "dN")
        if self.all_gpu:
            B = T  # граф обрабатывает только реальные строки, без записи KV за конец контекста
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
        if self.hyb_ids and (T > 1 or self._active_routes is not None):
            self._hyb_to_cpu(pos0, T)
        return T

    def _capture(self, B):
        """Capture использует рабочие адреса KV, но восстанавливает затронутый хвост."""
        if not 1 <= B <= min(16, self.maxt):
            raise ValueError("размер графа не помещается в контекст")
        saved = self.pos.clone()
        start = self.maxt - B
        # Граф должен ссылаться на настоящие KV: отдельные временные KV здесь непригодны.
        backups = [(cache, cache[:, start:].clone())
                   for caches in (self.Kc, self.Vc) for cache in caches.values()]
        self.pos.fill_(start)
        try:
            self._capture_at(B)
        finally:
            for cache, tail in backups:
                cache[:, start:].copy_(tail)
            self.pos.copy_(saved)

    def _capture_at(self, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._layers(self.xb[:B], B, 0)
                self._head(B)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self._layers(self.xb[:B], B, 0)
            self._head(B)
        self.graphs[B] = g

    def _prefill(self, tokens, start):
        """Послойно: каждый слой (в т.ч. подкачанный) сразу на весь запрос кусками по 16."""
        from ramvi_engine.kernels.prefill import enabled, layer
        blocked = enabled(len(tokens))
        self._active_routes = None
        n = len(tokens)
        self._set_phase("dN")
        X = self._emb(tokens).cuda()
        for l in range(self.L):
            if self.place[l] == "cpu":
                for c in range(0, n, 16):
                    self._cpu_layer(l, X[c:c + 16], start + c)
                continue
            W = self.W[l] if self.place[l] == "gpu" else self._acquire(l)
            if blocked:
                layer(self, l, W, X, start, self.Kc[l], self.Vc[l], gguf=True)
            else:
                for c in range(0, n, 16):
                    T = min(16, n - c)
                    self.pos.fill_(start + c)
                    self._batch_layer(l, X[c:c + T], T, W, self.Kc[l], self.Vc[l], None)
            if self.place[l] != "gpu":
                self._release()
        rmsg_k[(1,)](X[n - 1:n], self._norm, self.hb[:1], self.eps, self.D, DP=triton.next_power_of_2(self.D))
        self._head(1)
        if self.hyb_ids:
            self._hyb_to_cpu(start, n)

    def _pick(self, T, temperature, top_p, rng, mask=None, row0=0):
        lg = self.logits[row0:row0 + T]
        if mask is not None:
            from ramvi_engine.generation.constrain import Constraint
            Constraint.apply(lg, mask)
        if temperature <= 0:
            mx, am = lg.max(-1)
            self.last_conf = (1.0 / (lg - mx[:, None]).exp().sum(-1)).tolist()  # вероятность выбранного слова
            return am.tolist()
        p = torch.softmax(lg.double() / temperature, -1)
        sp, si = p.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p
        sp = sp * keep
        out = []
        for t in range(T):
            pr = (sp[t] / sp[t].sum()).cpu().numpy()
            out.append(int(si[t, (rng or np.random.default_rng()).choice(len(pr), p=pr)]))
        return out

    # ---------------- автоматический контроль видеопамяти ----------------
    def _layer_gpu_bytes(self, l):
        from ramvi_engine.kernels.gkernels import nbytes as nb_
        if self.place[l] in ("stream", "hyb"):
            return sum(self._pk(l)[n][0].numel() for n in GM)
        return sum(self.g.raw(f"blk.{l}.{GM[n]}.weight")[0].size for n in GM) * 1.1 + 2 * self.KV * self.maxt * self.HD * 2

    def _reset_loader(self):
        self.phase = None
        if not hasattr(self, "pending"):
            return
        while self.pending:
            job = self.pending.popleft()
            self.loader.wait(job)
            ev = torch.cuda.Event(); ev.record()
            job["rel"]["ev"] = ev
            job["rel"]["done"].set()
        self.phase = None

    def _ids(self):
        self.stream_ids = [i for i in range(self.L) if self.place[i] == "stream"]
        self.cpu_ids = [i for i in range(self.L) if self.place[i] == "cpu"]
        self.hyb_ids = [i for i in range(self.L) if self.place[i] == "hyb"]
        self.all_gpu = not self.stream_ids and not self.cpu_ids and not self.hyb_ids

    def _promote(self, l):
        """Слой l (по шине или на процессоре) — постоянно в видеокарту."""
        self._reset_loader()
        torch.cuda.synchronize()
        if self.place[l] in ("stream", "hyb"):
            pk = self._pk(l)
            weights = {n: _unpack(pk[n][0].to("cuda"), pk[n][1]) for n in GM}
            kc, vc = self.Kc[l], self.Vc[l]
        else:
            cl = self.W[l]
            weights = {n: repack(*self.g.raw(f"blk.{l}.{GM[n]}.weight")) for n in GM}
            kc = torch.from_numpy(cl.K).to("cuda", torch.float16)
            vc = torch.from_numpy(cl.V).to("cuda", torch.float16)
        self.W[l] = weights
        self.Kc[l], self.Vc[l] = kc, vc
        self.HW.pop(l, None)
        self.place[l] = "gpu"
        self._ids()
        self._measured_routes = {}
        self._active_routes = None
        self.graphs = {}

    def _demote(self, l):
        """Слой l из видеокарты — на подкачку по шине (при нехватке памяти)."""
        self._reset_loader()
        torch.cuda.synchronize()
        pk = {}
        for n in GM:
            host = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.W[l][n].items()}
            b, lay = _pack(host)  # упаковка на CPU: аварийный путь не выделяет копию весов в VRAM
            pk[n] = (self.arena.put(b), lay)
        self.W[l] = pk
        self.place[l] = "stream"
        self._ids()
        self._measured_routes = {}
        self._active_routes = None
        self.graphs = {}
        self.phase = None
        torch.cuda.empty_cache()
        if not hasattr(self, "loader"):
            import collections
            self.loader = Loader()
            self.pending, self.last_rel, self.cslot, self.order, self.phase = collections.deque(), {}, 0, None, None
        mx = {n: max(self.W[i][n][0].numel() for i in self.stream_ids) for n in GM}
        if not hasattr(self, "cbuf") or any(self.cbuf[0][n].numel() < mx[n] for n in GM):
            self.cbuf = None
            torch.cuda.empty_cache()
            self.cbuf = [{n: torch.empty(mx[n], dtype=torch.uint8, device="cuda") for n in GM} for _ in range(2)]
        torch.cuda.empty_cache()

    def calibrate_placement(self, batches=(1, 4, 16), repeats=3):
        """Явный прогрев альтернатив CPU/GPU/DMA, без работы параллельно с запросами."""
        from ramvi_engine.planning.measured_plan import calibrate
        return calibrate(self, batches, repeats)

    def configure_placement(self, profile):
        """Включить маршруты по измерениям; GPU-резидентность задаёт план памяти."""
        from ramvi_engine.planning.measured_plan import configure
        configure(self, profile)

    def autofit(self, margin=0.35 * GiB, warm=256):
        """Прогрев (чтение запроса, пачка, шаг), замер настоящего запаса видеопамяти и перенос в видеокарту
        стольких слоёв, сколько влезает с запасом margin (сначала подкачиваемые, потом процессорные)."""
        warm = min(warm, max(1, self.maxt - 17))
        toks = [min(1000, self.V - 1) + (i * 7919) % max(1, self.V - min(1000, self.V - 1))
                for i in range(warm + 17)]
        self.process(toks[:min(warm, self.maxt)], 0)
        if warm + 16 <= self.maxt:
            self.process(toks[warm:warm + 16], warm)
        if warm + 17 <= self.maxt:
            self.process(toks[warm + 16:warm + 17], warm + 16)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()  # кратковременный пик сверх постоянного
        moved = 0
        while True:
            # переносим из той группы вне видеокарты, которой больше: процессор и шина остаются загружены поровну
            side = self.cpu_ids or self.hyb_ids
            nout = len(self.stream_ids) + len(side)
            if self.stream_ids and (not side or len(side) < self.cpu_frac * nout - 0.5 or len(self.stream_ids) > (1 - self.cpu_frac) * nout):
                cand = self.stream_ids[0]
            else:
                cand = (side or [None])[0]
            if cand is None:
                break
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0]
            need = self._layer_gpu_bytes(cand)
            if free - peak - margin < need:
                break
            self._promote(cand)
            moved += 1
        self.ctx = []
        free = torch.cuda.mem_get_info()[0]
        self.log(f"контроль видеопамяти: пик сверх постоянного {peak / GiB:.2f} ГиБ, перенесено в видеокарту слоёв {moved}; "
                 f"теперь в видеокарте {self.place.count('gpu')}, по шине {len(self.stream_ids)}, на процессоре "
                 f"{len(self.cpu_ids)}, гибридных {len(self.hyb_ids)}; свободно {free / GiB:.2f} ГиБ")

    def process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None, mask=None, last_only=False):
        prefix = list(self.ctx[:pos0])
        import copy
        random_state = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
        original = None
        for attempt in range(self.L + 1):
            try:
                if attempt:
                    if rng is not None:
                        rng.bit_generator.state = copy.deepcopy(random_state)
                    if prefix:
                        self._process(prefix, 0)  # восстановить согласованный KV всех слоёв
                return self._process(tokens, pos0, temperature, top_p, rng, mask, last_only)
            except torch.OutOfMemoryError as exc:
                original = original or exc
                self.ctx = []
                gpu = [i for i in range(self.L) if self.place[i] == "gpu"]
                if not gpu or attempt == self.L:
                    raise original
                try:
                    for l in gpu[-(3 if self.all_gpu else 1):]:
                        self._demote(l)
                except Exception as recovery:
                    self.ctx = []
                    raise original from recovery
                self.log(f"нехватка видеопамяти: слоёв в видеокарте теперь {self.place.count('gpu')}")
                torch.cuda.empty_cache()

    def _process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None, mask=None, last_only=False):
        self._validate_process(tokens, pos0)
        if pos0 + len(tokens) > self.maxt:
            raise ValueError(f"текст длиннее {self.maxt} слов")
        previous = self.ctx[:pos0]
        try:
            if len(tokens) > 16:
                self._prefill(tokens, pos0)
                result = self._pick(1, temperature, top_p, rng, mask)
            else:
                T = self._decode(tokens, pos0)
                result = (self._pick(1, temperature, top_p, rng, mask, row0=T - 1) if last_only
                          else self._pick(T, temperature, top_p, rng, mask))
        except Exception:
            self.ctx = []
            raise
        self.ctx = previous + list(tokens)
        return result


def open_gguf(name, **kw):
    """Модель GGUF по имени Ollama или пути: MoE (qwen3moe) -> MoEEngine, иначе GGUFEngine."""
    g = GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)
    if g.meta.get("general.architecture") == "qwen3moe":
        from ramvi_engine.core.moe import MoEEngine
        return MoEEngine(name, **kw)
    return GGUFEngine(name, **kw)
