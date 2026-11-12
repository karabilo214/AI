"""Движок llm_app: точная генерация (текст как у fp32) на видеокарте NVIDIA с любым объёмом памяти.

Слои по плану памяти (plan.py):
  gpu    — сжатые без потерь 12 бит, постоянно в видеокарте;
  stream — сжатые 12 бит в закреплённой RAM, подкачиваются по шине (поток-помощник, куски по 10 МБ);
  cpu    — считаются процессором прямо из файла модели (memmap, bf16, счёт fp32), пока следующий слой едет по шине.
Чтение запроса — послойно: каждый слой (в т.ч. подкачанный) сразу обрабатывает весь запрос кусками по 16 слов;
слои процессора на это время подкачиваются в видеокарту в bf16. Начало, совпавшее с прошлым запросом, не
пересчитывается. Выбор слова — head.py (int8-словарь в видеокарте + точные строки из файла).
"""
import collections
import itertools
import queue
import threading
import time

import numpy as np
import torch
import triton

from .cpu_layer import CpuLayer
from .head import Head
from .model import Ckpt
from .plan import GiB, make_plan
from .tkernels import attn_k, gemv, qk_rope_cache_k, rmsg_k, rope_attn_k
from .tune import tune
from .wcomp import compress, gemvc1, gemvc_tc

MATS = ("qkv", "o", "gu", "down")
BUCKETS = (1, 2, 4, 8, 16)


def _pack(c):
    parts, lay, off = [], [], 0
    for k, v in c.items():
        if torch.is_tensor(v):
            b = v.contiguous().view(torch.uint8).flatten()
            lay.append((k, off, v.dtype, tuple(v.shape)))
            pad = (-b.numel()) % 256
            parts += [b, torch.zeros(pad, dtype=torch.uint8, device=b.device)]
            off += b.numel() + pad
        else:
            lay.append((k, None, v, None))
    return torch.cat(parts), lay


def _unpack(buf, lay):
    c = {}
    for k, off, dt, sh in lay:
        if off is None:
            c[k] = dt
        else:
            n = torch.Size(sh).numel() * torch.empty(0, dtype=dt).element_size()
            c[k] = buf[off:off + n].view(dt).view(sh)
    return c


class Arena:
    """Закреплённая RAM кусками ровно по 256 МиБ (torch округляет закреплённые куски до степени двойки;
    Windows даёт закрепить ограниченный объём — сверх бюджета память остаётся обычной)."""
    CH = 256 * 2**20

    def __init__(self, budget):
        self.budget, self.chunks, self.used, self.pinned, self.plain = budget, [], self.CH, 0, 0

    def put(self, t):
        n = t.numel() * t.element_size()
        if self.used + n > self.CH:
            if (len(self.chunks) + 1) * self.CH > self.budget or n > self.CH:
                self.plain += n
                return t.contiguous()
            self.chunks.append(torch.empty(self.CH, dtype=torch.uint8).pin_memory())
            self.used = 0
        v = self.chunks[-1][self.used:self.used + n].view(t.dtype).view(t.shape)
        v.copy_(t)
        self.used += (n + 255) // 256 * 256
        self.pinned += n
        return v


class Loader:
    """Поток-помощник: задания на подкачку по порядку; ждёт освобождения буфера, копирует кусками."""

    def __init__(self):
        self.q, self.cs = queue.Queue(), torch.cuda.Stream()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        torch.cuda.set_device(0)
        CH = 10 * 2**20
        while True:
            job = self.q.get()
            if job["wait"] is not None:
                job["wait"]["done"].wait()
                self.cs.wait_event(job["wait"]["ev"])
            pend = []
            for src, dst in job["pairs"]:
                for a in range(0, src.numel(), CH):
                    with torch.cuda.stream(self.cs):
                        dst[a:a + CH].copy_(src[a:a + CH], non_blocking=True)
                        e = torch.cuda.Event(); e.record(self.cs)
                    pend.append(e)
                    if len(pend) > 2:
                        pend.pop(0).synchronize()
            ev = torch.cuda.Event(); ev.record(self.cs)
            job["ev"] = ev
            job["done"].set()


def _u8(a):
    """numpy (в т.ч. memmap) -> плоский torch uint8 без копии."""
    return torch.from_numpy(np.asarray(a).reshape(-1).view(np.uint8))


class Engine:
    def __init__(self, name, maxt=2048, cpu=True, resident=None, log=print):
        t0 = time.time()
        ck = self.ck = Ckpt(name)
        self.L, self.D, self.H, self.KV, self.HD, self.FF, self.eps = ck.L, ck.D, ck.H, ck.KV, ck.HD, ck.FF, ck.eps
        self.maxt, self.log = maxt, log
        dev = "cuda"
        self.inv = (1.0 / (ck.theta ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))).to(dev)
        self.inv_np = self.inv.cpu().numpy()
        self.emb_in = ck.mm("model.embed_tokens.weight")
        P = ck.layer_params()
        tune([(N, K, m) for (N, K), m in zip((P["qkv"], P["o"], P["gu"], P["down"]), (0, 1, 0, 2))], log=log)
        self.plan = make_plan(ck, maxt, cpu=cpu, force_resident=resident, log=log)
        self.place = self.plan["place"]
        self.head = Head(ck.mm(ck.head_name()))
        arena = Arena(self.plan["pin_budget"])
        self.small, self.W, self.Kc, self.Vc = [], [], {}, {}
        for i in range(self.L):
            pl = self.place[i]
            p = f"model.layers.{i}."
            sm = {"ln1": ck.t(p + "input_layernorm.weight"), "ln2": ck.t(p + "post_attention_layernorm.weight"),
                  "qn": ck.t(p + "self_attn.q_norm.weight"), "kn": ck.t(p + "self_attn.k_norm.weight")}
            self.small.append({k: v.to(dev) for k, v in sm.items()})
            if pl == "cpu":
                self.W.append(CpuLayer(ck.layer_mm(i), sm, ck, maxt))
                continue
            w = ck.layer(i)
            c = {n: compress(w[n].to(dev)) for n in MATS}
            if pl == "gpu":
                self.W.append(c)
            else:
                pk = {}
                for n in MATS:
                    b, lay = _pack(c[n])
                    pk[n] = (arena.put(b.cpu()), lay)
                self.W.append(pk)
            del w, c
            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev)
            self.Vc[i] = torch.zeros_like(self.Kc[i])
            torch.cuda.empty_cache()
        self.arena = arena
        self.stream_ids = [i for i in range(self.L) if self.place[i] == "stream"]
        self.cpu_ids = [i for i in range(self.L) if self.place[i] == "cpu"]
        self.all_gpu = not self.stream_ids and not self.cpu_ids
        if not self.all_gpu:
            self.loader = Loader()
            self.pending, self.last_rel, self.cslot = collections.deque(), {}, 0
            self.order, self.phase = None, None
            if self.stream_ids:
                mx = {n: max(self.W[l][n][0].numel() for l in self.stream_ids) for n in MATS}
                self.cbuf = [{n: torch.empty(mx[n], dtype=torch.uint8, device=dev) for n in MATS} for _ in range(2)]
            if self.cpu_ids:
                self.bbuf = {n: torch.empty(*P[n], dtype=torch.bfloat16, device=dev) for n in MATS}
                self.Ks = torch.zeros(self.KV, maxt, self.HD, device=dev)
                self.Vs = torch.zeros_like(self.Ks)
                self.hx = torch.empty(16, self.D).pin_memory()
                self.xs = torch.cuda.Stream()
        f = lambda n: torch.zeros(16, n, device=dev)
        H, KV, HD = self.H, self.KV, self.HD
        self.xb, self.hb, self.qkvb, self.qb, self.attb, self.gub = (f(self.D), f(self.D), f((H + 2 * KV) * HD), f(H * HD),
                                                                      f(H * HD), f(2 * self.FF))
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.graphs = {}
        self.ctx = []  # слова, чьи k, v уже в кэше (по позициям)
        self.load_s = time.time() - t0
        log(f"загружено за {self.load_s:.0f} с: видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ, "
            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")

    # ---------------- подкачка ----------------
    def _request(self, l):
        if l is None:
            return
        kind = "c" if self.place[l] == "stream" else "b"
        if kind == "c":
            s = self.cslot; self.cslot ^= 1
            pairs = [(self.W[l][n][0], self.cbuf[s][n][:self.W[l][n][0].numel()]) for n in MATS]
        else:
            s = 0
            mm, H, KV, HD, FF, D = self.W[l].W, self.H, self.KV, self.HD, self.FF, self.D
            fl = {n: self.bbuf[n].view(-1).view(torch.uint8) for n in MATS}
            q, k = H * HD * D * 2, KV * HD * D * 2
            g = FF * D * 2
            pairs = [(_u8(mm["q"]), fl["qkv"][:q]), (_u8(mm["k"]), fl["qkv"][q:q + k]), (_u8(mm["v"]), fl["qkv"][q + k:]),
                     (_u8(mm["o"]), fl["o"]), (_u8(mm["gate"]), fl["gu"][:g]), (_u8(mm["up"]), fl["gu"][g:]),
                     (_u8(mm["down"]), fl["down"])]
        rel = {"done": threading.Event(), "ev": None}
        job = {"l": l, "kind": kind, "s": s, "pairs": pairs, "wait": self.last_rel.get((kind, s)), "done": threading.Event(),
               "ev": None, "rel": rel}
        self.last_rel[(kind, s)] = rel
        self.pending.append(job)
        self.loader.q.put(job)

    def _acquire(self, l):
        job = self.pending.popleft()
        assert job["l"] == l, (job["l"], l)
        job["done"].wait()
        torch.cuda.current_stream().wait_event(job["ev"])
        self.cur = job
        if job["kind"] == "c":
            return {n: _unpack(self.cbuf[job["s"]][n], self.W[l][n][1]) for n in MATS}
        return self.bbuf

    def _release(self, job=None):
        job = job or self.cur
        ev = torch.cuda.Event(); ev.record()
        job["rel"]["ev"] = ev
        job["rel"]["done"].set()
        self._request(next(self.order, None))

    def _set_phase(self, phase):
        if self.all_gpu or self.phase == phase:
            return
        while self.pending:  # дождаться и отпустить начатую подкачку прошлой фазы
            job = self.pending.popleft()
            job["done"].wait()
            ev = torch.cuda.Event(); ev.record()
            job["rel"]["ev"] = ev
            job["rel"]["done"].set()
        if phase == "prefill":
            self.order = iter([l for l in range(self.L) if self.place[l] != "gpu"])
        else:
            self.order = itertools.cycle(self.stream_ids) if self.stream_ids else iter(())
        self.phase = phase
        self._request(next(self.order, None))
        self._request(next(self.order, None))

    # ---------------- слои ----------------
    def _gpu_layer(self, l, W, x, T):
        """Слой на видеокарте для T строк x (позиции self.pos .. +T-1). W — сжатые матрицы."""
        H, KV, HD, D, s = self.H, self.KV, self.HD, self.D, self.small[l]
        K, V = self.Kc[l], self.Vc[l]
        if T == 1:
            qkv, att, gu = self.qkvb[:1], self.attb[:1], self.gub[:1]
            gemvc1(x, W["qkv"], qkv, 0, lnw=s["ln1"], eps=self.eps)
            rope_attn_k[(H,)](qkv, s["qn"], s["kn"], self.inv, self.pos, K, V, att, self.maxt, self.eps, HD ** -0.5,
                              NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)
            gemvc1(att, W["o"], x, 1)
            gemvc1(x, W["gu"], gu, 0, lnw=s["ln2"], eps=self.eps)
            gemvc1(gu, W["down"], x, 2)
            return
        self._batch_layer(l, x, T, W, K, V, gemvc_tc)

    def _batch_layer(self, l, x, T, W, K, V, mm):
        H, KV, HD, D, s = self.H, self.KV, self.HD, self.D, self.small[l]
        h, qkv, q, att, gu = self.hb[:T], self.qkvb[:T], self.qb[:T], self.attb[:T], self.gub[:T]
        DP = triton.next_power_of_2(D)
        rmsg_k[(T,)](x, s["ln1"], h, self.eps, D, DP=DP)
        mm(h, W["qkv"], qkv, 0, T)
        qk_rope_cache_k[(H + 2 * KV, T)](qkv, s["qn"], s["kn"], self.inv, self.pos, q, K, V, self.maxt, self.eps,
                                          NH=H, NKV=KV, HDIM=HD, half=HD // 2)
        attn_k[(H, T)](q, K, V, self.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)
        mm(att, W["o"], x, 1, T)
        rmsg_k[(T,)](x, s["ln2"], h, self.eps, D, DP=DP)
        mm(h, W["gu"], gu, 0, T)
        mm(gu, W["down"], x, 2, T)

    def _cpu_layer(self, l, x, pos0):
        T = x.shape[0]
        hx = self.hx[:T]
        self.xs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.xs):
            hx.copy_(x, non_blocking=True)
        self.xs.synchronize()
        self.W[l](hx.numpy(), int(pos0), self.inv_np)
        with torch.cuda.stream(self.xs):
            x.copy_(hx, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.xs)

    def _layers(self, x, T, pos0):
        for l in range(self.L):
            pl = self.place[l]
            if pl == "gpu":
                self._gpu_layer(l, self.W[l], x, T)
            elif pl == "stream":
                self._gpu_layer(l, self._acquire(l), x, T)
                self._release()
            else:
                self._cpu_layer(l, x, pos0)
        rmsg_k[(T,)](x, self.ck_norm, self.hb[:T], self.eps, self.D, DP=triton.next_power_of_2(self.D))

    @property
    def ck_norm(self):
        if not hasattr(self, "_norm"):
            self._norm = self.ck.t("model.norm.weight").cuda()
        return self._norm

    def _emb(self, tokens):
        e = np.asarray(self.emb_in[np.asarray(tokens)])
        return torch.from_numpy(e.view(np.int16)).view(torch.bfloat16).float()

    # ---------------- проходы ----------------
    def _decode(self, tokens, pos0):
        """T <= 16 слов: строки нормированного выхода в self.hb[:T] и границы словаря."""
        T = len(tokens)
        self._set_phase("decode")
        if self.all_gpu:
            B = next(b for b in BUCKETS if b >= T)
            self.xb[:B].zero_()
            self.xb[:T].copy_(self._emb(tokens))
            self.pos.fill_(pos0)
            if B not in self.graphs:
                self._capture(B)  # пробные прогоны портят вход — заполнить заново
                self.xb[:B].zero_()
                self.xb[:T].copy_(self._emb(tokens))
                self.pos.fill_(pos0)
            self.graphs[B].replay()
            return T
        self.xb[:T].copy_(self._emb(tokens))
        self.pos.fill_(pos0)
        self._layers(self.xb[:T], T, pos0)
        self.head.bounds(self.hb[:T], T)
        return T

    def _capture(self, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._layers(self.xb[:B], B, 0)
                self.head.bounds(self.hb[:B], B)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._layers(self.xb[:B], B, 0)
            self.head.bounds(self.hb[:B], B)
        self.graphs[B] = g

    def _prefill(self, tokens, start):
        """Слова tokens на позициях start.. (послойно, кусками по 16); возвращает выход последнего слова в hb[:1]."""
        n = len(tokens)
        self._set_phase("prefill")
        X = self._emb(tokens).cuda()
        D, H, KV, HD = self.D, self.H, self.KV, self.HD
        for l in range(self.L):
            pl = self.place[l]
            if pl == "cpu":
                W = self._acquire(l)
                cl = self.W[l]
                if start:
                    self.Ks[:, :start].copy_(torch.from_numpy(cl.K[:, :start]))
                    self.Vs[:, :start].copy_(torch.from_numpy(cl.V[:, :start]))
            else:
                W = self.W[l] if pl == "gpu" else self._acquire(l)
            for c in range(0, n, 16):
                T = min(16, n - c)
                xc = X[c:c + T]
                self.pos.fill_(start + c)
                if pl == "cpu":
                    self._batch_layer(l, xc, T, W, self.Ks, self.Vs, gemv)
                elif T == 1:
                    self._gpu_layer(l, W, xc, 1)
                else:
                    self._batch_layer(l, xc, T, W, self.Kc[l], self.Vc[l], gemvc_tc)
            if pl == "cpu":
                cl.K[:, start:start + n] = self.Ks[:, start:start + n].cpu().numpy()
                cl.V[:, start:start + n] = self.Vs[:, start:start + n].cpu().numpy()
            if pl != "gpu":
                self._release()
        rmsg_k[(1,)](X[n - 1:n], self.ck_norm, self.hb[:1], self.eps, D, DP=triton.next_power_of_2(D))
        self.head.bounds(self.hb[:1], 1)

    def process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None):
        """Прогнать слова tokens с позиции pos0 (k, v раньше pos0 уже в кэше). Возвращает выбранное следующее
        слово после каждого из них (после чтения запроса > 16 слов — только после последнего)."""
        if pos0 + len(tokens) > self.maxt:
            raise ValueError(f"текст длиннее {self.maxt} слов")
        self.ctx = self.ctx[:pos0] + list(tokens)
        if len(tokens) > 16:
            self._prefill(tokens, pos0)
            return self.head.pick(self.hb[:1], 1, temperature, top_p, rng, run_bounds=False)
        T = self._decode(tokens, pos0)
        return self.head.pick(self.hb[:T], T, temperature, top_p, rng, run_bounds=False)

    def common(self, seq):
        n = 0
        for a, b in zip(self.ctx, seq):
            if a != b:
                break
            n += 1
        return n

    # ---------------- генерация ----------------
    def generate(self, prompt, max_new=256, temperature=0.0, top_p=1.0, seed=None, draft=None, k=7, stats=None):
        """Слова по одному (генератор). Жадный выбор + draft (другой Engine той же семьи) — угадывание вперёд,
        текст тот же, что без него. stats (dict) заполняется: prompt_n, prompt_ms, gen_n, gen_ms."""
        rng = np.random.default_rng(seed)
        P = len(prompt)
        c = min(self.common(prompt), P - 1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tok = self.process(prompt[c:], c, temperature, top_p, rng)[-1]
        t1 = time.perf_counter()
        st = stats if stats is not None else {}
        st.update(prompt_n=P - c, prompt_ms=(t1 - t0) * 1000, gen_n=0, gen_ms=0.0)
        gen = [tok]
        yield tok
        while len(gen) < max_new and gen[-1] not in self.ck.eos:
            if temperature <= 0 and draft is not None:
                seq = prompt + gen
                d = draft.draft(seq, min(k, max_new - len(gen) - 1))
                res = self.process([gen[-1]] + d, len(seq) - 1)
                n = 0
                while n < len(d) and d[n] == res[n]:
                    n += 1
                new = d[:n] + [res[n]]
            else:
                new = self.process([gen[-1]], P + len(gen) - 1, temperature, top_p, rng)
            for t in new:
                if len(gen) >= max_new or gen[-1] in self.ck.eos:
                    break
                gen.append(t)
                st["gen_n"] = len(gen) - 1
                st["gen_ms"] = (time.perf_counter() - t1) * 1000
                yield t

    # ---------------- угадыватель ----------------
    def draft(self, seq, k):
        """Этот движок как угадыватель: жадно k слов после seq."""
        if k <= 0:
            return []
        c = min(self.common(seq), len(seq) - 1)
        out = [self.process(seq[c:], c)[-1]]
        for i in range(k - 1):
            out.append(self.process([out[-1]], len(seq) + i)[0])
        return out
