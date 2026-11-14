"""Движок llm_app: точная генерация (текст как у fp32) на видеокарте NVIDIA с любым объёмом памяти.

Слои по плану памяти (plan.py):
  gpu    — сжатые без потерь 12 бит, постоянно в видеокарте;
  stream — сжатые 12 бит в закреплённой RAM, подкачиваются по шине (поток-помощник, куски по 10 МБ);
  cpu    — считаются процессором прямо из файла модели (memmap, bf16, счёт fp32), пока следующий слой едет по шине.
Чтение запроса — послойно: каждый слой (в т.ч. подкачанный) сразу обрабатывает весь запрос кусками по 16 слов;
слои процессора на это время подкачиваются в видеокарту в bf16. Начало, совпавшее с прошлым запросом, не
пересчитывается. Выбор слова — head.py (int8-словарь в видеокарте + точные строки из файла).
"""
from ramvi_engine.config import env
import collections
import itertools
import queue
import threading
import time

import numpy as np
import torch
import triton

from ramvi_engine.cpu.cpu_layer import CpuLayer
from ramvi_engine.generation.head import Head
from ramvi_engine.formats.model import Ckpt
from ramvi_engine.planning.plan import GiB, make_plan
from ramvi_engine.kernels.tkernels import attn_k, gemv, qk_rope_cache_k, rmsg_k, rope_attn_k
from ramvi_engine.planning.tune import tune
from ramvi_engine.kernels.wcomp import compress, gemvc1, gemvc_tc

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
    """Подкачка с передачей ошибок, ограниченными ожиданиями и явным закрытием."""
    TIMEOUT = 60.0

    def __init__(self):
        self.q, self.cs = queue.Queue(), torch.cuda.Stream()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="llm-loader", daemon=True)
        self.thread.start()

    @classmethod
    def wait(cls, job, cancel=None):
        end = time.monotonic() + cls.TIMEOUT
        while not job["done"].wait(min(0.05, max(0, end - time.monotonic()))):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("подкачка закрыта")
            if time.monotonic() >= end:
                raise TimeoutError("истекло время ожидания подкачки")
        if job.get("error") is not None:
            raise RuntimeError("ошибка подкачки весов") from job["error"]

    def _event(self, event):
        # Кусок 10 МБ копируется < 1 мс. Опрос со сном здесь нельзя: на Windows wait(0.001) спит ~16 мс
        # (шаг таймера), и подкачка слоя замедлялась в ~20 раз. Остановка проверяется перед каждым куском.
        if self._stop.is_set():
            raise RuntimeError("подкачка закрыта")
        event.synchronize()

    def _run(self):
        CH = 10 * 2**20
        failure = None
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                if failure is not None:
                    raise RuntimeError("загрузчик ранее завершился с ошибкой") from failure
                if self._stop.is_set():
                    raise RuntimeError("подкачка закрыта")
                torch.cuda.set_device(self.cs.device)
                if job["wait"] is not None:
                    self.wait(job["wait"], self._stop)
                    self.cs.wait_event(job["wait"]["ev"])
                pend = []
                for src, dst in job["pairs"]:
                    for a in range(0, src.numel(), CH):
                        if self._stop.is_set():
                            raise RuntimeError("подкачка закрыта")
                        with torch.cuda.stream(self.cs):
                            dst[a:a + CH].copy_(src[a:a + CH], non_blocking=True)
                            event = torch.cuda.Event(); event.record(self.cs)
                        pend.append(event)
                        if len(pend) > 2:
                            self._event(pend.pop(0))
                event = torch.cuda.Event(); event.record(self.cs)
                job["ev"] = event
            except Exception as exc:
                failure = exc
                job["error"] = exc
                if job.get("rel") is not None:
                    job["rel"]["error"] = exc
                    job["rel"]["done"].set()
            finally:
                job["done"].set()

    def close(self):
        self._stop.set()
        # Освобождаем ожидание release, если потребитель уже отменён.
        self.q.put(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("поток подкачки не завершился")
        # Поток CPU мог закончить, когда последняя DMA-копия ещё выполняется.
        # Не освобождаем её буферы до завершения команд CUDA.
        end = time.monotonic() + self.TIMEOUT
        while not self.cs.query():
            if time.monotonic() >= end:
                raise TimeoutError("копирование CUDA не завершилось при закрытии")
            time.sleep(0.001)


def _u8(a):
    """numpy (в т.ч. memmap) -> плоский torch uint8 без копии."""
    return torch.from_numpy(np.asarray(a).reshape(-1).view(np.uint8))


class Engine:
    def __init__(self, name, maxt=2048, cpu=True, resident=None, log=print):
        if not isinstance(maxt, int) or isinstance(maxt, bool) or maxt < 1:
            raise ValueError("maxt должно быть положительным целым")
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

    def close(self):
        loader = getattr(self, "loader", None)
        if loader is not None:
            loader.close()

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
        self.loader.wait(job)
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
        if self.all_gpu or (self.phase == phase and phase != "prefill"):
            return
        while self.pending:  # дождаться и отпустить начатую подкачку прошлой фазы
            job = self.pending.popleft()
            self.loader.wait(job)
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
            B = T  # граф обрабатывает только реальные строки, без записи KV за конец контекста
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
                self.head.bounds(self.hb[:B], B)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self._layers(self.xb[:B], B, 0)
            self.head.bounds(self.hb[:B], B)
        self.graphs[B] = g

    def _prefill(self, tokens, start):
        """Слова tokens на позициях start.. (послойно, кусками по 16); возвращает выход последнего слова в hb[:1]."""
        from ramvi_engine.kernels.prefill import enabled, layer
        blocked = enabled(len(tokens))
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
            if blocked:
                layer(self, l, W, X, start,
                      self.Ks if pl == "cpu" else self.Kc[l],
                      self.Vs if pl == "cpu" else self.Vc[l])
            else:
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

    def _validate_process(self, tokens, pos0):
        if not isinstance(pos0, int) or isinstance(pos0, bool) or not 0 <= pos0 <= len(self.ctx):
            raise ValueError("позиция не принадлежит валидному префиксу")
        if not tokens:
            raise ValueError("пустой проход не поддерживается")
        vocab = getattr(self, "V", None)
        if vocab is None and hasattr(self, "emb_in"):
            vocab = self.emb_in.shape[0]
        for token in tokens:
            try:
                index = token.__index__()
            except (AttributeError, TypeError) as exc:
                raise ValueError("ID токена должен быть целым") from exc
            if isinstance(token, bool) or index < 0 or vocab is not None and index >= vocab:
                raise ValueError("ID токена вне словаря")

    def process(self, tokens, pos0, temperature=0.0, top_p=1.0, rng=None, last_only=False):
        """Прогнать слова tokens с позиции pos0 (k, v раньше pos0 уже в кэше). Возвращает выбранное следующее
        слово после каждого из них (после чтения запроса > 16 слов — только после последнего)."""
        self._validate_process(tokens, pos0)
        if pos0 + len(tokens) > self.maxt:
            raise ValueError(f"текст длиннее {self.maxt} слов")
        previous = self.ctx[:pos0]
        try:
            if len(tokens) > 16:
                self._prefill(tokens, pos0)
                result = self.head.pick(self.hb[:1], 1, temperature, top_p, rng, run_bounds=False)
            else:
                T = self._decode(tokens, pos0)
                if last_only:
                    result = self.head.pick(self.hb[T - 1:T], 1, temperature, top_p, rng, run_bounds=True)
                else:
                    result = self.head.pick(self.hb[:T], T, temperature, top_p, rng, run_bounds=False)
        except Exception:
            self.ctx = []  # частичный проход мог испортить любой слой кэша
            raise
        self.ctx = previous + list(tokens)
        return result


    def common(self, seq):
        n = 0
        for a, b in zip(self.ctx, seq):
            if a != b:
                break
            n += 1
        return n

    # ---------------- генерация ----------------
    def generate(self, prompt, max_new=256, temperature=0.0, top_p=1.0, seed=None, draft=None, k=None, stats=None,
                 force=None, cons=None):
        """Слова по одному (генератор). Жадный выбор + draft (другой движок той же семьи) — угадывание вперёд,
        текст тот же, что без него. k — постоянная длина черновика; k=None — по стоимости: черновик продлевается,
        пока вероятность «все слова угаданы» (произведение уверенностей угадывателя) выше порога
        thr = 2 * (шаг угадывателя + прирост прохода на слово) / проход, измеряемых на ходу.
        По умолчанию режим выбирается прежней политикой после черновика.
        RAMVI_DRAFT_GATE=1 включает экспериментальный выбор до запуска угадывателя
        по прошлым принятиям, с периодическими проверенными пробами.
        force(i) -> "step" | "spec" — принудительный режим на шаге i (для проверки, что текст не зависит от
        истории переключений). cons — ограничение грамматикой (constrain.Constraint; только движок GGUF).
        stats (dict): prompt_n, prompt_ms, gen_n, gen_ms, passes, drafted, accepted, steps1, switches."""
        import os
        gate_enabled = env("RAMVI_DRAFT_GATE", "0") == "1"
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 15):
            raise ValueError("k должно быть целым от 1 до 15 (None — адаптивно)")
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new < 0:
            raise ValueError("max_new должно быть неотрицательным целым")
        if max_new == 0:
            if stats is not None:
                stats.update(prompt_n=0, prompt_ms=0.0, first_token_ms=0.0,
                             gen_n=0, decode_n=0, gen_ms=0.0, passes=0, drafted=0, accepted=0)
            return
        if not prompt:
            raise ValueError("пустой запрос не поддерживается; передайте BOS или текст")
        rng = np.random.default_rng(seed)
        P = len(prompt)
        if P > self.maxt:
            raise ValueError(f"запрос длиннее контекста {self.maxt}")
        # Последний выданный токен ещё не обязан иметь строку KV.
        max_new = min(max_new, self.maxt - P + 1)
        c = min(self.common(prompt), P - 1)
        if cons is not None and not getattr(self, "supports_mask", False):
            raise NotImplementedError("ограничение грамматикой поддержано только для моделей GGUF")
        mk = (lambda path=(): cons.masks(list(path))) if cons is not None else (lambda path=(): (None, 0))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        bm, _ = mk()
        tok = self.process(prompt[c:], c, temperature, top_p, rng, last_only=True,
                           **({"mask": bm} if cons else {}))[-1]
        if cons:
            cons.commit(0, 0, tok)
        t1 = time.perf_counter()
        st = stats if stats is not None else {}
        st.update(prompt_n=P - c, prompt_ms=(t1 - t0) * 1000, first_token_ms=(t1 - t0) * 1000,
                  gen_n=1, decode_n=0, gen_ms=0.0, passes=0, drafted=0, accepted=0,
                  draft_calls=0, draft_skips=0, draft_probes=0, draft_ms=0.0, verify_ms=0.0, step_ms=0.0)
        gen = [tok]
        cost = self._cost = getattr(self, "_cost", {"td": None, "pts": []})  # шаг угадывателя, (T, время прохода)
        yield tok
        while len(gen) < max_new and gen[-1] not in self.ck.eos:
            step_start = time.perf_counter()
            if temperature <= 0 and draft is not None:
                seq = prompt + gen
                kmax = min(k or 15, 15, max_new - len(gen) - 1, self.maxt - len(seq),
                           max(0, draft.maxt - len(seq)))
                # O02 включается явно: предварительный выбор и проверенные пробы.
                # Обычная политика выбирает режим ниже, после черновика.
                if force:
                    mode = force(st["passes"])
                    if mode not in ("step", "spec"):
                        raise ValueError("force должен вернуть step или spec")
                elif not gate_enabled:
                    mode = "legacy"
                elif k is not None:
                    mode = "spec"
                else:
                    mode = self._draft_plan(st["passes"], kmax, draft)
                if kmax <= 0 and mode != "legacy":
                    mode = "step"
                ta = time.perf_counter()
                if mode != "step" and kmax > 0:
                    thr = None if k else self._draft_thr()
                    d = draft.draft(seq, kmax, cum=thr)
                    st["draft_calls"] += 1
                    st["draft_probes"] += int(mode == "probe")
                else:
                    d = []
                    st["draft_skips"] += 1
                tb = time.perf_counter()
                draft_s = tb - ta if mode != "step" and kmax > 0 else 0.0
                st["draft_ms"] += draft_s * 1000
                if d:
                    td = draft_s / len(d)
                    cost["td"] = td if cost["td"] is None else 0.8 * cost["td"] + 0.2 * td
                if mode == "legacy":
                    want_step = k is None and d and self._prefer_step(draft, len(d))
                else:
                    want_step = mode == "step" or not d
                if want_step != (cost.get("last") == "step"):
                    st["switches"] = st.get("switches", 0) + 1
                cost["last"] = "step" if want_step else "spec"
                if want_step:
                    bm, _ = mk()
                    res = self.process([gen[-1]], len(seq) - 1, **({"mask": bm} if cons else {}))  # черновику мало веры — дешёвый одиночный шаг
                    if cons:
                        cons.commit(0, 0, res[0])
                    tc = time.perf_counter()
                    if cost.get("prev") == "spec" and cost.get("c1") is not None:  # первый шаг после смены
                        cost["sw"] = 0.7 * cost.get("sw", 0.0) + 0.3 * max(0.0, (tc - tb) - cost["c1"])
                    else:
                        cost["c1"] = (tc - tb) if cost.get("c1") is None else 0.8 * cost["c1"] + 0.2 * (tc - tb)
                    new = res[:1]
                    st["steps1"] = st.get("steps1", 0) + 1
                    st["step_ms"] += (tc - tb) * 1000
                    if gate_enabled:
                        cost["gate_steps"] = cost.get("gate_steps", 0) + 1
                else:
                    bm, adv = mk(d)
                    res = self.process([gen[-1]] + d, len(seq) - 1, **({"mask": bm} if cons else {}))
                    tc = time.perf_counter()
                    if cost.get("prev") != "step":  # проход после смены режима в оценку прохода не берём
                        cost["pts"] = (cost["pts"] + [(len(d) + 1, tc - tb)])[-64:]
                    n = 0
                    while n < len(d) and d[n] == res[n]:
                        n += 1
                    new = d[:n] + [res[n]]
                    st["verify_ms"] += (tc - tb) * 1000
                    if gate_enabled:
                        cost["gate_steps"] = 0
                    # Не учим оценку скорости по хвосту после EOS, который не выдаётся.
                    if gate_enabled and not any(t in self.ck.eos for t in new):
                        duration = draft_s + tc - tb
                        if st["passes"] == 0:
                            # Новый запрос может иметь совсем другой характер.
                            # Первая проверенная проба заменяет старую оценку,
                            # сохраняя уже измеренную цену одиночного шага.
                            cost.pop("gate_yield", None)
                            cost.pop("gate_s", None)
                        cost["gate_yield"] = 0.5 * cost.get("gate_yield", n + 1) + 0.5 * (n + 1)
                        cost["gate_s"] = 0.5 * cost.get("gate_s", duration) + 0.5 * duration
                        cost["gate_samples"] = cost.get("gate_samples", 0) + 1
                    if cons:
                        cons.commit(adv, n, res[n])
                    st["drafted"] += len(d)
                    st["accepted"] += n
                    if d:  # поправка уверенности угадывателя: принято / ожидалось
                        e = self._expect(draft, len(d))
                        if e > 0.05:
                            cost["cal"] = min(1.5, max(0.2, 0.85 * cost.get("cal", 1.0) + 0.15 * n / e))
            else:
                bm, _ = mk()
                new = self.process([gen[-1]], P + len(gen) - 1, temperature, top_p, rng, **({"mask": bm} if cons else {}))
                if cons:
                    cons.commit(0, 0, new[0])
            st["gen_ms"] += (time.perf_counter() - step_start) * 1000
            st["passes"] += 1
            if draft is not None and temperature <= 0:
                cost["prev"] = cost.get("last")
            for t in new:
                if len(gen) >= max_new or gen[-1] in self.ck.eos:
                    break
                gen.append(t)
                st["gen_n"] = len(gen)
                st["decode_n"] = len(gen) - 1
                yield t

    DRAFT_PROBE_INTERVAL = 8

    def _draft_plan(self, pass_index, kmax, draft):
        """Решение без запуска draft: реальная отдача прошлых проверок / полная цена.

        В режиме step через DRAFT_PROBE_INTERVAL шагов выполняется проверяемая
        контрольная попытка. Первый проход нового запроса также обновляет статистику.
        Фиксированный k и force обходят эту политику.
        """
        if kmax <= 0:
            return "step"
        c = self._cost
        key = (id(draft), tuple(getattr(self, "place", ())))
        if c.get("gate_key") != key:
            for name in ("gate_yield", "gate_s", "gate_samples", "gate_steps"):
                c.pop(name, None)
            c["gate_key"] = key
        if pass_index == 0:
            return "probe"
        if c.get("gate_samples", 0) < 3:
            return "spec"
        if c.get("c1") is None:
            return "step"  # измерить настоящий одиночный шаг, не оплачивая draft
        last_step = c.get("last") == "step"
        if last_step and c.get("gate_steps", 0) >= self.DRAFT_PROBE_INTERVAL:
            return "probe"
        switching = c.get("sw", 0.0)
        spec_s = c["gate_s"] + (switching if last_step else 0.0)
        step_s = c["c1"] + (switching if not last_step else 0.0)
        spec_rate = c["gate_yield"] / max(spec_s, 1e-6)
        # CPU-режим оплачивает и периодические проверенные пробы. Сравнение
        # с идеальным CPU без проб давало ложные переключения и регрессию.
        interval = self.DRAFT_PROBE_INTERVAL
        step_rate = (interval + c["gate_yield"]) / max(
            interval * step_s + c["gate_s"] + 2 * switching, 1e-6)
        if last_step:
            return "spec" if spec_rate > step_rate * 1.15 else "step"
        return "step" if step_rate > spec_rate * 1.15 else "spec"

    @staticmethod
    def _expect(draft, n):
        """Ожидаемое число принятых слов черновика: сумма произведений уверенностей."""
        cf = getattr(draft, "draft_confs", None)
        if not cf:
            return 0.0
        e, p = 0.0, 1.0
        for c in cf[:n]:
            p *= c
            e += p
        return e

    def _prefer_step(self, draft, nd):
        """Одиночный шаг выгоднее проверки черновика? (по измеренным стоимостям и ожидаемому приёму)."""
        c = self._cost
        if not c["pts"] or len(c["pts"]) < 3:
            return False
        if c.get("c1") is None:
            return len(c["pts"]) % 8 == 3  # изредка пробуем шаг, чтобы узнать его стоимость
        T = np.array([p[0] for p in c["pts"]], float)
        y = np.array([p[1] for p in c["pts"]], float)
        if T.std() > 0.5:
            b, a = np.polyfit(T, y, 1)
        else:
            a, b = y.mean(), 0.0
        cn = a + b * (nd + 1)
        e = self._expect(draft, nd) * c.get("cal", 1.0)
        sw = c.get("sw", 0.0)  # цена переключения режима (измеряется: лишнее время первого прохода после смены)
        spec_rate = (1.0 + e) / max(cn + (sw if c.get("last") == "step" else 0.0), 1e-4)
        step_rate = 1.0 / max(c["c1"] + (sw if c.get("last") != "step" else 0.0), 1e-4)
        h = 1.15  # гистерезис: менять режим, только если другой заметно выгоднее
        if c.get("last") == "step":
            return not (spec_rate > step_rate * h)
        return step_rate > spec_rate * h

    def _draft_thr(self):
        """Порог для произведения уверенностей угадывателя: 2 * (шаг угадывателя + прирост прохода на слово) / проход."""
        c = getattr(self, "_cost", None)
        if not c or c["td"] is None or len(c["pts"]) < 4:
            return 0.05
        T = np.array([p[0] for p in c["pts"]], float)
        y = np.array([p[1] for p in c["pts"]], float)
        if T.std() > 0.5:
            b, a = np.polyfit(T, y, 1)
        else:
            a, b = y.mean(), 0.0
        a = max(a, 1e-3)
        return float(min(0.6, max(0.01, 2 * (c["td"] + max(b, 0.0)) / a)))

    # ---------------- угадыватель ----------------
    last_conf = None  # уверенность (вероятность выбранного слова) после process, если движок её считает

    def draft(self, seq, k, cum=None):
        """Этот движок как угадыватель: жадно до k слов после seq; с cum — обрывает, когда произведение
        уверенностей (вероятность, что все слова черновика угаданы) падает ниже cum."""
        if k <= 0:
            return []
        c = min(self.common(seq), len(seq) - 1)
        out = [self.process(seq[c:], c)[-1]]
        confs = [self.last_conf[-1] if self.last_conf else 1.0]
        p = confs[0]
        for i in range(k - 1):
            if cum is not None and p < cum:
                break
            out.append(self.process([out[-1]], len(seq) + i)[0])
            confs.append(self.last_conf[-1] if self.last_conf else 1.0)
            p *= confs[-1]
        self.draft_confs = confs
        return out
