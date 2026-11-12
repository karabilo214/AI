"""Движок для любой Qwen3 (проверяется на Qwen3-4B, 8 ГБ в bf16) на видеокарте 8 ГБ. Только одно слово (T=1).

Режимы весов слоёв:
  "c12"   — сжатые без потерь (12 бит, wcomp) и целиком в видеокарте; шаг записан в CUDA-граф;
  "off16" — bf16 в закреплённой памяти RAM, слой подкачивается по шине перед счётом (2 буфера, подкачка
            следующего слоя идёт во время счёта текущего);
  "off12" — то же, но по шине идут сжатые 12-битные веса.
Словарь (общий с входом): bf16 в видеокарте + int8-копия для точного выбора слова (vocab8).
Чтение запроса — теми же шагами по одному слову (запрос лежит в hist, шаг сам берёт следующее слово).
"""
import glob
import json
import os
import time

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from fused2 import rope_attn_k
from fused_decode import attn_k, gemv, qk_rope_cache_k
from vocab8 import Vocab8
from q4 import CFGQ, gemvq4, quant4
from wcomp import CFG1, CFGT, compress, gemvc1, gemvc_tc

# настройки ядер, подобранные в графе (sweep_tc.py, sweep_q4.py)
CFGQ.update({(6144, 2560, 0): (16, 2), (2560, 4096, 1): (8, 1), (19456, 2560, 0): (8, 1), (2560, 9728, 2): (8, 1)})
CFG1.update({(6144, 2560, 0): (4, 2), (2560, 4096, 1): (8, 2), (19456, 2560, 0): (4, 2), (2560, 9728, 2): (4, 2),
             (4096, 1024, 0): (2, 1), (1024, 2048, 1): (2, 1), (6144, 1024, 0): (2, 1), (1024, 3072, 2): (4, 2)})
CFGT.update({(6144, 2560, 0): (64, 4, 1), (2560, 4096, 1): (64, 4, 2), (19456, 2560, 0): (64, 4, 1), (2560, 9728, 2): (32, 2, 4),
             (4096, 1024, 0): (16, 2, 1), (1024, 2048, 1): (16, 2, 1), (6144, 1024, 0): (64, 4, 1), (1024, 3072, 2): (32, 2, 1)})

MATS = ("qkv", "o", "gu", "down")
PIN_GIB = float(os.environ.get("PIN_GIB", "6.0"))


def snap(name):
    root = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    return sorted(glob.glob(os.path.join(root, f"models--Qwen--{name}", "snapshots", "*")))[-1]


class Ckpt:
    """Чтение весов из safetensors по одному тензору (bf16, на CPU)."""

    def __init__(self, name):
        d = snap(name)
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.files = {}
        for f in glob.glob(os.path.join(d, "*.safetensors")):
            h = safe_open(f, framework="pt", device="cpu")
            for k in h.keys():
                self.files[k] = h

    def t(self, k):
        return self.files[k].get_tensor(k)

    def layer(self, i):
        p = f"model.layers.{i}."
        a, m = p + "self_attn.", p + "mlp."
        return {"qkv": torch.cat([self.t(a + "q_proj.weight"), self.t(a + "k_proj.weight"), self.t(a + "v_proj.weight")]),
                "o": self.t(a + "o_proj.weight"),
                "gu": torch.cat([self.t(m + "gate_proj.weight"), self.t(m + "up_proj.weight")]),
                "down": self.t(m + "down_proj.weight"),
                "ln1": self.t(p + "input_layernorm.weight"), "ln2": self.t(p + "post_attention_layernorm.weight"),
                "qn": self.t(a + "q_norm.weight"), "kn": self.t(a + "k_norm.weight")}


@triton.jit
def rmsg_k(X, W, O, eps, D, DP: tl.constexpr):
    t = tl.program_id(0)
    X += t * D
    O += t * D
    r = tl.arange(0, DP)
    m = r < D
    x = tl.load(X + r, mask=m, other=0.0)
    v = tl.sum(x * x, 0) / D
    tl.store(O + r, tl.load(W + r, mask=m, other=0.0).to(tl.float32) * (x * tl.rsqrt(v + eps)), mask=m)


def _pack(c):
    """Сжатую матрицу -> (байты, раскладка) для подкачки одним куском."""
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


class BigEngine:
    def __init__(self, name="Qwen3-4B", mode="c12", maxt=1024, share=None, resident=0, cpu_alt=False):
        """mode: c12 / off16 / off12 / q4 (4.5 бит с потерями, целиком в видеокарте — для угадывателя).
        share: другой движок той же модели — взять у него словарь (bf16 и int8), чтобы не держать дважды.
        resident: (off16/off12) сколько первых слоёв держать в видеокарте постоянно, остальные подкачивать.
        cpu_alt: слои вне видеокарты через один считает процессор (cpu_layer, веса bf16 из RAM), пока следующий
        слой едет по шине."""
        ck = Ckpt(name)
        cfg = ck.cfg
        assert cfg["tie_word_embeddings"]
        self.mode = mode
        self.res = mode in ("c12", "q4")  # все слои в видеокарте
        self.R = resident
        L = cfg["num_hidden_layers"]
        self.cpu_ids = set(range(resident + 1, L, 2)) if (cpu_alt and not self.res) else set()
        self.stream_ids = [] if self.res else [i for i in range(resident, L) if i not in self.cpu_ids]
        self.D, self.L, self.H, self.KV = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.HD, self.FF, self.eps = cfg["head_dim"], cfg["intermediate_size"], cfg["rms_norm_eps"]
        self.maxt = maxt
        dev = "cuda"
        self.emb = share.emb if share else ck.t("model.embed_tokens.weight").to(dev)  # [V, D] bf16, он же словарь
        self.norm = ck.t("model.norm.weight").to(dev)
        self.inv = (1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.HD, 2, dtype=torch.float32) / self.HD))).to(dev)
        self.small, self.W, self.lay = [], [], []
        self.pinned = self.unpinned = 0
        t0 = time.time()
        for i in range(self.L):
            w = ck.layer(i)
            self.small.append({k: w[k].to(dev) for k in ("ln1", "ln2", "qn", "kn")})
            if i in self.cpu_ids:
                from cpu_layer import CpuLayer
                self.W.append(CpuLayer(w, cfg, maxt))
            elif mode == "c12" or (mode == "off12" and i < resident):
                self.W.append({n: compress(w[n].to(dev)) for n in MATS})
            elif mode == "q4":
                self.W.append({n: quant4(w[n].to(dev)) for n in MATS})
            elif mode == "off16" and i < resident:
                self.W.append({n: w[n].to(dev).contiguous() for n in MATS})
            elif mode == "off16":
                self.W.append({n: self._pin(w[n].contiguous()) for n in MATS})
            elif mode == "off12":
                pk = {}
                for n in MATS:
                    b, lay = _pack(compress(w[n].to(dev)))
                    pk[n] = (self._pin(b.cpu()), lay)
                self.W.append(pk)
            torch.cuda.empty_cache()
        self.load_s = time.time() - t0
        D, H, KV, HD = self.D, self.H, self.KV, self.HD
        self.inv_np = self.inv.cpu().numpy()
        if mode.startswith("off") and self.stream_ids:  # два буфера на видеокарте под слой
            if mode == "off16":
                self.slots = [{n: torch.empty_like(self.W[self.stream_ids[-1]][n], device=dev) for n in MATS} for _ in range(2)]
            else:
                mx = {n: max(self.W[l][n][0].numel() for l in self.stream_ids) for n in MATS}
                self.slots = [{n: torch.empty(mx[n], dtype=torch.uint8, device=dev) for n in MATS} for _ in range(2)]
            self.cs = torch.cuda.Stream()
            self.loaded = [torch.cuda.Event() for _ in range(2)]
            self.used = [torch.cuda.Event() for _ in range(2)]
        self.v8 = share.v8 if share else Vocab8(self.emb)
        f = lambda n: torch.zeros(16, n, device=dev)
        self.xb, self.hb, self.qkvb, self.attb, self.gub, self.qb = (f(D), f(D), f((H + 2 * KV) * HD), f(H * HD),
                                                                      f(2 * self.FF), f(H * HD))
        self.x, self.h, self.qkv, self.att, self.gu = self.xb[:1], self.hb[:1], self.qkvb[:1], self.attb[:1], self.gub[:1]
        self.logits = torch.zeros(16, self.emb.shape[0], device=dev)
        self.tok = torch.zeros(16, dtype=torch.long, device=dev)
        self.vout = torch.zeros(16, dtype=torch.long, device=dev)
        self.vgraphs = {}
        self.K = torch.zeros(self.L, KV, maxt, HD, device=dev)
        self.V = torch.zeros_like(self.K)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.plen = torch.zeros(1, dtype=torch.long, device=dev)
        self.hist = torch.zeros(maxt + 1, dtype=torch.long, device=dev)
        self.graph = None
        self.k = 0  # счётчик подкачанных слоёв (для буферов)

    def _pin(self, t):
        """Закрепить в RAM (быстрее по шине) в пределах PIN_GIB (Windows даёт закрепить ~6.75 ГиБ, после отказа
        CUDA ломается), остальное — обычной памятью."""
        n = t.numel() * t.element_size()
        CH = 256 * 2**20  # куски ровно по 256 МиБ: torch округляет закреплённые куски до степени двойки
        if not hasattr(self, "_arena"):
            self._arena, self._used = [], CH
        if self._used + n > CH:
            if (len(self._arena) + 1) * CH > PIN_GIB * 2**30 or n > CH:
                self.unpinned += n
                return t
            self._arena.append(torch.empty(CH, dtype=torch.uint8).pin_memory())
            self._used = 0
        buf = self._arena[-1][self._used:self._used + n]
        self._used += (n + 255) // 256 * 256
        self.pinned += n
        v = buf.view(t.dtype).view(t.shape)
        v.copy_(t)
        return v

    # ---------- подкачка ----------
    def _fetch(self, j):
        """Подкачать j-й по счёту подкачиваемый слой в буфер j % 2 (поток копирования)."""
        s, l = j % 2, self.stream_ids[j % len(self.stream_ids)]
        with torch.cuda.stream(self.cs):
            self.cs.wait_event(self.used[s])
            for n in MATS:
                if self.mode == "off16":
                    self.slots[s][n].copy_(self.W[l][n], non_blocking=True)
                else:
                    src = self.W[l][n][0]
                    self.slots[s][n][:src.numel()].copy_(src, non_blocking=True)
            self.loaded[s].record(self.cs)

    def _layer_w(self, l):
        if self.res or l < self.R:
            return self.W[l]
        s = self.k % 2
        if self.cpu_ids:
            with self.cv:
                self.cv.wait_for(lambda: self.k in self.ready)
                le = self.ready.pop(self.k)
            torch.cuda.current_stream().wait_event(le)
        else:
            torch.cuda.current_stream().wait_event(self.loaded[s])
        if self.mode == "off16":
            return self.slots[s]
        return {n: _unpack(self.slots[s][n], self.W[l][n][1]) for n in MATS}

    def _cpu(self, l, x, pos0):
        """Слой на процессоре: x уходит в RAM своим потоком CUDA (не в очереди за весами), считается, возвращается."""
        T = x.shape[0]
        if not hasattr(self, "hx"):
            self.hx = torch.empty(16, self.D).pin_memory()
            self.xs = torch.cuda.Stream()
        hx = self.hx[:T]
        self.xs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.xs):
            hx.copy_(x, non_blocking=True)
        self.xs.synchronize()
        xn = hx.numpy()
        self.W[l](xn, int(pos0), self.inv_np)
        with torch.cuda.stream(self.xs):
            x.copy_(hx, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.xs)

    # ---- подкачка потоком-помощником кусками по 10 МБ (не больше 2 в очереди): маленькие пересылки x
    #      успевают проскочить между кусками, и процессор считает, пока идёт передача ----
    def _start_loader(self):
        import queue
        import threading
        self.rel_q = queue.Queue()
        self.ready, self.cv = {}, threading.Condition()
        self.rel_q.put(None); self.rel_q.put(None)  # оба буфера свободны
        threading.Thread(target=self._loader, daemon=True).start()

    def _loader(self):
        torch.cuda.set_device(0)
        CH = 10 * 2**20
        j = 0
        while True:
            ev = self.rel_q.get()
            s, l = j % 2, self.stream_ids[j % len(self.stream_ids)]
            if ev is not None:
                self.cs.wait_event(ev)
            pend = []
            for n in MATS:
                if self.mode == "off16":
                    src, dst = self.W[l][n].view(-1).view(torch.uint8), self.slots[s][n].view(-1).view(torch.uint8)
                else:
                    src = self.W[l][n][0]
                    dst = self.slots[s][n][:src.numel()]
                for a in range(0, src.numel(), CH):
                    with torch.cuda.stream(self.cs):
                        dst[a:a + CH].copy_(src[a:a + CH], non_blocking=True)
                        e = torch.cuda.Event(); e.record(self.cs)
                    pend.append(e)
                    if len(pend) > 2:
                        pend.pop(0).synchronize()
            le = torch.cuda.Event(); le.record(self.cs)
            with self.cv:
                self.ready[j] = le
                self.cv.notify_all()
            j += 1

    def _layer_done(self, l):
        if l in self.stream_ids:
            s = self.k % 2
            if self.cpu_ids:
                e = torch.cuda.Event(); e.record()
                self.rel_q.put(e)
            else:
                self.used[s].record()
                self._fetch(self.k + 2)
            self.k += 1

    # ---------- шаг ----------
    def _mm(self, x, W, y, mode, lnw=None):
        if self.mode == "q4":
            if lnw is not None:
                rmsg_k[(1,)](x, lnw, self.h, self.eps, self.D, DP=triton.next_power_of_2(self.D))
                x = self.h
            gemvq4(x, W, y, mode, 1)
        elif self.mode == "off16":
            if lnw is not None:
                rmsg_k[(1,)](x, lnw, self.h, self.eps, self.D, DP=triton.next_power_of_2(self.D))
                x = self.h
            gemv(x, W, y, mode, 1)
        else:
            gemvc1(x, W, y, mode, lnw=lnw, eps=self.eps)

    def _step(self):
        H, KV, HD = self.H, self.KV, self.HD
        tok = self.hist.index_select(0, self.pos)
        self.x.copy_(self.emb.index_select(0, tok).float())
        for l in range(self.L):
            if l in self.cpu_ids:
                self._cpu(l, self.x, self.pos.item())
                continue
            W, s = self._layer_w(l), self.small[l]
            self._mm(self.x, W["qkv"], self.qkv, 0, lnw=s["ln1"])
            rope_attn_k[(H,)](self.qkv, s["qn"], s["kn"], self.inv, self.pos, self.K[l], self.V[l], self.att, self.maxt,
                              self.eps, HD ** -0.5, NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)
            self._mm(self.att, W["o"], self.x, 1)
            self._mm(self.x, W["gu"], self.gu, 0, lnw=s["ln2"])
            self._mm(self.gu, W["down"], self.x, 2)
            self._layer_done(l)
        rmsg_k[(1,)](self.x, self.norm, self.h, self.eps, self.D, DP=triton.next_power_of_2(self.D))
        out = self.v8(self.h, 1)
        nxt = torch.where(self.pos + 1 < self.plen, self.hist.index_select(0, self.pos + 1), out)
        self.hist.index_copy_(0, self.pos + 1, nxt)
        self.pos.add_(1)

    def _mmT(self, x, W, y, mode, T):
        if self.mode == "q4":
            gemvq4(x, W, y, mode, T)
        elif self.mode == "off16":
            gemv(x, W, y, mode, T)
        else:
            gemvc_tc(x, W, y, mode, T)

    def _verify_body(self, T):
        H, KV, HD, D = self.H, self.KV, self.HD, self.D
        DP = triton.next_power_of_2(D)
        x, h, qkv, q, att, gu = self.xb[:T], self.hb[:T], self.qkvb[:T], self.qb[:T], self.attb[:T], self.gub[:T]
        x.copy_(self.emb.index_select(0, self.tok[:T]).float())
        for l in range(self.L):
            if l in self.cpu_ids:
                self._cpu(l, x, self.pos0_py)
                continue
            W, s = self._layer_w(l), self.small[l]
            rmsg_k[(T,)](x, s["ln1"], h, self.eps, D, DP=DP)
            self._mmT(h, W["qkv"], qkv, 0, T)
            qk_rope_cache_k[(H + 2 * KV, T)](qkv, s["qn"], s["kn"], self.inv, self.pos, q, self.K[l], self.V[l],
                                              self.maxt, self.eps, NH=H, NKV=KV, HDIM=HD, half=HD // 2)
            attn_k[(H, T)](q, self.K[l], self.V[l], self.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)
            self._mmT(att, W["o"], x, 1, T)
            rmsg_k[(T,)](x, s["ln2"], h, self.eps, D, DP=DP)
            self._mmT(h, W["gu"], gu, 0, T)
            self._mmT(gu, W["down"], x, 2, T)
            self._layer_done(l)
        rmsg_k[(T,)](x, self.norm, h, self.eps, D, DP=DP)
        if T <= 2:
            out = self.v8(h, T)
        else:
            lg = self.logits[:T]
            gemv(h, self.emb, lg, 0, T)
            out = lg.argmax(-1)
        self.vout[:T].copy_(out)
        return self.vout[:T]

    @torch.no_grad()
    def verify(self, tokens, pos0):
        """Проход пачки слов tokens (позиции pos0..), возвращает самые вероятные следующие слова (список).
        c12: пачка дополняется нулями до 1/2/4/8/16 и идёт записанным графом (ответы на дополнение не нужны)."""
        T = len(tokens)
        if self.res:
            B = next(b for b in (1, 2, 4, 8, 16) if b >= T)
            self.tok[:B].zero_()
            self.tok[:T].copy_(torch.as_tensor(tokens))
            self.pos.fill_(pos0)
            if B not in self.vgraphs:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._verify_body(B)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._verify_body(B)
                self.vgraphs[B] = g
            self.vgraphs[B].replay()
            return self.vout[:T].tolist()
        self.tok[:T].copy_(torch.as_tensor(tokens))
        self.pos.fill_(pos0)
        self.pos0_py = pos0
        return self._verify_body(T).tolist()

    @torch.no_grad()
    def prefill(self, prompt):
        """Запрос пачками по 16 слов; возвращает слово после запроса."""
        if self.stream_ids and self.k == 0 and not hasattr(self, "rel_q"):
            if self.cpu_ids:
                self._start_loader()
            else:
                self._fetch(0); self._fetch(1)
        for i in range(0, len(prompt), 16):
            out = self.verify(prompt[i:i + 16], i)
        return out[-1]

    def _capture(self):
        pos0 = self.pos.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            self._step()
        self.graph = g
        self.pos.copy_(pos0)

    def step(self):
        if self.res:
            if self.graph is None:
                self._capture()
            self.graph.replay()
        else:
            self._step()

    @torch.no_grad()
    def generate(self, prompt, n):
        """Жадно n слов после prompt. Возвращает (слова, мс на чтение запроса, мс на одно слово генерации)."""
        if self.stream_ids and self.k == 0 and not hasattr(self, "rel_q"):
            if self.cpu_ids:
                self._start_loader()
            else:
                self._fetch(0); self._fetch(1)
        P = len(prompt)
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        self.prefill(prompt[:-1])  # позиции 0..P-2 — пачками по 16
        self.hist[:P].copy_(torch.as_tensor(prompt, device="cuda"))
        self.plen.fill_(P)
        self.pos.fill_(P - 1)
        e1.record()
        for _ in range(n):
            self.step()
        e2.record()
        torch.cuda.synchronize()
        return self.hist[P:P + n].tolist(), e0.elapsed_time(e1), e1.elapsed_time(e2) / n


# ---------- честный эталон: обычный torch fp32, проверка готового текста одним проходом ----------
class Reference:
    """Веса bf16 в RAM, слой за слоем переводится в fp32 на видеокарте; математика как у HF Qwen3 (fp32)."""

    def __init__(self, name):
        self.ck = Ckpt(name)
        c = self.ck.cfg
        self.c = c
        self.emb = self.ck.t("model.embed_tokens.weight")
        self.norm = self.ck.t("model.norm.weight").cuda().float()
        self.L = c["num_hidden_layers"]

    @staticmethod
    def rms(x, w, eps):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    @torch.no_grad()
    def argmax(self, tokens):
        c = self.c
        H, KV, HD, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
        T = len(tokens)
        x = self.emb[torch.as_tensor(tokens)].cuda().float()
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, HD, 2, dtype=torch.int64).float().cuda() / HD))
        fr = torch.arange(T, device="cuda").float()[:, None] * inv[None]
        e = torch.cat([fr, fr], -1)
        cos, sin = e.cos(), e.sin()
        rot = lambda t: torch.cat([-t[..., HD // 2:], t[..., :HD // 2]], -1)
        mask = torch.full((T, T), float("-inf"), device="cuda").triu(1)
        for i in range(self.L):
            w = {k: v.cuda().float() for k, v in self.ck.layer(i).items()}
            h = self.rms(x, w["ln1"], eps)
            qkv = h @ w["qkv"].T
            q = qkv[:, :H * HD].view(T, H, HD)
            k = qkv[:, H * HD:(H + KV) * HD].view(T, KV, HD)
            v = qkv[:, (H + KV) * HD:].view(T, KV, HD)
            q, k = self.rms(q, w["qn"], eps), self.rms(k, w["kn"], eps)
            q = q * cos[:, None] + rot(q) * sin[:, None]
            k = k * cos[:, None] + rot(k) * sin[:, None]
            k = k.repeat_interleave(H // KV, 1)
            v = v.repeat_interleave(H // KV, 1)
            a = (q.transpose(0, 1) @ k.permute(1, 2, 0)) * HD ** -0.5 + mask  # [H, T, T]
            a = torch.softmax(a, -1)
            o = (a @ v.transpose(0, 1)).transpose(0, 1).reshape(T, H * HD)
            x = x + o @ w["o"].T
            h = self.rms(x, w["ln2"], eps)
            gu = h @ w["gu"].T
            F = c["intermediate_size"]
            x = x + (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w["down"].T
            del w
        h = self.rms(x, self.norm, eps)
        best = torch.full((T,), -float("inf"), device="cuda")
        arg = torch.zeros(T, dtype=torch.long, device="cuda")
        second = torch.full((T,), -float("inf"), device="cuda")
        for i in range(0, self.emb.shape[0], 16384):
            lg = h @ self.emb[i:i + 16384].cuda().float().T
            top = lg.topk(2, -1)
            for j in range(2):
                val, idx = top.values[:, j], top.indices[:, j] + i
                better = val > best
                second = torch.where(better, best, torch.maximum(second, val))
                arg = torch.where(better, idx, arg)
                best = torch.where(better, val, best)
        return arg.tolist(), (best - second).tolist()
