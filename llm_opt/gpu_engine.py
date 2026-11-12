"""Движок Qwen3-0.6B для видеокарты (CUDA) с записанными графами: проход T слов — одна команда.

Кэш k, v фиксированного размера [L, KV, MAXT, HD]; позиция — тензор, поэтому граф не зависит от длины текста.
Внимание считается по всем MAXT позициям с маской «видны только 0..pos+t» (лишняя работа мала).
Для каждого T (1, 2, 4, 8, 16) свой граф. Выход: самое вероятное слово после каждой из T позиций.
"""
import torch
import torch.nn.functional as F

from core import EPS, HD, H, KV, L, Qwen

try:
    from triton_kernels import mm16
except ImportError:
    mm16 = None

G = H // KV


class GpuEngine:
    """mode: "fp32" — всё fp32; "bf16" — всё bf16; "w16" — веса bf16, счёт fp32: вход матрицы раскладывается
    на 3 части bf16 (старшие/средние/младшие 8 бит мантиссы), умножение bf16 x bf16 с накоплением в fp32,
    три части идут строками одного умножения (веса читаются один раз); "w16s2" — то же с 2 частями."""

    def __init__(self, dtype=torch.bfloat16, maxt=1024, Ts=(1, 2, 4, 8, 16), mode=None):
        mode = mode or {torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
        self.mode = mode
        self.split = {"w16": 3, "w16s2": 2}.get(mode, 0)
        self.tri = mode == "tri"  # веса bf16 [N, K], своё ядро Triton со счётом в fp32
        wdt = torch.float32 if mode == "fp32" else torch.bfloat16
        dtype = torch.bfloat16 if mode == "bf16" else torch.float32  # тип счёта
        self.m = Qwen("cuda", wdt)
        self.dtype, self.maxt = dtype, maxt
        # склеенные матрицы: q|k|v и gate|up — меньше умножений и меньше раскладок входа
        for w in self.m.layers:
            w["qkv"] = torch.cat([w["q"], w["k"], w["v"]], 1).contiguous()
            w["gu"] = torch.cat([w["gate"], w["up"]], 1).contiguous()
            for n in ("q", "k", "v", "gate", "up"):
                del w[n]
        if self.tri:
            for w in self.m.layers:
                for n in ("qkv", "o", "gu", "down"):
                    w[n] = w[n].T.contiguous()  # [N, K]
            self.m.head = self.m.emb  # [V, K] — уже нужная раскладка
        torch.cuda.empty_cache()
        self.K = torch.zeros(L, KV, maxt, HD, device="cuda", dtype=dtype)
        self.V = torch.zeros_like(self.K)
        self.graphs, self.io = {}, {}
        for T in Ts:
            self._capture(T)

    def mm(self, x, W):
        """x [T, in] в типе счёта, W [in, out] (для tri — [out, in])."""
        if self.tri:
            return mm16(x, W)
        if not self.split:
            return x @ W
        T = x.shape[0]
        parts, r = [], x
        for _ in range(self.split):
            p = r.to(torch.bfloat16)
            parts.append(p)
            r = r - p.float()
        Y = torch.mm(torch.cat(parts), W, out_dtype=torch.float32)
        out = Y[:T]
        for i in range(1, self.split):
            out = out + Y[i * T:(i + 1) * T]
        return out

    def _rms(self, x, w):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
        return (w * x32.to(x.dtype)).to(self.dtype)

    def _rot(self, x):
        return torch.cat((-x[..., HD // 2:], x[..., :HD // 2]), -1)

    def _forward(self, tok, pos0):
        """tok [T] long, pos0 [1] long — всё на GPU, без синхронизаций (годится для графа)."""
        m = self.m
        T = tok.shape[0]
        pos = pos0 + torch.arange(T, device="cuda")
        fr = pos[:, None].float() * m.inv_freq[None]
        e = torch.cat((fr, fr), -1)
        cos, sin = e.cos().to(self.dtype), e.sin().to(self.dtype)
        visible = torch.arange(self.maxt, device="cuda")[None, :] <= pos[:, None]  # [T, MAXT]
        bias = torch.zeros(T, self.maxt, device="cuda", dtype=torch.float32).masked_fill(~visible, float("-inf"))
        x = m.emb[tok].to(self.dtype)
        for l, w in enumerate(m.layers):
            h = self._rms(x, w["ln1"])
            qkv = self.mm(h, w["qkv"])
            q = self._rms(qkv[:, :H * HD].reshape(T, H, HD), w["qn"])
            k = self._rms(qkv[:, H * HD:(H + KV) * HD].reshape(T, KV, HD), w["kn"])
            v = qkv[:, (H + KV) * HD:].reshape(T, KV, HD)
            q = q * cos[:, None] + self._rot(q) * sin[:, None]
            k = k * cos[:, None] + self._rot(k) * sin[:, None]
            self.K[l].index_copy_(1, pos, k.transpose(0, 1))
            self.V[l].index_copy_(1, pos, v.transpose(0, 1))
            qg = q.view(T, KV, G, HD).permute(1, 2, 0, 3).reshape(KV, G * T, HD)  # головы группы вместе
            s = (qg @ self.K[l].transpose(1, 2)).float() * HD ** -0.5  # [KV, G*T, MAXT]
            s = s.view(KV, G, T, self.maxt) + bias
            a = F.softmax(s, dim=-1).to(self.dtype).view(KV, G * T, self.maxt)
            o = (a @ self.V[l]).view(KV, G, T, HD).permute(2, 0, 1, 3).reshape(T, H * HD)
            x = x + self.mm(o, w["o"])
            h = self._rms(x, w["ln2"])
            gu = self.mm(h, w["gu"])
            x = x + self.mm(F.silu(gu[:, :3072]) * gu[:, 3072:], w["down"])
        return self.mm(self._rms(x, m.norm), m.head).argmax(-1)

    def _capture(self, T):
        tok = torch.zeros(T, dtype=torch.long, device="cuda")
        pos0 = torch.zeros(1, dtype=torch.long, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self._forward(tok, pos0)  # прогрев
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self._forward(tok, pos0)
        self.graphs[T], self.io[T] = g, (tok, pos0, out)
        self.K.zero_(); self.V.zero_()

    @torch.no_grad()
    def run(self, tokens, pos0):
        """Прогоняет слова tokens (позиции pos0..), возвращает тензор самых вероятных следующих слов (на GPU)."""
        T = len(tokens)
        if T in self.graphs:
            tok, p, out = self.io[T]
            tok.copy_(torch.as_tensor(tokens, device="cuda"))
            p.fill_(pos0)
            self.graphs[T].replay()
            return out
        return self._forward(torch.as_tensor(tokens, device="cuda"), torch.tensor([pos0], device="cuda"))

    @torch.no_grad()
    def prefill(self, tokens):
        """Текст запроса целиком без графа (длина любая)."""
        return self._forward(torch.as_tensor(tokens, device="cuda"), torch.tensor([0], device="cuda"))
