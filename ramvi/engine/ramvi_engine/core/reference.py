"""Честный эталон: обычный torch fp32 (математика HF Qwen3), веса слой за слоем в видеокарту.
Проверяет готовый текст одним проходом: самое вероятное слово в каждой позиции запрос+текст."""
import numpy as np
import torch

from ramvi_engine.formats.model import Ckpt


class Reference:
    def __init__(self, name):
        self.ck = Ckpt(name)
        self.emb = self.ck.mm("model.embed_tokens.weight")
        self.head = self.ck.mm(self.ck.head_name())
        self.norm = self.ck.t("model.norm.weight").cuda().float()

    @staticmethod
    def rms(x, w, eps):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    @staticmethod
    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)

    @torch.no_grad()
    def argmax(self, tokens):
        ck = self.ck
        H, KV, HD, eps, F = ck.H, ck.KV, ck.HD, ck.eps, ck.FF
        T = len(tokens)
        x = self._t(self.emb[np.asarray(tokens)]).cuda().float()
        inv = 1.0 / (ck.theta ** (torch.arange(0, HD, 2, dtype=torch.int64).float().cuda() / HD))
        fr = torch.arange(T, device="cuda").float()[:, None] * inv[None]
        e = torch.cat([fr, fr], -1)
        cos, sin = e.cos(), e.sin()
        rot = lambda t: torch.cat([-t[..., HD // 2:], t[..., :HD // 2]], -1)
        mask = torch.full((T, T), float("-inf"), device="cuda").triu(1)
        for i in range(ck.L):
            w = {k: v.cuda().float() for k, v in ck.layer(i).items()}
            h = self.rms(x, w["ln1"], eps)
            qkv = h @ w["qkv"].T
            q = self.rms(qkv[:, :H * HD].view(T, H, HD), w["qn"], eps)
            k = self.rms(qkv[:, H * HD:(H + KV) * HD].view(T, KV, HD), w["kn"], eps)
            v = qkv[:, (H + KV) * HD:].view(T, KV, HD)
            q = q * cos[:, None] + rot(q) * sin[:, None]
            k = k * cos[:, None] + rot(k) * sin[:, None]
            k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            a = torch.softmax((q.transpose(0, 1) @ k.permute(1, 2, 0)) * HD ** -0.5 + mask, -1)
            x = x + (a @ v.transpose(0, 1)).transpose(0, 1).reshape(T, H * HD) @ w["o"].T
            h = self.rms(x, w["ln2"], eps)
            gu = h @ w["gu"].T
            x = x + (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ w["down"].T
            del w
        h = self.rms(x, self.norm, eps)
        best = torch.full((T,), -float("inf"), device="cuda")
        arg = torch.zeros(T, dtype=torch.long, device="cuda")
        for i in range(0, self.head.shape[0], 16384):
            lg = h @ self._t(self.head[i:i + 16384]).cuda().float().T
            v, j = lg.max(-1)
            better = v > best
            arg = torch.where(better, j + i, arg)
            best = torch.where(better, v, best)
        return arg.tolist()
