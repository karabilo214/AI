"""Своя реализация Qwen3-0.6B на чистом torch: всё видно и во всё можно вмешаться.

Веса берутся из оригинальной модели. check_core.py проверяет совпадение с transformers.

Вмешательства (opts, все необязательны):
  skip       — множество ("attn", l) / ("mlp", l): часть слоя не прибавляется
  head_mult  — {l: тензор[16]}: множитель выхода каждой головы внимания
  mlp_mask   — функция (l, h) -> h': правка внутренних нейронов MLP (h = silu(gate)*up)
  rec        — функция (имя, l, тензор): запись промежуточных величин
  timer      — объект с .start(имя) / .stop(имя): замер времени частей
"""
import glob
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))
              + glob.glob(os.path.join(os.environ.get("HF_HUB_CACHE", "/nonexistent"), "models--Qwen--Qwen3-0.6B/snapshots/*")))[0]

L, D, H, KV, HD, FF = 28, 1024, 16, 8, 128, 3072
EPS, THETA = 1e-6, 1_000_000.0


def rms(x, w):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return w * x32.to(x.dtype)


def rot_half(x):
    a, b = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat((-b, a), dim=-1)


class Qwen:
    def __init__(self, device="cpu", dtype=torch.float32):
        sd = {}
        for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
            sd.update(load_file(f))
        g = lambda k: sd[k].to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.emb = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.head = self.emb.T.contiguous()  # tie_word_embeddings; хранится транспонированной
        self.layers = []
        for l in range(L):
            p = f"model.layers.{l}."
            self.layers.append({k: g(p + v) for k, v in dict(
                ln1="input_layernorm.weight", ln2="post_attention_layernorm.weight",
                q="self_attn.q_proj.weight", k="self_attn.k_proj.weight",
                v="self_attn.v_proj.weight", o="self_attn.o_proj.weight",
                qn="self_attn.q_norm.weight", kn="self_attn.k_norm.weight",
                gate="mlp.gate_proj.weight", up="mlp.up_proj.weight",
                down="mlp.down_proj.weight").items()})
            for n in ["q", "k", "v", "o", "gate", "up", "down"]:
                # храним W^T: на этом macOS MPS падает на умножении на транспонированный вид
                self.layers[-1][n] = self.layers[-1][n].T.contiguous()
        self.inv_freq = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32, device=device) / HD))

    def rope(self, pos):
        fr = pos[:, None].float() * self.inv_freq[None]
        e = torch.cat((fr, fr), -1)
        return e.cos().to(self.dtype), e.sin().to(self.dtype)

    @torch.no_grad()
    def forward(self, ids, opts=None, cache=None, return_hidden=False):
        """ids: [B, T]. cache: список (k, v) по слоям или None. Возвращает логиты [B, T, V]."""
        o = opts or {}
        skip, head_mult = o.get("skip", ()), o.get("head_mult", {})
        mlp_mask, rec, tm = o.get("mlp_mask"), o.get("rec"), o.get("timer")
        im = o.get("in_mask") or (lambda name, l, x: x)  # правка входа матриц до умножения
        heads_fn = o.get("heads_fn")  # (l, a, out[B,H,T,HD], v[B,KV,S,HD]) -> out: правка выхода голов по токенам
        delta_fn = o.get("delta_fn")  # (вид, l, x, прибавка) -> прибавка: правка вклада части слоя по токенам
        st = (lambda n: tm.start(n)) if tm else (lambda n: None)
        sp = (lambda n: tm.stop(n)) if tm else (lambda n: None)

        B, T = ids.shape
        past = 0 if not cache or cache[0] is None else cache[0][0].shape[2]
        st("embed"); x = self.emb[ids]; sp("embed")
        pos = torch.arange(past, past + T, device=self.device)
        cos, sin = self.rope(pos)
        mask = None
        if T > 1:
            mask = torch.full((T, past + T), float("-inf"), device=self.device, dtype=self.dtype)
            mask = torch.triu(mask, diagonal=past + 1)
        hidden = [x] if return_hidden else None

        for l, w in enumerate(self.layers):
            # ---- внимание ----
            st("attn_proj")
            h = rms(x, w["ln1"])
            if rec:
                rec("in_qkv", l, h)
            h = im("in_qkv", l, h)
            q = (h @ w["q"]).view(B, T, H, HD)
            k = (h @ w["k"]).view(B, T, KV, HD)
            v = (h @ w["v"]).view(B, T, KV, HD).transpose(1, 2)
            q = rms(q, w["qn"]).transpose(1, 2)
            k = rms(k, w["kn"]).transpose(1, 2)
            q = q * cos + rot_half(q) * sin
            k = k * cos + rot_half(k) * sin
            sp("attn_proj")
            if cache is not None:
                if cache[l] is not None:
                    k = torch.cat((cache[l][0], k), 2)
                    v = torch.cat((cache[l][1], v), 2)
                cache[l] = (k, v)
            st("attn_core")
            kk = k.repeat_interleave(H // KV, dim=1)
            vv = v.repeat_interleave(H // KV, dim=1)
            s = (q.contiguous() @ kk.transpose(2, 3).contiguous()) * HD ** -0.5
            if mask is not None:
                s = s + mask
            a = F.softmax(s, dim=-1, dtype=torch.float32).to(self.dtype)
            if rec:
                rec("attn_w", l, a)
            out = a @ vv.contiguous()  # [B, H, T, HD]
            if heads_fn is not None:
                out = heads_fn(l, a, out, v)
            if l in head_mult:
                out = out * head_mult[l].view(1, H, 1, 1)
            sp("attn_core")
            st("attn_proj")
            out = out.transpose(1, 2).reshape(B, T, H * HD)
            if rec:
                rec("in_o", l, out)
            out = im("in_o", l, out)
            out = out @ w["o"]
            sp("attn_proj")
            if rec:
                rec("attn_out", l, out)
            if delta_fn is not None:
                out = delta_fn("attn", l, x, out)
            if ("attn", l) not in skip:
                x = x + out

            # ---- MLP ----
            st("mlp")
            h = rms(x, w["ln2"])
            if rec:
                rec("in_mlp", l, h)
            h = im("in_mlp", l, h)
            m = F.silu(h @ w["gate"]) * (h @ w["up"])
            if rec:
                rec("mlp_h", l, m)
            if mlp_mask is not None:
                m = mlp_mask(l, m)
            m = im("mlp_h", l, m)
            out = m @ w["down"]
            sp("mlp")
            if rec:
                rec("mlp_out", l, out)
            if delta_fn is not None:
                out = delta_fn("mlp", l, x, out)
            if ("mlp", l) not in skip:
                x = x + out
            if return_hidden:
                hidden.append(x)

        st("head")
        if o.get("head_fn") is not None:  # своя замена словаря: получает конечные состояния
            logits = o["head_fn"](rms(x, self.norm))
            sp("head")
            return logits
        logits = rms(x, self.norm) @ self.head
        sp("head")
        return (logits, hidden) if return_hidden else logits

    def logits_from_hidden(self, x):
        return rms(x, self.norm) @ self.head


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAP)


def wiki_tokens(split="test"):
    """Весь текст wikitext-2 одной строкой токенов (кэшируется в data/)."""
    path = os.path.join(HERE, "data", f"wiki_{split}_ids.pt")
    if os.path.exists(path):
        return torch.load(path)
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(HERE, "data", "wikitext-2-raw-v1",
                                   f"{split}-00000-of-00001.parquet"))
    text = "".join(t.column("text").to_pylist())
    ids = torch.tensor(tokenizer()(text)["input_ids"], dtype=torch.long)
    torch.save(ids, path)
    return ids


def windows(ids, n, T=512, start=0):
    """n непересекающихся окон по T токенов: [n, T]."""
    return torch.stack([ids[start + i * T: start + (i + 1) * T] for i in range(n)])


def loss_of(logits, ids):
    """Средняя потеря предсказания следующего токена и доля верных (топ-1)."""
    lg = logits[:, :-1].float()
    y = ids[:, 1:]
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
    acc = (lg.argmax(-1) == y).float().mean()
    return loss.item(), acc.item()
