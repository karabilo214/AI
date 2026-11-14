"""Независимая математика torch fp32 по исходным HF/GGUF весам, слой за слоем.

Не использует native kernels, repack, CpuMat или Engine.process. Quant decoder
в этом файле независим от formats.gguf.dequant. Это диагностический, не быстрый путь.
"""

from importlib import import_module
from typing import Any

from ramvi_engine.model_location import is_gguf_model, resolve_gguf_name

np = import_module("numpy")
torch = import_module("torch")


def decode_quant(raw: Any, kind: str, count: int) -> Any:
    """GGUF blocks -> float32, независимо от inference decoder."""
    data = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if kind == "F32":
        return data.view("<f4")[:count].copy()
    if kind == "F16":
        return data.view("<f2")[:count].astype(np.float32)
    if kind == "BF16":
        return (data.view("<u2")[:count].astype(np.uint32) << 16).view(np.float32)
    if kind == "Q8_0":
        blocks = data.reshape(-1, 34)
        scale = blocks[:, :2].copy().view("<f2").astype(np.float32)
        return (blocks[:, 2:].view(np.int8).astype(np.float32) * scale).reshape(-1)[:count]
    if kind == "Q4_K":
        blocks = data.reshape(-1, 144)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        minimum = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        encoded = blocks[:, 4:16].astype(np.int32)
        scales, offsets = [], []
        for sub in range(8):
            if sub < 4:
                scales.append(encoded[:, sub] & 63)
                offsets.append(encoded[:, sub + 4] & 63)
            else:
                scales.append((encoded[:, sub + 4] & 15) | ((encoded[:, sub - 4] >> 6) << 4))
                offsets.append((encoded[:, sub + 4] >> 4) | ((encoded[:, sub] >> 6) << 4))
        packed = blocks[:, 16:].reshape(-1, 4, 32)
        quant = np.stack([packed & 15, packed >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
        result = d[:, :, None] * np.stack(scales, axis=1)[:, :, None] * quant
        result -= minimum[:, :, None] * np.stack(offsets, axis=1)[:, :, None]
        return result.reshape(-1)[:count].astype(np.float32)
    if kind == "Q6_K":
        blocks = data.reshape(-1, 210)
        position = np.arange(256)
        half, local = position // 128, position % 128
        quarter, within = local // 32, local % 32
        low_bytes = blocks[:, half * 64 + (quarter % 2) * 32 + within].astype(np.int32)
        low = np.where(quarter < 2, low_bytes & 15, low_bytes >> 4)
        high = (blocks[:, 128 + half * 32 + within].astype(np.int32) >> (quarter * 2)) & 3
        quant = (low | (high << 4)) - 32
        scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
        selected_scales = scales[:, half * 8 + quarter * 2 + within // 16]
        d = blocks[:, 208:210].copy().view("<f2").astype(np.float32)
        return (d * selected_scales * quant).reshape(-1)[:count].astype(np.float32)
    raise NotImplementedError(f"oracle quantization {kind}")


class TorchReference:
    """Полный causal forward без KV-cache; храним на GPU только текущие матрицы."""

    def __init__(self, name: str, *, device: str = "cuda", native_kv: bool = True) -> None:
        self.device, self.native_kv = device, native_kv
        self.gguf = is_gguf_model(name)
        self.g: Any = None
        self.ck: Any = None
        if self.gguf:
            from ramvi_engine.formats.gguf import GGUF, find_ollama

            self.g = GGUF(resolve_gguf_name(name, find_ollama))  # type: ignore[no-untyped-call]
            self.arch = self.g.meta["general.architecture"]
            if self.arch not in ("qwen3", "qwen3moe"):
                raise NotImplementedError(self.arch)

            def meta(key: str) -> Any:
                return self.g.meta[f"{self.arch}.{key}"]

            self.L, self.D, self.H, self.KV, self.HD = (
                int(meta(key))
                for key in (
                    "block_count",
                    "embedding_length",
                    "attention.head_count",
                    "attention.head_count_kv",
                    "attention.key_length",
                )
            )
            self.eps, self.theta = (
                float(meta("attention.layer_norm_rms_epsilon")),
                float(meta("rope.freq_base")),
            )
            self.F = int(
                meta(
                    "expert_feed_forward_length"
                    if self.arch == "qwen3moe"
                    else "feed_forward_length"
                )
            )
            self.E = int(meta("expert_count")) if self.arch == "qwen3moe" else 0
            self.K = int(meta("expert_used_count")) if self.arch == "qwen3moe" else 0
            self.head_name = (
                "output.weight" if "output.weight" in self.g.tensors else "token_embd.weight"
            )
            self.V = self.g.raw(self.head_name)[1]
        else:
            from ramvi_engine.formats.model import Ckpt

            self.ck = Ckpt(name)  # type: ignore[no-untyped-call]
            self.arch = self.ck.cfg["model_type"]
            self.L, self.D, self.H, self.KV, self.HD = (
                self.ck.L,
                self.ck.D,
                self.ck.H,
                self.ck.KV,
                self.ck.HD,
            )
            self.eps, self.theta, self.F = self.ck.eps, self.ck.theta, self.ck.FF
            self.E = self.K = 0
            self.V = self.ck.cfg["vocab_size"]
            self.head_name = self.ck.head_name()

    def weight(self, name: str, begin: int = 0, end: int | None = None) -> Any:
        if not self.gguf:
            tensor = self.ck.t(name)
            if tensor.ndim > 1:
                tensor = tensor[begin:end]
            return tensor.to(self.device, dtype=torch.float32)
        buf, n, k, kind = self.g.raw(name)
        end = n if end is None else end
        block_size, block_elements = {
            "F32": (4, 1),
            "F16": (2, 1),
            "BF16": (2, 1),
            "Q8_0": (34, 32),
            "Q4_K": (144, 256),
            "Q6_K": (210, 256),
        }[kind]
        row_bytes = k // block_elements * block_size
        values = decode_quant(buf[begin * row_bytes : end * row_bytes], kind, (end - begin) * k)
        tensor = torch.from_numpy(values.reshape(end - begin, k)).to(self.device)
        return tensor

    def linear(self, x: Any, name: str, *, chunk: int = 1024) -> Any:
        """Project independent rows without decoding a whole large GGUF matrix."""
        if not self.gguf:
            return x @ self.weight(name).T
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        _, rows, _, _ = self.g.raw(name)
        return torch.cat(
            [
                x @ self.weight(name, start, min(rows, start + chunk)).T
                for start in range(0, rows, chunk)
            ],
            dim=-1,
        )

    def norm_weight(self, name: str) -> Any:
        return self.weight(name).reshape(-1)

    @staticmethod
    def rms(x: Any, weights: Any, eps: float) -> Any:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weights

    def _names(self, layer: int) -> dict[str, str]:
        if self.gguf:
            p = f"blk.{layer}."
            return {
                name: p + value + ".weight"
                for name, value in {
                    "q": "attn_q",
                    "k": "attn_k",
                    "v": "attn_v",
                    "o": "attn_output",
                    "ln1": "attn_norm",
                    "ln2": "ffn_norm",
                    "qn": "attn_q_norm",
                    "kn": "attn_k_norm",
                    "gate": "ffn_gate",
                    "up": "ffn_up",
                    "down": "ffn_down",
                }.items()
            }
        p = f"model.layers.{layer}."
        names = {
            name: p + "self_attn." + value + ".weight"
            for name, value in {
                "q": "q_proj",
                "k": "k_proj",
                "v": "v_proj",
                "o": "o_proj",
                "qn": "q_norm",
                "kn": "k_norm",
            }.items()
        }
        names.update(
            {
                name: p + "mlp." + value + ".weight"
                for name, value in {
                    "gate": "gate_proj",
                    "up": "up_proj",
                    "down": "down_proj",
                }.items()
            }
        )
        names.update(
            {"ln1": p + "input_layernorm.weight", "ln2": p + "post_attention_layernorm.weight"}
        )
        return names

    def _moe(self, h: Any, layer: int) -> Any:
        prefix = f"blk.{layer}."
        probabilities = torch.softmax(h @ self.weight(prefix + "ffn_gate_inp.weight").T, dim=-1)
        ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, : self.K]
        weights = probabilities.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(h)
        for expert in torch.unique(ids).tolist():
            tokens, slots = torch.where(ids == expert)
            selected = h[tokens]
            gate = (
                selected
                @ self.weight(
                    prefix + "ffn_gate_exps.weight", expert * self.F, (expert + 1) * self.F
                ).T
            )
            up = (
                selected
                @ self.weight(
                    prefix + "ffn_up_exps.weight", expert * self.F, (expert + 1) * self.F
                ).T
            )
            down = self.weight(
                prefix + "ffn_down_exps.weight", expert * self.D, (expert + 1) * self.D
            )
            values = (torch.nn.functional.silu(gate) * up) @ down.T
            output.index_add_(0, tokens, values * weights[tokens, slots, None])
        return output

    def hidden(self, tokens: list[int], *, trace: bool = False) -> tuple[Any, list[Any]]:
        if not tokens or any(
            isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.V for t in tokens
        ):
            raise ValueError("неверные token IDs oracle")
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            embedding = "token_embd.weight" if self.gguf else "model.embed_tokens.weight"
            x = torch.cat([self.weight(embedding, token, token + 1) for token in tokens])
            length = len(tokens)
            inv = 1.0 / self.theta ** (
                torch.arange(0, self.HD, 2, device=self.device).float() / self.HD
            )
            angles = torch.arange(length, device=self.device).float()[:, None] * inv[None]
            cosine, sine = (
                torch.cat([angles, angles], -1).cos()[:, None],
                torch.cat([angles, angles], -1).sin()[:, None],
            )
            mask = torch.full((length, length), -torch.inf, device=self.device).triu(1)

            def rotate(value: Any) -> Any:
                return torch.cat([-value[..., self.HD // 2 :], value[..., : self.HD // 2]], -1)

            history = []
            for layer in range(self.L):
                n = self._names(layer)
                h = self.rms(x, self.norm_weight(n["ln1"]), self.eps)
                q = self.rms(
                    self.linear(h, n["q"]).reshape(length, self.H, self.HD),
                    self.norm_weight(n["qn"]),
                    self.eps,
                )
                k = self.rms(
                    self.linear(h, n["k"]).reshape(length, self.KV, self.HD),
                    self.norm_weight(n["kn"]),
                    self.eps,
                )
                v = self.linear(h, n["v"]).reshape(length, self.KV, self.HD)
                q, k = q * cosine + rotate(q) * sine, k * cosine + rotate(k) * sine
                if self.gguf and self.native_kv:
                    k, v = k.half().float(), v.half().float()
                k, v = (
                    k.repeat_interleave(self.H // self.KV, dim=1),
                    v.repeat_interleave(self.H // self.KV, dim=1),
                )
                attention = torch.softmax(
                    (q.transpose(0, 1) @ k.permute(1, 2, 0)) * self.HD**-0.5 + mask, dim=-1
                )
                values = (
                    (attention @ v.transpose(0, 1))
                    .transpose(0, 1)
                    .reshape(length, self.H * self.HD)
                )
                x = x + self.linear(values, n["o"])
                h = self.rms(x, self.norm_weight(n["ln2"]), self.eps)
                if self.arch == "qwen3moe":
                    x = x + self._moe(h, layer)
                else:
                    gate, up = self.linear(h, n["gate"]), self.linear(h, n["up"])
                    x = x + self.linear(torch.nn.functional.silu(gate) * up, n["down"])
                if trace:
                    history.append(x[-1].detach().cpu().clone())
            norm = "output_norm.weight" if self.gguf else "model.norm.weight"
            return self.rms(x, self.norm_weight(norm), self.eps), history
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def project(self, hidden: Any, chunk: int = 1024) -> Any:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return torch.cat(
                [
                    hidden @ self.weight(self.head_name, start, min(self.V, start + chunk)).T
                    for start in range(0, self.V, chunk)
                ],
                dim=-1,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def logits(self, tokens: list[int], positions: list[int] | None = None) -> Any:
        hidden, _ = self.hidden(tokens)
        return self.project(hidden[positions or [-1]])

    def close(self) -> None:
        if self.g is not None:
            self.g.close()
