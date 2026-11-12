"""Чтение модели из папки Hugging Face (safetensors + config.json) по одному тензору.
Поддержано: семейство Qwen3 (плотные). Описание отличий архитектур — здесь (ARCH)."""
import glob
import json
import os

import torch
from safetensors import safe_open

ARCH = {
    # model_type: нормировка голов q/k, смещения в q/k/v
    "qwen3": {"qk_norm": True, "qkv_bias": False},
}


def find_model(name):
    """Путь к модели: папка, или имя в кэше Hugging Face (Qwen3-8B / Qwen/Qwen3-8B)."""
    if os.path.isdir(name):
        return name
    root = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    org, _, m = name.rpartition("/")
    hits = sorted(glob.glob(os.path.join(root, f"models--{org or 'Qwen'}--{m}", "snapshots", "*")))
    if not hits:
        raise FileNotFoundError(f"модель {name} не найдена (папка или кэш {root})")
    return hits[-1]


class Ckpt:
    def __init__(self, name):
        self.dir = find_model(name)
        self.cfg = json.load(open(os.path.join(self.dir, "config.json")))
        mt = self.cfg["model_type"]
        if mt not in ARCH:
            raise NotImplementedError(f"архитектура {mt} пока не поддержана (есть: {', '.join(ARCH)})")
        self.arch = ARCH[mt]
        self.files, self._paths = {}, {}
        for f in glob.glob(os.path.join(self.dir, "*.safetensors")):
            h = safe_open(f, framework="pt", device="cpu")
            self._paths[id(h)] = f
            for k in h.keys():
                self.files[k] = h
        c = self.cfg
        self.D, self.L, self.H, self.KV = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = c.get("head_dim", self.D // self.H)
        self.FF, self.eps, self.theta = c["intermediate_size"], c["rms_norm_eps"], c.get("rope_theta", 10000.0)
        self.tied = c.get("tie_word_embeddings", False)
        eos = c.get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        gp = os.path.join(self.dir, "generation_config.json")
        if os.path.exists(gp):
            g = json.load(open(gp)).get("eos_token_id", [])
            self.eos |= set(g if isinstance(g, list) else [g])

    def t(self, k):
        return self.files[k].get_tensor(k)

    def mm(self, k):
        """Тензор bf16 прямо из файла (np.memmap uint16, без копии в RAM: страницы читает система по мере нужды)."""
        import struct

        import numpy as np
        f = self.files[k]
        path = self._paths.get(id(f))
        with open(path, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            meta = json.loads(h.read(n))[k]
        assert meta["dtype"] == "BF16", meta["dtype"]
        a, b = meta["data_offsets"]
        return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=tuple(meta["shape"]))

    def head_name(self):
        return "model.embed_tokens.weight" if self.tied or "lm_head.weight" not in self.files else "lm_head.weight"

    def layer_mm(self, i):
        """Матрицы слоя как memmap (для слоёв на процессоре): отдельно q, k, v, gate, up."""
        p = f"model.layers.{i}."
        a, m = p + "self_attn.", p + "mlp."
        return {"q": self.mm(a + "q_proj.weight"), "k": self.mm(a + "k_proj.weight"), "v": self.mm(a + "v_proj.weight"),
                "o": self.mm(a + "o_proj.weight"), "gate": self.mm(m + "gate_proj.weight"), "up": self.mm(m + "up_proj.weight"),
                "down": self.mm(m + "down_proj.weight")}

    def layer(self, i):
        p = f"model.layers.{i}."
        a, m = p + "self_attn.", p + "mlp."
        w = {"qkv": torch.cat([self.t(a + "q_proj.weight"), self.t(a + "k_proj.weight"), self.t(a + "v_proj.weight")]),
             "o": self.t(a + "o_proj.weight"),
             "gu": torch.cat([self.t(m + "gate_proj.weight"), self.t(m + "up_proj.weight")]),
             "down": self.t(m + "down_proj.weight"),
             "ln1": self.t(p + "input_layernorm.weight"), "ln2": self.t(p + "post_attention_layernorm.weight")}
        if self.arch["qk_norm"]:
            w["qn"], w["kn"] = self.t(a + "q_norm.weight"), self.t(a + "k_norm.weight")
        return w

    def layer_params(self):
        D, H, KV, HD, FF = self.D, self.H, self.KV, self.HD, self.FF
        return {"qkv": ((H + 2 * KV) * HD, D), "o": (D, H * HD), "gu": (2 * FF, D), "down": (D, FF)}
