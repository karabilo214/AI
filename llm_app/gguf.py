"""Чтение моделей GGUF (формат llama.cpp / Ollama): заголовок, метаданные, тензоры как memmap без копии.
Распаковка на процессоре (numpy) для Q4_K, Q6_K, Q8_0, F16, BF16, F32 — для проверки ядер и строк словаря.
find_ollama(name) — путь к файлу модели Ollama по имени вида "qwen3:14b"."""
import json
import os
import struct

import numpy as np

QK_K = 256
# тип: (имя, байт на блок, чисел в блоке)
TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 8: ("Q8_0", 34, 32), 12: ("Q4_K", 144, 256), 14: ("Q6_K", 210, 256),
         30: ("BF16", 2, 1)}


def find_ollama(name):
    """'qwen3:14b' -> путь к blob с весами в ~/.ollama/models."""
    root = os.environ.get("OLLAMA_MODELS", os.path.join(os.path.expanduser("~"), ".ollama", "models"))
    model, _, tag = name.partition(":")
    ns = "library" if "/" not in model else model.split("/")[0]
    m = os.path.join(root, "manifests", "registry.ollama.ai", ns, model.split("/")[-1], tag or "latest")
    man = json.load(open(m))
    dg = next(l["digest"] for l in man["layers"] if l["mediaType"] == "application/vnd.ollama.image.model")
    return os.path.join(root, "blobs", dg.replace(":", "-"))


class GGUF:
    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        assert f.read(4) == b"GGUF", "не GGUF"
        ver, = struct.unpack("<I", f.read(4))
        nt, nkv = struct.unpack("<QQ", f.read(16))
        self.meta = {}
        for _ in range(nkv):
            k = self._str(f)
            self.meta[k] = self._val(f, struct.unpack("<I", f.read(4))[0])
        self.tensors = {}
        for _ in range(nt):
            name = self._str(f)
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            self.tensors[name] = (dims, typ, off)
        al = self.meta.get("general.alignment", 32)
        self.data = (f.tell() + al - 1) // al * al
        f.close()
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    @staticmethod
    def _str(f):
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def _val(self, f, t):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            return struct.unpack("<" + fmt[t], f.read(struct.calcsize(fmt[t])))[0]
        if t == 8:
            return self._str(f)
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            if et in fmt and n > 1000:  # большие числовые массивы — пропускаем содержимое
                f.seek(n * struct.calcsize(fmt[et]), 1)
                return ("array", et, n)
            return [self._val(f, et) for _ in range(n)]
        raise ValueError(f"тип метаданных {t}")

    def info(self, name):
        dims, typ, off = self.tensors[name]
        return dims, TYPES[typ][0]

    def raw(self, name):
        """Байты тензора (memmap) и (N строк, K в строке, тип). В GGUF dims[0] — длина строки."""
        dims, typ, off = self.tensors[name]
        tn, bs, be = TYPES[typ]
        K = dims[0]
        N = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        nbytes = N * (K // be) * bs
        a = self.data + off
        return self.mm[a:a + nbytes], N, K, tn

    def rows(self, name, idx):
        """Распаковать строки idx (numpy float32 [len(idx), K])."""
        buf, N, K, tn = self.raw(name)
        bs, be = TYPES[self.tensors[name][1]][1:]
        rb = K // be * bs
        out = np.empty((len(idx), K), np.float32)
        for i, r in enumerate(idx):
            out[i] = dequant(np.asarray(buf[r * rb:(r + 1) * rb]), tn, K)
        return out

    def full(self, name):
        buf, N, K, tn = self.raw(name)
        return dequant(np.asarray(buf), tn, N * K).reshape(N, K) if N > 1 else dequant(np.asarray(buf), tn, K)


def _scale_min_k4(sc):
    """sc [n, 12] uint8 -> (scale [n, 8], min [n, 8]) 6-битные."""
    sc = sc.astype(np.int32)
    d = np.empty((sc.shape[0], 8), np.int32)
    m = np.empty_like(d)
    d[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    d[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return d, m


def dequant(b, tn, n):
    if tn == "F32":
        return b.view(np.float32)[:n].copy()
    if tn == "F16":
        return b.view(np.float16)[:n].astype(np.float32)
    if tn == "BF16":
        return (b.view(np.uint16)[:n].astype(np.uint32) << 16).view(np.float32)
    if tn == "Q8_0":
        blk = b.reshape(-1, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        return (d * blk[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)[:n]
    if tn == "Q4_K":
        blk = b.reshape(-1, 144)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)
        dm = blk[:, 2:4].copy().view(np.float16).astype(np.float32)
        sc, mn = _scale_min_k4(blk[:, 4:16])
        qs = blk[:, 16:].reshape(-1, 4, 32)
        q = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)  # подблоки по 32
        y = d[:, :, None] * sc[:, :, None] * q - dm[:, :, None] * mn[:, :, None]
        return y.reshape(-1)[:n]
    if tn == "Q6_K":
        blk = b.reshape(-1, 210)
        ql, qh = blk[:, :128].astype(np.int32), blk[:, 128:192].astype(np.int32)
        sc = blk[:, 192:208].view(np.int8).astype(np.float32)
        d = blk[:, 208:210].copy().view(np.float16).astype(np.float32)
        y = np.empty((blk.shape[0], 256), np.float32)
        for h in range(2):
            L, Hh, S = ql[:, h * 64:(h + 1) * 64], qh[:, h * 32:(h + 1) * 32], sc[:, h * 8:(h + 1) * 8]
            q1 = (L[:, :32] & 0xF) | (((Hh >> 0) & 3) << 4)
            q2 = (L[:, 32:] & 0xF) | (((Hh >> 2) & 3) << 4)
            q3 = (L[:, :32] >> 4) | (((Hh >> 4) & 3) << 4)
            q4 = (L[:, 32:] >> 4) | (((Hh >> 6) & 3) << 4)
            for k, q in enumerate((q1, q2, q3, q4)):
                s = np.repeat(S[:, [2 * k, 2 * k + 1]], 16, axis=1)  # первые 16 чисел — масштаб is, следующие — is+1
                y[:, h * 128 + k * 32:h * 128 + (k + 1) * 32] = d[:, :1] * s * (q - 32)
        return y.reshape(-1)[:n]
    raise NotImplementedError(tn)
