"""Чтение моделей GGUF (формат llama.cpp / Ollama): заголовок, метаданные, тензоры как memmap без копии.
Распаковка на процессоре (numpy) для Q4_K, Q6_K, Q8_0, F16, BF16, F32 — для проверки ядер и строк словаря.
find_ollama(name) — путь к файлу модели Ollama по имени вида "qwen3:14b"."""
import json
import math
import numbers
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
        self.size = os.path.getsize(path)
        with open(path, "rb") as f:
            if self._read(f, 4) != b"GGUF":
                raise ValueError("не GGUF")
            ver, = struct.unpack("<I", self._read(f, 4))
            if ver not in (2, 3):
                raise ValueError(f"неподдерживаемая версия GGUF: {ver}")
            nt, nkv = struct.unpack("<QQ", self._read(f, 16))
            if nt + nkv > self.size // 12:
                raise ValueError("неверное число записей GGUF")
            self.meta = {}
            for _ in range(nkv):
                key = self._str(f)
                if key in self.meta:
                    raise ValueError("повтор метаданных GGUF")
                self.meta[key] = self._val(f, struct.unpack("<I", self._read(f, 4))[0])
            self.tensors = {}
            for _ in range(nt):
                name = self._str(f)
                nd, = struct.unpack("<I", self._read(f, 4))
                if not 1 <= nd <= 4:
                    raise ValueError("неверная размерность тензора GGUF")
                dims = struct.unpack(f"<{nd}Q", self._read(f, 8 * nd))
                typ, off = struct.unpack("<IQ", self._read(f, 12))
                if typ not in TYPES:
                    raise NotImplementedError(f"тип тензора GGUF {typ}")
                if name in self.tensors or any(d <= 0 for d in dims):
                    raise ValueError("неверные имя или размеры тензора")
                if dims[0] % TYPES[typ][2]:
                    raise ValueError("строка не кратна блоку квантования")
                self.tensors[name] = (dims, typ, off)
            al = self.meta.get("general.alignment", 32)
            if not isinstance(al, int) or al <= 0 or al > 4096 or al & (al - 1):
                raise ValueError("неверное выравнивание GGUF")
            self.data = (f.tell() + al - 1) // al * al
        for dims, typ, off in self.tensors.values():
            byte_count = math.prod(dims) // TYPES[typ][2] * TYPES[typ][1]
            if off % al or self.data + off + byte_count > self.size:
                raise ValueError("тензор за границами GGUF или не выровнен")
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _read(self, f, n):
        if n < 0 or n > self.size - f.tell():
            raise ValueError("обрезанный GGUF")
        value = f.read(n)
        if len(value) != n:
            raise ValueError("обрезанный GGUF")
        return value

    def _str(self, f):
        n, = struct.unpack("<Q", self._read(f, 8))
        return self._read(f, n).decode("utf-8")

    def _val(self, f, t, depth=0):
        if depth > 32:
            raise ValueError("слишком глубокие метаданные GGUF")
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            return struct.unpack("<" + fmt[t], self._read(f, struct.calcsize(fmt[t])))[0]
        if t == 8:
            return self._str(f)
        if t == 9:
            et, n = struct.unpack("<IQ", self._read(f, 12))
            if n > self.size or et in fmt and n * struct.calcsize(fmt[et]) > self.size - f.tell():
                raise ValueError("неверный размер массива GGUF")
            if et in fmt and n > 1000:  # большие числовые массивы — пропускаем содержимое
                f.seek(n * struct.calcsize(fmt[et]), 1)
                return ("array", et, n)
            return [self._val(f, et, depth + 1) for _ in range(n)]
        raise ValueError(f"тип метаданных {t}")

    def close(self):
        """После close все ранее выданные представления memmap использовать нельзя."""
        if self.mm is not None:
            self.mm._mmap.close()
            self.mm = None

    def info(self, name):
        dims, typ, off = self.tensors[name]
        return dims, TYPES[typ][0]

    def raw(self, name):
        """Байты тензора (memmap) и (N строк, K в строке, тип). В GGUF dims[0] — длина строки."""
        if self.mm is None:
            raise RuntimeError("GGUF закрыт")
        dims, typ, off = self.tensors[name]
        tn, bs, be = TYPES[typ]
        K = dims[0]
        N = math.prod(dims[1:]) if len(dims) > 1 else 1
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
            if not isinstance(r, numbers.Integral) or isinstance(r, bool) or not 0 <= r < N:
                raise ValueError("ID строки вне словаря/тензора")
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


def validate_qwen(g):
    """Проверить метаданные и формы до выделения рабочих CUDA-буферов."""
    arch = g.meta.get('general.architecture')
    def integer(key):
        value = g.meta.get(f'{arch}.{key}')
        if type(value) is not int or value <= 0:
            raise ValueError(f'неверный параметр архитектуры {key}')
        return value
    L,D,H,KV,HD = [integer(key) for key in ('block_count','embedding_length','attention.head_count','attention.head_count_kv','attention.key_length')]
    if H % KV or HD & (HD-1) or HD < 2 or D % 256:
        raise ValueError('неподдерживаемые GQA/head_dim/embedding_length')
    for key in ('rope.freq_base','attention.layer_norm_rms_epsilon'):
        value=g.meta.get(f'{arch}.{key}')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'неверный параметр {key}')
    if g.meta.get(f'{arch}.rope.scaling.type','none') != 'none':
        raise NotImplementedError('масштабирование RoPE пока не поддерживается')
    if g.meta.get(f'{arch}.rope.dimension_count',HD) != HD:
        raise NotImplementedError('частичный RoPE пока не поддерживается')
    def tensor(name,k,n=None,matrix=False,expert=False):
        try:
            _,actual_n,actual_k,kind=g.raw(name)
        except KeyError as exc:
            raise ValueError(f'нет обязательного тензора {name}') from exc
        if actual_k!=k or n is not None and actual_n!=n:
            raise ValueError(f'форма {name} не соответствует архитектуре')
        allowed=('Q4_K','Q6_K') if expert else ('Q4_K','Q6_K','F16','F32')
        if matrix and kind not in allowed:
            raise NotImplementedError(f'тип {kind} у {name} не поддерживается')
        return actual_n
    V=tensor('token_embd.weight',D)
    tensor('output_norm.weight',D,1)
    if 'output.weight' in g.tensors:
        tensor('output.weight',D,V,matrix=True)
    if arch=='qwen3moe':
        E,K,F=[integer(key) for key in ('expert_count','expert_used_count','expert_feed_forward_length')]
        if K>E or K>256 or F%256:
            raise ValueError('неподдерживаемые размеры экспертов')
    else:
        F=integer('feed_forward_length')
        if F%256:
            raise ValueError('feed_forward_length не кратна 256')
    for i in range(L):
        prefix=f'blk.{i}.'
        for name,k,n in (('attn_norm',D,1),('ffn_norm',D,1),('attn_q_norm',HD,1),('attn_k_norm',HD,1)):
            tensor(prefix+name+'.weight',k,n)
        for name,k,n in (('attn_q',D,H*HD),('attn_k',D,KV*HD),('attn_v',D,KV*HD),('attn_output',H*HD,D)):
            tensor(prefix+name+'.weight',k,n,matrix=True)
        if arch=='qwen3moe':
            tensor(prefix+'ffn_gate_inp.weight',D,E)
            for name,k,n in (('ffn_gate_exps',D,E*F),('ffn_up_exps',D,E*F),('ffn_down_exps',F,E*D)):
                tensor(prefix+name+'.weight',k,n,matrix=True,expert=True)
        else:
            for name,k,n in (('ffn_gate',D,F),('ffn_up',D,F),('ffn_down',F,D)):
                tensor(prefix+name+'.weight',k,n,matrix=True)


def eos_ids(g):
    ids=set()
    tokens=g.meta.get('tokenizer.ggml.tokens',[])
    for key in ('tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id','tokenizer.ggml.eom_token_id'):
        if key not in g.meta:
            continue
        value=g.meta[key]
        if type(value) is not int or value<0 or tokens and value>=len(tokens):
            raise ValueError('неверный EOS ID в GGUF')
        ids.add(value)
    if not ids:
        raise ValueError('GGUF не содержит EOS ID')
    # Qwen3 завершает raw-текст также endoftext; ID берём из словаря, не из констант.
    if g.meta.get('general.architecture') in ('qwen3', 'qwen3moe'):
        tokens = g.meta.get('tokenizer.ggml.tokens', [])
        for marker in ('<|im_end|>', '<|endoftext|>'):
            if marker in tokens:
                ids.add(tokens.index(marker))
    return ids
