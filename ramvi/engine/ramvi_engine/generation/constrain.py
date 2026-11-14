"""Ограничение ответа грамматикой (xgrammar): JSON вообще или по схеме — ответ гарантированно ей соответствует.
Совместимо с угадыванием: маски считаются по пути черновика, после проверки лишнее откатывается."""
from ramvi_engine.config import ensure_dir, work_dir
import json
import threading
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import OrderedDict

import torch

try:
    import xgrammar as xgr
except ImportError:  # без xgrammar ограничения недоступны
    xgr = None

_COMPILERS = OrderedDict()
_GRAMMARS = OrderedDict()
_COMPILE_LOCK = threading.Lock()


def _compiler(tokenizer, V):
    key = (id(tokenizer), V)
    if key not in _COMPILERS:
        ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=V)
        _COMPILERS[key] = (xgr.GrammarCompiler(ti), ti, tokenizer)
        while len(_COMPILERS) > 4:
            _COMPILERS.popitem(last=False)
    _COMPILERS.move_to_end(key)
    return _COMPILERS[key][:2]


def _schema(schema):
    if schema is None:
        return None, "builtin"
    obj = json.loads(schema) if isinstance(schema, str) else schema
    if not isinstance(obj, (dict, bool)):
        raise ValueError("JSON Schema должна быть объектом или boolean")
    encoded = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    if len(encoded.encode()) > 32768:
        raise ValueError("JSON Schema слишком велика")
    stack, nodes = [(obj, 0)], 0
    while stack:
        value, depth = stack.pop(); nodes += 1
        if nodes > 4096 or depth > 32:
            raise ValueError("JSON Schema слишком сложная")
        if isinstance(value, dict):
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ValueError("внешние ссылки JSON Schema не поддерживаются")
            stack.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            stack.extend((v, depth + 1) for v in value)
    import jsonschema
    try:
        jsonschema.Draft202012Validator.check_schema(obj)
    except jsonschema.SchemaError as exc:
        raise ValueError("некорректная JSON Schema") from exc
    return obj, encoded


def _compile_bounded(info, schema, timeout=30):
    with tempfile.TemporaryDirectory(prefix="ramvi-grammar-", dir=ensure_dir(work_dir())) as folder:
        input_path = Path(folder) / "input.json"
        output_path = Path(folder) / "output.json"
        input_path.write_text(json.dumps({"tokenizer": info.serialize_json(), "schema": schema}), encoding="utf-8")
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name("grammar_worker.py")),
                            str(input_path), str(output_path)], check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("истекло время компиляции JSON Schema") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError("не удалось скомпилировать JSON Schema: " + exc.stderr[-1000:]) from exc
        if output_path.stat().st_size > 128 * 2**20:
            raise ValueError("скомпилированная грамматика слишком велика")
        return xgr.CompiledGrammar.deserialize_json(output_path.read_text(encoding="utf-8"), info)


class Constraint:
    def __init__(self, tokenizer, V, schema=None):
        """schema: None — любой JSON-объект/массив; dict/str — JSON Schema."""
        if xgr is None:
            raise RuntimeError("для JSON по схеме нужен пакет xgrammar")
        self.schema, encoded = _schema(schema)
        with _COMPILE_LOCK:
            comp, ti = _compiler(tokenizer, V)
            key = (id(comp), encoded)
            if key not in _GRAMMARS:
                grammar = _compile_bounded(ti, None if schema is None else encoded)
                _GRAMMARS[key] = (grammar, comp)  # удерживаем compiler, не допускаем повторного использования id
                while len(_GRAMMARS) > 16:
                    _GRAMMARS.popitem(last=False)
            _GRAMMARS.move_to_end(key)
            g = _GRAMMARS[key][0]
        self.m = xgr.GrammarMatcher(g, max_rollback_tokens=64)
        self.V = V
        self.bm = xgr.allocate_token_bitmask(17, ti.vocab_size)

    def masks(self, path):
        """Маски для len(path)+1 позиций: позиция i — после принятия path[:i]. Матчер продвигается по пути,
        пока слова допустимы; возвращает (bitmask [n+1, ...] на видеокарте, сколько слов пути принято)."""
        if len(path) > 15:
            raise ValueError("путь грамматики длиннее проверяющей пачки")
        n = 0
        self.m.fill_next_token_bitmask(self.bm, 0)
        for i, t in enumerate(path):
            if self.m.is_terminated() or not self.m.accept_token(t):
                break
            n += 1
            if self.m.is_terminated():  # принято слово конца — дальше масок нет
                break
            self.m.fill_next_token_bitmask(self.bm, i + 1)
        k = len(path) + 1
        if self.m.is_terminated():
            # После EOS строка n ещё не заполнялась: не читаем неинициализированную память.
            # Проверка остановит выдачу на EOS, но CUDA рассчитывает и остаток пачки.
            for i in range(n, k):
                self.bm[i] = self.bm[max(0, n - 1)]
        else:
            for i in range(n + 1, k):
                self.bm[i] = self.bm[n]
        return self.bm[:k].to("cuda", non_blocking=True), n

    def commit(self, advanced, keep, tok):
        """После проверки: из advanced принятых матчером слов оставить keep, затем принять tok."""
        if advanced > keep:
            self.m.rollback(advanced - keep)
        if not self.m.is_terminated():
            if not self.m.accept_token(tok):
                raise RuntimeError("выбранный токен нарушает грамматику")

    def validate(self, text):
        obj = json.loads(text)
        if self.schema is not None:
            import jsonschema
            try:
                jsonschema.Draft202012Validator(self.schema).validate(obj)
            except jsonschema.ValidationError as exc:
                raise ValueError("ответ не соответствует JSON Schema") from exc
        return obj

    def done(self):
        return self.m.is_terminated()

    @staticmethod
    def apply(logits, bitmask):
        xgr.apply_token_bitmask_inplace(logits, bitmask[:logits.shape[0]], vocab_size=logits.shape[1])
