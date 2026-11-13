"""Ограничение ответа грамматикой (xgrammar): JSON вообще или по схеме — ответ гарантированно ей соответствует.
Совместимо с угадыванием: маски считаются по пути черновика, после проверки лишнее откатывается."""
import json

import torch

try:
    import xgrammar as xgr
except ImportError:  # без xgrammar ограничения недоступны
    xgr = None

_COMPILERS = {}


def _compiler(tokenizer, V):
    key = (id(tokenizer), V)
    if key not in _COMPILERS:
        ti = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=V)
        _COMPILERS[key] = (xgr.GrammarCompiler(ti), ti)
    return _COMPILERS[key]


class Constraint:
    def __init__(self, tokenizer, V, schema=None):
        """schema: None — любой JSON-объект/массив; dict/str — JSON Schema."""
        if xgr is None:
            raise RuntimeError("для JSON по схеме нужен пакет xgrammar")
        comp, ti = _compiler(tokenizer, V)
        if schema is None:
            g = comp.compile_builtin_json_grammar()
        else:
            g = comp.compile_json_schema(schema if isinstance(schema, str) else json.dumps(schema))
        self.m = xgr.GrammarMatcher(g, max_rollback_tokens=64)
        self.V = V
        self.bm = xgr.allocate_token_bitmask(17, ti.vocab_size)

    def masks(self, path):
        """Маски для len(path)+1 позиций: позиция i — после принятия path[:i]. Матчер продвигается по пути,
        пока слова допустимы; возвращает (bitmask [n+1, ...] на видеокарте, сколько слов пути принято)."""
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
        for i in range(n + 1, k):  # дальше пути матчер не дошёл — эти строки не понадобятся
            self.bm[i] = self.bm[n]
        return self.bm[:k].to("cuda", non_blocking=True), n

    def commit(self, advanced, keep, tok):
        """После проверки: из advanced принятых матчером слов оставить keep, затем принять tok."""
        if advanced > keep:
            self.m.rollback(advanced - keep)
        if not self.m.is_terminated():
            self.m.accept_token(tok)

    def done(self):
        return self.m.is_terminated()

    @staticmethod
    def apply(logits, bitmask):
        xgr.apply_token_bitmask_inplace(logits, bitmask[:logits.shape[0]], vocab_size=logits.shape[1])
