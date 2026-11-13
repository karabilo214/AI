"""Совместимость ID перед использованием общего токенового черновика."""
import hashlib,json

def fingerprint(tokenizer):
    payload={'vocab':tokenizer.get_vocab(),'special':tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def validate_tokenizer(engine,tokenizer):
    vocab=tokenizer.get_vocab()
    if hasattr(engine,'g'):
        tokens=engine.g.meta.get('tokenizer.ggml.tokens')
        if not isinstance(tokens,list):
            raise ValueError('в GGUF отсутствует словарь токенизатора')
        if any(not isinstance(idx,int) or not 0 <= idx < len(tokens) or tokens[idx] != token for token,idx in vocab.items()):
            raise ValueError('ID/словарь токенизатора не совпадают с GGUF')
        if len(tokens) != engine.V:
            raise ValueError('словарь GGUF не совпадает с выходной матрицей')
    if not engine.ck.eos.issubset(set(vocab.values())):
        raise ValueError('EOS модели отсутствует в токенизаторе')
    engine.tokenizer_fingerprint=fingerprint(tokenizer)
    return engine.tokenizer_fingerprint

def validate_draft(target, draft, target_tokenizer, draft_tokenizer):
    if target_tokenizer.get_vocab()!=draft_tokenizer.get_vocab() or target_tokenizer.special_tokens_map!=draft_tokenizer.special_tokens_map:
        raise ValueError('токенизаторы target и draft несовместимы')
    validate_tokenizer(target,target_tokenizer)
    validate_tokenizer(draft,draft_tokenizer)
