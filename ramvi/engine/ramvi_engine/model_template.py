"""Политика chat template для явно помеченных thinking-only GGUF."""

from typing import Any


def configure_model_template(engine: Any, tokenizer: Any) -> None:
    """Не подменять thinking-only шаблон общим переключаемым токенизатором."""
    metadata = getattr(getattr(engine, "g", None), "meta", {})
    finetune = metadata.get("general.finetune", "")
    required = isinstance(finetune, str) and finetune.casefold() == "thinking"
    legacy = getattr(tokenizer, "chat_template", None)
    source = "tokenizer"
    if required:
        template = metadata.get("tokenizer.chat_template")
        if not isinstance(template, str) or not template or len(template.encode()) > 65536:
            raise ValueError("Thinking-only GGUF требует встроенный chat template")
        if "<think>" not in template:
            raise ValueError("Thinking-only GGUF не содержит открывающего thinking-маркера")
        tokenizer.chat_template = template
        source = "gguf"
    tokenizer.ramvi_legacy_chat_template = legacy
    tokenizer.ramvi_thinking_required = required
    tokenizer.ramvi_template_source = source
    engine.thinking_required = required
    engine.chat_template_source = source
