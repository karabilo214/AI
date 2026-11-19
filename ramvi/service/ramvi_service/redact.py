"""8.4 (раздел 11 ТЗ): маскирование секретов в сообщениях для интерфейса и журналов.

Тексты ошибок сторонних сервисов (например, «Incorrect API key provided: sk-…»),
подробности сбоя движка и внутренние ошибки могут содержать ключи. Перед отправкой
интерфейсу каждая строка параметров ошибки проходит `mask_secrets`: ключ заменяется
началом и многоточием («sk-a…»), имя поля сохраняется.
"""

import re
from typing import Any

_KEEP = 4
_PATTERNS = (
    re.compile(r"\bramvi-[A-Za-z0-9_\-]{16,}"),  # ключи клиентов RAMVI
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),  # OpenAI-совместимые ключи (в т.ч. sk-proj-…)
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),  # ключи Google
    re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{16,}"),
)
_BEARER = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=\-]{8,})")
_FIELD = re.compile(
    r"(?i)\b(api[_-]?key|x-api-key|token|access[_-]?token|password|passwd|secret)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&]{6,})"
)


def _short(value: str) -> str:
    return value[:_KEEP] + "…"


def mask_secrets(text: str) -> str:
    """Строка без ключей: ключ — первые символы и многоточие."""
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: _short(m.group(0)), text)
    text = _BEARER.sub(lambda m: m.group(1) + _short(m.group(2)), text)
    return _FIELD.sub(lambda m: m.group(1) + m.group(2) + _short(m.group(3)), text)


def mask_value(value: Any) -> Any:
    """Рекурсивно для параметров ошибки (строки, списки, словари)."""
    if isinstance(value, str):
        return mask_secrets(value)
    if isinstance(value, dict):
        return {k: mask_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [mask_value(v) for v in value]
    return value
