"""Что каталог показывает: только подтверждённо поддерживаемые файлы (4.2 MOD-01).

Подтверждено = движок умеет такие матрицы (`formats.gguf.validate_qwen`) **и** файл
этого вида проверен на железе: qwen3 Q4_K_M (WIDE30, экзамен 1), qwen3 F16
(пункт 3.18), qwen3moe Q4_K_M (этап 2). Новый квант добавляется сюда только после
проверки на железе. Q6_K, Q8_0, Q5_*, BF16 и кванты IQ* скрыты: у одних матриц нет
ядер движка, другие не проверены целиком.
"""

from ramvi_engine.capabilities import capability_catalog

from ramvi_service.catalog.entry import CatalogEntry

CONFIRMED_QUANTS: dict[str, tuple[str, ...]] = {
    "qwen3": ("Q4_K_M", "F16"),
    "qwen3moe": ("Q4_K_M",),  # у экспертов движок принимает только Q4_K/Q6_K
}


def capabilities(model_format: str, architecture: str) -> tuple[str, ...]:
    """Возможности из карты движка (2.1), а не из обещаний провайдера."""
    for row in capability_catalog()["rows"]:
        if row["format"] == model_format and row["architecture"] == architecture:
            found = ["chat"]
            if row["tools"]["supported"]:
                found.append("tools")
            if row["json"]["supported"]:
                found.append("json")
            if row["draft"]["supported"]:
                found.append("draft")
            return tuple(found)
    return ()


def unsupported_reason(entry: CatalogEntry) -> str | None:
    """Почему файл не показывается; None — поддерживается."""
    if entry.format != "gguf":
        return "format"
    allowed = CONFIRMED_QUANTS.get(entry.architecture)
    if allowed is None:
        return "architecture"
    if entry.quantization not in allowed:
        return "quantization"
    return None
