"""Запись каталога моделей — общая схема всех провайдеров (4.2 MOD-01).

Провайдер (Hugging Face, реестр Ollama) переводит свои метаданные в `CatalogEntry`;
UI и подбор модели видят только эту схему. Обязательные поля по ТЗ: версия, ссылка,
размер, контрольная сумма, архитектура, квантование, лицензия, возможности.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ramvi_service import web

SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
REQUIRED = (
    "version",
    "url",
    "bytes",
    "checksum",
    "architecture",
    "quantization",
    "license",
    "capabilities",
)


@dataclass(frozen=True)
class CatalogEntry:
    """Один файл модели, доступный для скачивания (только метаданные)."""

    id: str  # «провайдер:источник/файл», стабилен между обновлениями каталога
    provider: str
    name: str  # для показа: «Qwen3-8B Q4_K_M»
    source: str  # репозиторий или модель:тег у провайдера
    version: str  # ревизия: коммит Hugging Face или digest манифеста Ollama
    url: str  # адрес файла, привязанный к этой ревизии
    page: str  # страница модели у провайдера
    file_name: str
    bytes: int
    checksum: str  # «sha256:<hex>» — проверяется загрузчиком до активации (4.3)
    format: str
    architecture: str
    quantization: str
    license: str
    capabilities: tuple[str, ...] = ()
    parameters: str | None = None  # размер модели, как его называет провайдер
    context_length: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CatalogEntry":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        known["capabilities"] = tuple(known.get("capabilities", ()))
        return cls(**known)


def problems(entry: CatalogEntry) -> list[str]:
    """Пустые или неверные обязательные поля; пустой список — запись годится."""
    found = [name for name in REQUIRED if not getattr(entry, name)]
    if entry.bytes is not None and (isinstance(entry.bytes, bool) or entry.bytes <= 0):
        found.append("bytes")
    if entry.checksum and not SHA256.fullmatch(entry.checksum):
        found.append("checksum")
    if entry.url and not web.allowed(entry.url):  # HTTPS (HTTP — только 127.0.0.1)
        found.append("url")
    return sorted(set(found))
