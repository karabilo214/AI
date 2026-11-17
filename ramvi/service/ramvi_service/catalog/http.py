"""Чтение метаданных каталога по сети: только GET, только HTTPS, ограниченный размер.

Правила адресов и User-Agent — общие с загрузками (`ramvi_service.web`).
"""

from collections.abc import Mapping

from ramvi_service import web

USER_AGENT = web.USER_AGENT
TIMEOUT = 20.0
LIMIT = 8 * 1024 * 1024  # метаданные; больше — ошибка, а не чтение в память


class FetchError(OSError):
    """Сеть, статус или размер ответа; каталог остаётся прежним."""


def fetch(url: str, headers: Mapping[str, str] | None = None, *, limit: int = LIMIT) -> bytes:
    """Тело ответа 200; иначе FetchError."""
    try:
        with web.OPENER.open(web.request(url, dict(headers or {})), timeout=TIMEOUT) as response:
            body: bytes = response.read(limit + 1)
    except OSError as exc:  # Refused, URLError, HTTPError, таймаут
        raise FetchError(f"{url}: {exc}") from exc
    if len(body) > limit:
        raise FetchError(f"{url}: response larger than {limit} bytes")
    return body
