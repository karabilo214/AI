"""Чтение метаданных каталога по сети: только GET, только HTTPS, ограниченный размер.

Запрос не несёт ничего о пользователе и железе: общий User-Agent, без cookies и
ключей. Обычный HTTP разрешён только для 127.0.0.1/localhost (тестовые серверы).
"""

import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit

USER_AGENT = "RAMVI"
TIMEOUT = 20.0
LIMIT = 8 * 1024 * 1024  # метаданные; больше — ошибка, а не чтение в память
LOOPBACK = ("127.0.0.1", "localhost")


class FetchError(OSError):
    """Сеть, статус или размер ответа; каталог остаётся прежним."""


def _check(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK):
        return
    raise FetchError(f"only https: {url}")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _check(newurl)  # переадресация тоже только на HTTPS
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirects())


def fetch(url: str, headers: Mapping[str, str] | None = None, *, limit: int = LIMIT) -> bytes:
    """Тело ответа 200; иначе FetchError."""
    _check(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with _OPENER.open(request, timeout=TIMEOUT) as response:
            body: bytes = response.read(limit + 1)
    except OSError as exc:  # URLError, HTTPError, таймаут
        raise FetchError(f"{url}: {exc}") from exc
    if len(body) > limit:
        raise FetchError(f"{url}: response larger than {limit} bytes")
    return body
