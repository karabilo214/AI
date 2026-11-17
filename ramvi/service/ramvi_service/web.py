"""Общие правила выхода в сеть сервиса: каталог (4.2) и загрузки (4.3).

Только HTTPS, переадресация тоже только на HTTPS; обычный HTTP — лишь для
127.0.0.1/localhost (локальные тестовые серверы). Общий User-Agent, без cookies,
ключей и сведений о пользователе и железе.
"""

import urllib.request
from urllib.parse import urlsplit

USER_AGENT = "RAMVI"
LOOPBACK = ("127.0.0.1", "localhost")


class Refused(OSError):
    """Адрес или переадресация нарушают правило «только HTTPS»."""


def allowed(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK)


def check(url: str) -> None:
    if not allowed(url):
        raise Refused(f"only https: {url}")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check(newurl)  # переадресация тоже только на HTTPS; Range сохраняется urllib
        return super().redirect_request(req, fp, code, msg, headers, newurl)


OPENER = urllib.request.build_opener(_Redirects())


def request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    check(url)
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
