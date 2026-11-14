"""Локальный транспорт: без userinfo, перенаправлений и системного proxy."""

import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit


def loopback_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as error:
        raise ValueError("неверный локальный URL") from error
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or port is None
        or not 0 < port < 65536
    ):
        raise ValueError("нужен чистый loopback HTTP URL с портом")
    host = "[::1]" if parts.hostname == "::1" else parts.hostname
    return urlunsplit(("http", f"{host}:{port}", "", "", ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def local_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
