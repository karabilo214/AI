"""8.6, экзамен 4: ловушка «нет сети» для всех процессов Python комплекта.

Кладётся на время проверки в `runtime/python/Lib/site-packages/sitecustomize.py` и
работает, только если задана переменная RAMVI_OFFLINE_GUARD (путь к журналу): любое
соединение и DNS-запрос не к петле 127.0.0.1 отклоняются и пишутся в журнал. Нужна, когда
нет прав администратора на правило брандмауэра. После проверки файл удаляется.
"""

import os

_LOG = os.environ.get("RAMVI_OFFLINE_GUARD")
if _LOG:
    import socket
    import sys

    _LOOP = {"127.0.0.1", "::1", "localhost"}
    _connect, _connect_ex, _getaddrinfo = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def _note(what: str) -> None:
        with open(_LOG, "a", encoding="utf-8") as log:
            log.write(f"{os.getpid()}\t{' '.join(sys.argv)[:160]}\t{what}\n")

    def _outside(sock: socket.socket, address: object) -> bool:
        host = address[0] if isinstance(address, tuple) else str(address)
        return sock.family in (socket.AF_INET, socket.AF_INET6) and host not in _LOOP

    def connect(self: socket.socket, address: object) -> None:
        if _outside(self, address):
            _note(f"BLOCKED connect {address}")
            raise OSError(10051, "offline exam: network is unreachable")
        return _connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        if _outside(self, address):
            _note(f"BLOCKED connect_ex {address}")
            return 10051
        return _connect_ex(self, address)  # type: ignore[arg-type]

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        name = host.decode() if isinstance(host, bytes) else host
        numeric = (
            isinstance(name, str) and all(p.isdigit() for p in name.split(".")) and "." in name
        )
        if name is not None and name not in _LOOP and not numeric:
            _note(f"BLOCKED dns {name}")
            raise socket.gaierror(11001, "offline exam: no DNS")
        return _getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    _note("guard on")
