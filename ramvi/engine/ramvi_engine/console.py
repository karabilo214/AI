"""Единая кодировка вывода CLI в терминал и перенаправленные Windows-каналы."""

import io
import sys


def configure_output() -> None:
    """Сохранить русские сообщения при перенаправлении stdout/stderr на Windows."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
