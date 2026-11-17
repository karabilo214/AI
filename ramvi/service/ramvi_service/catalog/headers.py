"""Заголовки GGUF моделей каталога (4.4): чтение по сети только начала файла.

Заголовок читается запросом Range (сначала 8 МиБ, при нехватке — больше, не более
64 МиБ) и хранится в `<папка данных>/catalog/headers/<sha256>.json`. Файл с тем же
SHA-256 не меняется, поэтому копия не устаревает и повторно в сеть не ходим.
Веса не скачиваются.
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from ramvi_service import gguf_meta, web

FIRST = 8 << 20
LIMIT = 64 << 20
TIMEOUT = 60.0
SHA = re.compile(r"sha256:([0-9a-f]{64})")

ReadRange = Callable[[str, int], bytes]


def read_range(url: str, size: int) -> bytes:
    """Первые `size` байт; сервер без Range — читаем столько же и закрываем."""
    request = web.request(url, {"Range": f"bytes=0-{size - 1}"})
    with web.OPENER.open(request, timeout=TIMEOUT) as response:
        data: bytes = response.read(size)
    return data


def parse_prefix(read: Callable[[int], bytes], total: int | None = None) -> gguf_meta.GGUFMeta:
    """Читать начало файла, пока заголовок не разберётся (или не кончится файл/предел)."""
    size = FIRST
    while True:
        size = min(size, LIMIT, total) if total else min(size, LIMIT)
        data = read(size)
        try:
            return gguf_meta.parse(data)
        except gguf_meta.Truncated as short:
            if len(data) < size or size >= LIMIT or (total and size >= total):
                raise gguf_meta.NotGGUF("header longer than the readable prefix") from None
            size = max(short.need + (1 << 20), size * 2)


class HeaderStore:
    """Копии заголовков по SHA-256."""

    def __init__(self, folder: str | os.PathLike[str], read: ReadRange = read_range) -> None:
        self.folder, self.read = Path(folder), read

    def _path(self, checksum: str) -> Path | None:
        match = SHA.fullmatch(checksum)
        return self.folder / f"{match.group(1)}.json" if match else None

    def cached(self, checksum: str) -> gguf_meta.GGUFMeta | None:
        path = self._path(checksum)
        if path is None:
            return None
        try:
            return gguf_meta.GGUFMeta.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def fetch(self, url: str, checksum: str, total: int | None = None) -> gguf_meta.GGUFMeta:
        """Из копии или по сети; OSError/NotGGUF — заголовок не получен."""
        found = self.cached(checksum)
        if found is not None:
            return found
        path = self._path(checksum)
        if path is None:
            raise gguf_meta.NotGGUF("bad checksum")
        header = parse_prefix(lambda size: self.read(url, size), total)
        self.folder.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(header.json()), encoding="utf-8")
        os.replace(temp, path)
        return header


def read_file_header(path: str | os.PathLike[str]) -> gguf_meta.GGUFMeta:
    """Заголовок локального файла (установленная модель)."""
    file = Path(path)
    total = file.stat().st_size

    def read(size: int) -> bytes:
        with file.open("rb") as handle:
            return handle.read(size)

    return parse_prefix(read, total)
