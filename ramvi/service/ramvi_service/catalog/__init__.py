"""Каталог моделей (4.2 MOD-01): адаптеры провайдеров, общая схема, локальная копия.

`Catalog.view()` читает только диск — работает без сети и показывает последнюю
сохранённую копию. `Catalog.refresh()` идёт к провайдерам; провайдер, который не
ответил, сохраняет прежние записи и получает пометку ошибки — его записи не
пропадают из-за обрыва сети. В показ попадают только записи с полным набором
обязательных полей и подтверждённым форматом (`support.py`).

Копия — JSON (`<папка данных>/catalog/catalog.json`, ТЗ §10: «каталог моделей как
JSON-cache»), запись атомарная: временный файл + os.replace.

4.4: при обновлении для показываемых записей читается и заголовок GGUF (только
начало файла, `headers.py`) — по нему считается статус совместимости без сети.
"""

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ramvi_service.catalog.entry import CatalogEntry, problems
from ramvi_service.catalog.headers import HeaderStore
from ramvi_service.catalog.providers import HuggingFace, OllamaRegistry, Provider
from ramvi_service.catalog.sources import HUGGINGFACE_REPOS, OLLAMA_TAGS
from ramvi_service.catalog.support import unsupported_reason
from ramvi_service.gguf_meta import NotGGUF
from ramvi_service.i18n import msg

SCHEMA = 1

__all__ = ["Catalog", "CatalogEntry", "default_providers"]


def default_providers() -> list[Provider]:
    return [HuggingFace(HUGGINGFACE_REPOS), OllamaRegistry(OLLAMA_TAGS)]


class Catalog:
    """Каталог поверх нескольких провайдеров с копией на диске."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        providers: list[Provider] | None = None,
        *,
        clock: Callable[[], float] = time.time,
        headers: HeaderStore | None = None,
    ) -> None:
        self.path = Path(data_dir) / "catalog" / "catalog.json"
        self.headers = headers if headers is not None else HeaderStore(self.path.parent / "headers")
        self.providers = default_providers() if providers is None else list(providers)
        self.clock = clock
        self._lock = threading.Lock()

    # --- диск ----------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}  # нет копии или она повреждена — каталог пуст, приложение работает
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return {}
        providers = data.get("providers")
        return providers if isinstance(providers, dict) else {}

    def _save(self, providers: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"schema": SCHEMA, "providers": providers}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        os.replace(temp, self.path)

    # --- обновление ----------------------------------------------------------------

    def refresh(self) -> dict[str, Any]:
        """Спросить всех провайдеров; при ошибке у провайдера остаётся прежняя копия."""
        with self._lock:
            stored = self._load()
            for provider in self.providers:
                block = dict(stored.get(provider.name) or {"entries": [], "fetched_at": None})
                block["checked_at"] = self.clock()
                try:
                    entries = provider.entries()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    block["error"] = msg("catalog.unavailable", provider=provider.name).json()
                    block["error_detail"] = type(exc).__name__  # без адресов и текста ответа
                else:
                    valid = [e for e in entries if not problems(e)]
                    block.update(
                        entries=[e.json() for e in valid],
                        rejected=len(entries) - len(valid),
                        fetched_at=self.clock(),
                        error=None,
                        error_detail=None,
                    )
                stored[provider.name] = block
            self._save(stored)
            self._fetch_headers()
        return self.view()

    def _fetch_headers(self) -> None:
        """Заголовки показываемых записей, которых ещё нет в копии; ошибка — без статуса."""
        missing = [e for e in self.view()["entries"] if self.headers.cached(e["checksum"]) is None]

        def one(entry: dict[str, Any]) -> None:
            try:
                self.headers.fetch(entry["url"], entry["checksum"], entry["bytes"])
            except (OSError, NotGGUF):
                pass  # статус останется «нет данных» до следующего обновления

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, missing))

    # --- показ ---------------------------------------------------------------------

    def view(self, *, include_unsupported: bool = False) -> dict[str, Any]:
        """Записи из локальной копии (без сети) и состояние каждого провайдера."""
        entries: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}
        for name, block in self._load().items():
            shown = hidden = 0
            for raw in block.get("entries", []):
                try:
                    entry = CatalogEntry.from_json(raw)
                except TypeError:
                    hidden += 1
                    continue
                reason = unsupported_reason(entry) if not problems(entry) else "fields"
                if reason is None or include_unsupported:
                    entries.append({**entry.json(), "unsupported": reason})
                if reason is None:
                    shown += 1
                else:
                    hidden += 1
            summary[name] = {
                "fetched_at": block.get("fetched_at"),
                "checked_at": block.get("checked_at"),
                "error": block.get("error"),
                "shown": shown,
                "hidden": hidden,
            }
        offline = any(item["error"] for item in summary.values())
        return {"providers": summary, "entries": entries, "stale": offline}
