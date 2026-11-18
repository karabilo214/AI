"""7.6 (CORE-API-01): версионированное описание возможностей RAMVI поверх API.

Одна схема (`ramvi.capabilities/1`) для интерфейса и внешних клиентов: версия
Module API, формат событий, возможности ядра (чат, агент, внешний API) и модули —
версии, состояние, возможности, права, объявленные события. Права модулей
определяет Host (одобренные при регистрации); клиент не может получить их, подставив
`module_id`: владелец запроса всегда выводится из токена (Authority), а служебные
поля из тела внешнего запроса отбрасываются (`api_server.RESERVED`).

Внешний клиент видит сокращённое описание: модули без прав и схем событий.
"""

from typing import Any

from ramvi_service.contracts.modules import MODULE_API_VERSION
from ramvi_service.events import EVENT_VERSION

SCHEMA = "ramvi.capabilities/1"
EXTERNAL_FORMATS = ("openai", "anthropic", "ollama")


def describe(host: Any, api: Any = None, *, external: bool = False) -> dict[str, Any]:
    """Описание возможностей. `external` — для внешнего клиента (без прав и схем)."""
    modules = [] if host is None else host.describe()
    if external:
        modules = [
            {k: m[k] for k in ("module_id", "version", "api_version", "state")} for m in modules
        ]
    core: dict[str, Any] = {
        "chat": True,
        "agent": host is not None and getattr(host, "agent", None) is not None,
        "context": host is not None and getattr(host, "context", None) is not None,
    }
    if api is not None:
        core["external_api"] = {
            "formats": list(EXTERNAL_FORMATS),
            "streaming": True,
            "network": api.options.network,
            "queue": api.queue_status(),
        }
    return {
        "schema": SCHEMA,
        "module_api": MODULE_API_VERSION,
        "events": {"version": EVENT_VERSION, "type": "module.<module_id>.<name>"},
        "core": core,
        "modules": modules,
    }
