"""Управление текущей настроенной моделью без изменения путей из HTTP-запроса."""

from typing import Any

from ramvi_engine.lifecycle import EngineSession
from ramvi_engine.protocol import EventContext, EventFactory

PATHS = {"/engine/load", "/engine/warmup", "/engine/unload", "/engine/restart", "/engine/cancel"}


def lifecycle_post(
    app: Any, path: str, req: dict[str, Any], context: EventContext | None = None
) -> dict[str, Any]:
    session = app.eng
    if not isinstance(session, EngineSession):
        raise NotImplementedError("движок запущен без управляемого жизненного цикла")
    allowed = (
        {"tokens", "max_new"}
        if path == "/engine/warmup"
        else ({"request_id"} if path == "/engine/cancel" else set())
    )
    if set(req) - allowed:
        raise ValueError("неизвестные параметры lifecycle")
    factory = EventFactory(context) if context is not None else None
    if factory and path in ("/engine/load", "/engine/restart"):
        app.events.publish(factory.emit("engine.loading", {"state": "loading", "model": app.name}))
    if path == "/engine/warmup":
        tokens = req.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("warmup требует tokens")
        return session.warmup(tokens, max_new=req.get("max_new", 2))
    if path == "/engine/cancel":
        return session.cancel(req.get("request_id"))
    operations = {
        "/engine/load": session.load,
        "/engine/unload": session.unload,
        "/engine/restart": session.restart,
        "/engine/cancel": session.cancel,
    }
    try:
        result = operations[path]()
    except Exception as exc:
        if factory:
            app.events.publish(
                factory.emit("engine.failed", {"state": "error", "error_type": type(exc).__name__})
            )
        raise
    if factory and path in ("/engine/load", "/engine/restart", "/engine/unload"):
        kind = "engine.unloaded" if path == "/engine/unload" else "engine.ready"
        app.events.publish(factory.emit(kind, {"state": result["state"], "model": app.name}))
    return result
