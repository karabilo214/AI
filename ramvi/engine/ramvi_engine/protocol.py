"""Версии и строгий envelope внутренних событий без вычислительных зависимостей."""

import json
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

API_VERSION = "1.0"
EVENT_VERSION = 1
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TYPE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_-]*)+\Z")


def validate_id(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("invalid identifier")
    return value


def versions() -> dict[str, Any]:
    return {
        "api": API_VERSION,
        "events": EVENT_VERSION,
        "metrics": 1,
        "resources": 1,
        "capabilities": 1,
    }


def check_api_version(value: str) -> None:
    if not re.fullmatch(r"\d+\.\d+", value) or value.split(".")[0] != API_VERSION.split(".")[0]:
        raise ValueError("incompatible API version")


def _json_value(value: Any, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("event payload too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for child in value:
            _json_value(child, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for child in value.values():
            _json_value(child, depth + 1)
        return
    raise ValueError("event payload must be finite JSON")


@dataclass(frozen=True)
class EventContext:
    request_id: str
    module_id: str = "engine"
    session_id: str | None = None
    response_id: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        validate_id(self.request_id)
        validate_id(self.module_id)
        for value in (self.session_id, self.response_id, self.run_id):
            if value is not None:
                validate_id(value)


# Обязательные и допустимые поля core payload; сторонние типы только в module namespace.
_PAYLOADS: dict[str, tuple[dict[str, type], dict[str, type]]] = {
    "generation.started": ({"model": str}, {}),
    "engine.failed": ({"state": str, "error_type": str}, {}),
    "generation.delta": ({"kind": str, "text": str}, {}),
    "generation.completed": ({"finish": str, "metrics": dict}, {}),
    "generation.failed": ({"error_type": str, "outcome": str}, {}),
    "engine.loading": ({"state": str}, {"model": str}),
    "engine.ready": ({"state": str}, {"model": str}),
    "engine.unloaded": ({"state": str}, {"model": str}),
    "engine.oom": ({"state": str}, {"model": str}),
}


def validate_event(event: dict[str, Any]) -> None:
    expected = {
        "version",
        "type",
        "request_id",
        "module_id",
        "session_id",
        "response_id",
        "run_id",
        "event_id",
        "sequence",
        "monotonic_ns",
        "payload",
    }
    if (
        not isinstance(event, dict)
        or set(event) != expected
        or type(event["version"]) is not int
        or event["version"] != EVENT_VERSION
    ):
        raise ValueError("invalid event envelope/version")
    EventContext(
        event["request_id"],
        event["module_id"],
        event["session_id"],
        event["response_id"],
        event["run_id"],
    )
    validate_id(event["event_id"])
    for name in ("sequence", "monotonic_ns"):
        if isinstance(event[name], bool) or not isinstance(event[name], int) or event[name] < 0:
            raise ValueError("invalid event order/time")
    kind, payload = event["type"], event["payload"]
    if not isinstance(kind, str) or not _TYPE.fullmatch(kind) or not isinstance(payload, dict):
        raise ValueError("invalid event type/payload")
    if kind in _PAYLOADS:
        required, optional = _PAYLOADS[kind]
        if not set(required) <= set(payload) or set(payload) - (set(required) | set(optional)):
            raise ValueError("unexpected event payload fields")
        for key, cls in (required | optional).items():
            if key in payload and not isinstance(payload[key], cls):
                raise ValueError("invalid event payload field type")
        if kind == "generation.delta" and payload["kind"] not in ("text", "reasoning"):
            raise ValueError("invalid generation delta kind")
    elif not kind.startswith("module." + event["module_id"] + "."):
        raise ValueError("unregistered event type or foreign module namespace")
    _json_value(payload)
    if len(json.dumps(event, ensure_ascii=False, allow_nan=False).encode()) > 65536:
        raise ValueError("event too large")


class EventFactory:
    def __init__(self, context: EventContext) -> None:
        self.context = context
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "version": EVENT_VERSION,
                "type": kind,
                "request_id": self.context.request_id,
                "module_id": self.context.module_id,
                "session_id": self.context.session_id,
                "response_id": self.context.response_id,
                "run_id": self.context.run_id,
                "event_id": uuid.uuid4().hex,
                "sequence": self._sequence,
                "monotonic_ns": time.monotonic_ns(),
                "payload": payload,
            }
            validate_event(event)
            snapshot: dict[str, Any] = json.loads(
                json.dumps(event, ensure_ascii=False, allow_nan=False)
            )
            validate_event(snapshot)
            return snapshot
