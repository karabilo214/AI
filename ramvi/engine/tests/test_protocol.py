"""Схема/версия, корреляция и bounded replay событий."""

import copy
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from typing import Any

import pytest
from ramvi_engine.protocol import EventContext, EventFactory, check_api_version, validate_event
from ramvi_engine.server.app import App, make_handler
from ramvi_engine.server.events import EventLog

from .test_metrics import Engine, Tokenizer


def test_strict_event_schema_and_owner() -> None:
    factory = EventFactory(
        EventContext("request", module_id="character", session_id="session", response_id="response")
    )
    event = factory.emit("module.character.state", {"state": "idle"})
    validate_event(event)
    assert event["sequence"] == 1 and event["version"] == 1
    assert factory.emit("module.character.state", {"state": "speaking"})["sequence"] == 2
    for bad in (
        {**event, "version": 2},
        {**event, "extra": True},
        {**event, "sequence": True},
        {**event, "type": "module.other.state"},
    ):
        with pytest.raises(ValueError):
            validate_event(bad)
    with pytest.raises(ValueError):
        factory.emit("generation.delta", {"kind": "text", "text": "x", "module_id": "other"})
    with pytest.raises(ValueError):
        factory.emit("module.character.state", {"x": float("nan")})
    with pytest.raises(ValueError):
        factory.emit("module.character.state", {"x": "x" * 70000})
    check_api_version("1.999")
    with pytest.raises(ValueError):
        check_api_version("2.0")


def test_bounded_event_replay_is_independent() -> None:
    log = EventLog(2)
    factory = EventFactory(EventContext("one"))
    for _ in range(4):
        log.publish(factory.emit("generation.delta", {"kind": "text", "text": "hello"}))
    result = log.read(after=1, request_id="one")
    assert result["gap"] and len(result["events"]) == 2
    result["events"].clear()
    assert len(log.read()["events"]) == 2
    assert not log.read(request_id="other")["events"]


def test_http_versions_headers_and_real_correlated_events() -> None:
    app = App(Engine(), Tokenizer(), "fixture", api_key="secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def call(
        method: str, path: str, body: dict[str, Any] | None = None, version: str = "1.0"
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.request(
                method,
                path,
                None if body is None else json.dumps(body),
                {
                    "Authorization": "Bearer secret",
                    "X-Request-ID": "correlation",
                    "X-RAMVI-API-Version": version,
                },
            )
            response = conn.getresponse()
            return response.status, json.loads(response.read()), dict(response.getheaders())
        finally:
            conn.close()

    try:
        code, body, headers = call("GET", "/engine/versions")
        assert code == 200 and body["versions"]["api"] == "1.0"
        assert headers["X-RAMVI-API-Version"] == "1.0"
        assert call("GET", "/engine/versions", version="2.0")[0] == 426
        code, _, _ = call("POST", "/v1/completions", {"prompt": "test"})
        assert code == 200
        code, body, _ = call("GET", "/engine/events?request_id=correlation")
        assert code == 200
        events = [row["event"] for row in body["events"]]
        assert (
            events[0]["type"] == "generation.started"
            and events[-1]["type"] == "generation.completed"
        )
        assert all(e["request_id"] == "correlation" and e["module_id"] == "engine" for e in events)
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
        for event in events:
            validate_event(copy.deepcopy(event))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
