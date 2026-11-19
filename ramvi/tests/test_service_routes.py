"""Реестр методов Scoped Services: у каждого метода есть право, и оно проверяется до обработчика."""

import inspect
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import services
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority
from ramvi_service.events import EventBus
from ramvi_service.resources import ResourceBroker
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.services import ROUTES, ScopedServices, handler

METHODS = {
    "settings.get",
    "settings.set",
    "data.get",
    "data.set",
    "data.query",
    "data.execute",
    "secrets.get",
    "secrets.set",
    "secrets.delete",
    "events.begin",
    "events.emit",
    "events.cancel",
    "events.subscribe",
    "events.poll",
    "events.unsubscribe",
    "resources.acquire",
    "resources.release",
    "llm.capabilities",
    "llm.complete",
    "llm.stream",
    "llm.read",
    "llm.cancel",
    "context.capture",
    "agent.create",
    "agent.status",
    "agent.events",
    "agent.cancel",
    "agent.forget",
    "speech.catalog",
    "speech.configure",
    "speech.key",
    "speech.start",
    "speech.status",
    "speech.cancel",
    "voice.capabilities",
    "voice.listen.start",
    "voice.listen.stop",
    "voice.listen.cancel",
    "voice.session.start",
    "voice.session.read",
    "voice.session.cancel",
    "voice.session.playback",
}
# Право, которое ожидается для вызова с PAYLOAD.
EXPECTED = {method: (method.split(".")[0], "*") for method in METHODS}
EXPECTED["context.capture"] = ("context.ui", "editor")
PAYLOAD: dict[str, Any] = {"application": "editor", "window_id": "w1"}


class NoAdapter:
    def capture(
        self, application: str, window: str, kind: str, cancel: threading.Event
    ) -> dict[str, Any]:
        raise AssertionError("обработчик подменён, адаптер не вызывается")


def scope(tmp_path: Path) -> tuple[ScopedServices, GrantStore]:
    authority = Authority()
    owner = authority.issue("chat", "user", "a")
    grants = GrantStore(authority)
    result = ScopedServices(
        owner,
        authority,
        grants,
        ScopedData(tmp_path, owner.module_id),
        ScopedSecrets(MemorySecrets(), owner.module_id),
        ResourceBroker(authority, {}),
        EventBus(authority),
        context=ContextProvider(authority, grants, NoAdapter()),
    )
    result.active = True
    return result, grants


def test_every_method_is_registered_with_permission() -> None:
    assert set(ROUTES) == METHODS
    for method, route in ROUTES.items():
        assert route.permission(PAYLOAD) == EXPECTED[method], method


@pytest.mark.parametrize("method", sorted(METHODS))
def test_call_without_permission_is_rejected_before_handler(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def run(_scope: ScopedServices, _payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(method)
        return {"ok": True}

    spy = replace(ROUTES[method], run=run)
    monkeypatch.setitem(ROUTES, method, spy)
    services_scope, grants = scope(tmp_path)
    with pytest.raises(PermissionError):
        services_scope.call(method, dict(PAYLOAD))
    assert calls == []
    grants.allow(services_scope.owner, *EXPECTED[method])
    assert services_scope.call(method, dict(PAYLOAD)) == {"ok": True}
    assert calls == [method]


def test_unknown_method_is_one_common_error(tmp_path: Path) -> None:
    services_scope, grants = scope(tmp_path)
    grants.allow(services_scope.owner, "settings")
    for method in ("settings.unknown", "nothing", ""):
        with pytest.raises(ValueError, match="unknown Scoped Services method"):
            services_scope.call(method, {})


def test_inactive_module_reaches_only_inactive_methods(tmp_path: Path) -> None:
    services_scope, grants = scope(tmp_path)
    services_scope.active = False
    for method in ("resources", "llm"):
        grants.allow(services_scope.owner, method)
    for method in ("resources.release", "llm.complete", "llm.stream", "context.capture"):
        with pytest.raises(PermissionError, match="not active"):
            services_scope.call(method, dict(PAYLOAD))
    grants.allow(services_scope.owner, "settings")
    assert services_scope.call("settings.get", {}) == {"value": {}}


def test_dispatch_has_no_branches_on_method_names() -> None:
    source = inspect.getsource(ScopedServices._dispatch)
    assert "method ==" not in source and "startswith" not in source
    assert not re.search(r"[\"'][a-z]+\.[a-z_]+[\"']", source)


def test_registration_requires_permission_and_unique_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(services, "ROUTES", dict(ROUTES))
    with pytest.raises(ValueError, match="needs a permission"):
        handler("x.y", permission="")
    with pytest.raises(ValueError, match="already registered"):
        handler("settings.get", permission="settings")(lambda _scope, _payload: {})


def test_stop_denies_new_speech_before_releasing_and_releases_after_inflight(
    tmp_path: Path,
) -> None:
    services_scope, grants = scope(tmp_path)
    grants.allow(services_scope.owner, "speech")
    calls = []

    class Speech:
        def release(self, owner: Any) -> None:
            assert not services_scope.active
            with pytest.raises(PermissionError, match="scope stopped"):
                services_scope.call("speech.start", {})
            calls.append(owner)

    services_scope.speech = Speech()
    services_scope.stop()
    assert calls == [services_scope.owner, services_scope.owner]
