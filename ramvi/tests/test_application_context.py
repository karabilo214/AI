"""6.11: независимые гранты, свежесть, бюджет и отсутствие скрытых снимков."""

import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, ProviderCapabilities
from ramvi_service.system_context import SystemContextAdapter


class Adapter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.data: dict[str, Any] = {"text": "private selected text", "structure": {"elements": []}}

    def capture(self, application: str, window: str, kind: str, cancel: Any) -> dict[str, Any]:
        self.calls.append(kind)
        return self.data if kind == "ui" else {"image": "data:image/png;base64,private"}


def setup() -> tuple[Any, Any, Any, Any, Adapter]:
    authority = Authority()
    owner = authority.issue("character", "user", "session")
    grants = GrantStore(authority)
    adapter = Adapter()
    return (
        owner,
        grants,
        ContextProvider(authority, grants, adapter, min_interval=0),
        authority,
        adapter,
    )


@pytest.mark.parametrize("permission", ["context.ui", "context.screen", "agent.control"])
def test_three_grants_do_not_imply_each_other(permission: str) -> None:
    owner, grants, context, _, adapter = setup()
    grants.allow(owner, permission, "app")
    for kind in ("ui", "screen"):
        if permission == "context." + kind:
            context.capture(
                owner,
                "app",
                "window",
                kind=kind,
                capabilities=ProviderCapabilities(image_input=True),
            )
        else:
            with pytest.raises(PermissionError):
                context.capture(
                    owner,
                    "app",
                    "window",
                    kind=kind,
                    capabilities=ProviderCapabilities(image_input=True),
                )
    assert len(adapter.calls) == (0 if permission == "agent.control" else 1)
    if permission != "agent.control":
        with pytest.raises(PermissionError):
            grants.require(owner, "agent.control", "app")


def test_structured_context_first_and_no_automatic_screen() -> None:
    owner, grants, context, _, adapter = setup()
    grants.allow(owner, "context.ui", "app")
    grants.allow(owner, "context.screen", "app")
    assert context.capture_best(owner, "app", "window").text
    assert adapter.calls == ["ui"]
    adapter.data = {"text": ""}
    with pytest.raises(NotImplementedError):
        context.capture_best(owner, "app", "window")
    assert adapter.calls == ["ui", "ui"]


def test_revoked_and_reallowed_read_cannot_reuse_snapshot() -> None:
    owner, grants, context, _, _ = setup()
    grants.allow(owner, "context.ui", "app")
    grants.allow(owner, "egress.text", "app")
    snapshot = context.capture(owner, "app", "window")
    assert context.authorize_egress(owner, snapshot)["text"]
    grants.revoke(owner, "context.ui", "app")
    grants.allow(owner, "context.ui", "app")
    with pytest.raises(PermissionError):
        context.authorize_egress(owner, snapshot)
    new = context.capture(owner, "app", "window")
    context.authorize_egress(owner, new)
    with pytest.raises(PermissionError):
        context.validate_snapshot(owner, replace(new, window_id="forged"))
    with pytest.raises(TimeoutError):
        context.validate_snapshot(owner, new, max_age=-1)


def test_total_budget_and_rate_precede_collection() -> None:
    owner, grants, _, authority, adapter = setup()
    grants.allow(owner, "context.ui", "app")
    context = ContextProvider(authority, grants, adapter, min_interval=0, max_bytes_per_minute=1)
    with pytest.raises(TimeoutError):
        context.capture(owner, "app", "window")
    context = ContextProvider(authority, grants, adapter, min_interval=60)
    context.capture(owner, "app", "window")
    calls = len(adapter.calls)
    with pytest.raises(TimeoutError):
        context.capture(owner, "app", "window")
    assert len(adapter.calls) == calls


def test_revocation_during_capture_or_send_drops_result() -> None:
    owner, grants, _, authority, adapter = setup()
    grants.allow(owner, "context.ui", "app")

    def revoked_capture(application: str, window: str, kind: str, cancel: Any) -> dict[str, Any]:
        grants.revoke(owner, "context.ui", "app")
        grants.allow(owner, "context.ui", "app")
        return {"text": "private"}

    adapter.capture = revoked_capture  # type: ignore[method-assign]
    context = ContextProvider(authority, grants, adapter, min_interval=0)
    with pytest.raises(PermissionError):
        context.capture(owner, "app", "window")
    adapter.capture = Adapter().capture  # type: ignore[method-assign]
    snapshot = context.capture(owner, "app", "window")
    grants.allow(owner, "egress.text", "app")

    def sender(data: Any, cancel: threading.Event) -> None:
        grants.revoke(owner, "egress.text", "app")
        assert cancel.is_set()

    with pytest.raises(PermissionError):
        context.transmit(owner, snapshot, sender)


def test_unsupported_os_has_explicit_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "unsupported")
    adapter = SystemContextAdapter()
    assert adapter.capabilities() == {
        "ui": False,
        "screen": False,
        "control": False,
        "backend": None,
    }
    with pytest.raises(NotImplementedError):
        adapter.current_target()


def test_scoped_external_model_cannot_bypass_application_egress(tmp_path: Path) -> None:
    from ramvi_service.events import EventBus
    from ramvi_service.resources import ResourceBroker, SharedLLM
    from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
    from ramvi_service.services import ScopedServices

    owner, grants, context, authority, _ = setup()

    class Provider:
        capabilities = ProviderCapabilities(external=True)

        def complete(self, request: Any, cancel: Any) -> dict[str, Any]:
            return {"reply": "ok"}

        def cancel(self) -> None:
            pass

        def unload(self) -> None:
            pass

    model = SharedLLM(authority, Provider())
    scope = ScopedServices(
        owner,
        authority,
        grants,
        ScopedData(tmp_path, owner.module_id),
        ScopedSecrets(MemorySecrets(), owner.module_id),
        ResourceBroker(authority, {}),
        EventBus(authority),
        model,
        context,
    )
    scope.active = True
    for right in ("llm", "egress.text"):
        grants.allow(owner, right)
    grants.allow(owner, "context.ui", "app")
    try:
        assert scope.call("llm.complete", {"request": {"messages": []}}) == {"reply": "ok"}
        scope.call("context.capture", {"application": "app", "window_id": "window", "kind": "auto"})
        with pytest.raises(PermissionError):
            scope.call("llm.complete", {"request": {"messages": []}})
        grants.allow(owner, "egress.text", "app")
        assert scope.call("llm.complete", {"request": {"messages": []}}) == {"reply": "ok"}
        grants.revoke(owner, "context.ui", "app")
        grants.allow(owner, "context.ui", "app")
        with pytest.raises(PermissionError, match="revoked context"):
            scope.call("llm.complete", {"request": {"messages": []}})
        # 9.28: модуль отпустил данные окна — разговор продолжается, новое чтение — по гранту.
        grants.revoke(owner, "context.ui", "app")
        assert scope.call("context.release", {"application": "app"}) == {"released": True}
        assert scope.call("llm.complete", {"request": {"messages": []}}) == {"reply": "ok"}
        with pytest.raises(PermissionError):
            scope.call(
                "context.capture", {"application": "app", "window_id": "window", "kind": "ui"}
            )
        assert not list(tmp_path.rglob("*.json"))
    finally:
        scope.stop()


def test_released_owner_cannot_revive_old_snapshot_by_regrant() -> None:
    owner, grants, context, _, _ = setup()
    grants.allow(owner, "context.ui", "app")
    grants.allow(owner, "egress.text", "app")
    snapshot = context.capture(owner, "app", "window")
    grants.release_owner(owner)
    grants.allow(owner, "context.ui", "app")
    grants.allow(owner, "egress.text", "app")
    with pytest.raises(PermissionError):
        context.authorize_egress(owner, snapshot)
