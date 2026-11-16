"""Контракт модулей: независимые workers, namespaces, события, ресурсы и контекст."""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, Manifest, Owner, ProviderCapabilities
from ramvi_service.events import EventBus, EventSpec
from ramvi_service.modules.host import ModuleHost
from ramvi_service.resources import ResourceBroker, SharedLLM
from ramvi_service.scoped import MemorySecrets, ScopedData, ScopedSecrets
from ramvi_service.services import ScopedServices, ServiceClient, ServiceGateway


@pytest.fixture
def identities() -> tuple[Authority, Owner, Owner]:
    authority = Authority()
    return (
        authority,
        authority.issue("chat", "user", "a"),
        authority.issue("companion", "user", "b"),
    )


def test_manifest_validation_and_lazy_import(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path)
    host.register(Manifest("optional", "1.0.0", "not_installed_character.factory:create"))
    assert host.status()["optional"]["state"] == "registered"
    host.dispose("optional")
    assert host.status()["optional"]["worker"] is None
    host.close()
    with pytest.raises(ValueError):
        Manifest("../escape", "1.0.0", "a:create")
    with pytest.raises(ValueError):
        Manifest("module", "1.0.0", "a:create", api_version="2.0")
    with pytest.raises(ValueError):
        Manifest("module", "1.0.0", "a:create", required=("module",))


def test_capabilities_and_dependency_diagnostics(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path)
    host.register(Manifest("one", "1.0.0", "a:create", required=("two",)))
    with pytest.raises(RuntimeError, match="dependency"):
        host.initialize("one")
    with pytest.raises(ValueError, match="cycle"):
        host.register(Manifest("two", "1.0.0", "a:create", required=("one",)))
    host.register(Manifest("needs_audio", "1.0.0", "a:create", capabilities=("audio",)))
    with pytest.raises(NotImplementedError):
        host.initialize("needs_audio")
    assert host.status()["needs_audio"]["worker"] is None
    host.close()


def test_two_actual_workers_lifecycle_checkpoint_and_scoped_gateway(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=3)
    permissions = ("settings", "events", "resources")
    for name in ("chat", "companion"):
        host.register(
            Manifest(name, "1.0.0", "ramvi.tests.module_fixture:create", permissions=permissions),
            approved_permissions=permissions,
        )
    try:
        host.activate("chat")
        host.activate("companion")
        host.events.register_schema("chat", "tick", EventSpec({"value": int}))
        owner = host._entry("chat").owner
        assert owner is not None
        response = host.events.begin(owner)
        subscription = host.events.subscribe(owner)
        assert host.invoke("chat", "event", {"response": response}) == {"delivered": True}
        event = host.events.poll(owner, subscription)
        assert event is not None and event["module_id"] == "chat"
        pid = host.status()["chat"]["worker"]["pid"]
        assert host.invoke("chat", "settings", {"value": "first"}) == {
            "value": {"module": "chat", "value": "first"}
        }
        assert (
            host.invoke("companion", "settings", {"value": "second"})["value"]["value"] == "second"
        )
        host.suspend("companion")
        assert host.status()["companion"]["worker"] is None
        host.resume("companion")
        assert host.invoke("companion", "value") == {"value": 2}
        host.deactivate("companion")
        host.deactivate("companion")
        assert host.status()["chat"]["worker"]["pid"] == pid
        assert host.invoke("chat", "value") == {"value": 1}
        assert host.resources.status()["leases"] == 0
        host.activate("companion")
        assert host.invoke("companion", "value") == {"value": 3}
    finally:
        host.close()
    assert not any(thread.name == "ramvi-services" for thread in threading.enumerate())


def test_worker_crash_and_timeout_do_not_break_peer(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=0.3)
    for name in ("first", "second"):
        host.register(Manifest(name, "1.0.0", "ramvi.tests.module_fixture:create"))
    try:
        host.activate("first")
        host.activate("second")
        with pytest.raises((RuntimeError, TimeoutError)):
            host.invoke("first", "crash")
        assert host.status()["first"]["state"] == "error"
        assert host.invoke("second", "value") == {"value": 1}
        host.activate("first")
        host.timeout = 0.2
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            host.invoke("first", "hang")
        assert time.monotonic() - started < 5
        assert host.invoke("second", "value") == {"value": 1}
    finally:
        host.close()


def test_namespace_settings_secrets_and_transactional_migrations(tmp_path: Path) -> None:
    one, two = ScopedData(tmp_path, "one"), ScopedData(tmp_path, "two")
    one.write_json("state.json", {"value": 1})
    two.write_json("state.json", {"value": 2})

    def fail(stage: ScopedData) -> None:
        stage.write_json("state.json", {"value": 99})
        raise ValueError("failed migration")

    with pytest.raises(ValueError):
        one.migrate(2, {1: fail})
    assert one.read_json("state.json") == {"value": 1}
    one.migrate(2, {1: lambda stage: stage.write_json("state.json", {"value": 3})})
    assert one.schema_version == 2
    assert two.read_json("state.json") == {"value": 2}
    for path in ("../two/state.json", "/absolute", "C:\\outside", "a/../../other", "."):
        with pytest.raises(PermissionError):
            one.path(path)
    secret = MemorySecrets()
    first, second = ScopedSecrets(secret, "one"), ScopedSecrets(secret, "two")
    first.set("api_key", "one")
    second.set("api_key", "two")
    first.delete("api_key")
    assert first.get("api_key") is None and second.get("api_key") == "two"
    one.reset()
    assert two.read_json("state.json") == {"value": 2}


def test_scope_rejects_symlink_escape(tmp_path: Path) -> None:
    data = ScopedData(tmp_path / "modules", "one")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (data.root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit test symlink")
    with pytest.raises(PermissionError):
        data.write_json("link/secret.json", {})


def test_event_schema_identity_bounds_and_late_drop(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, one, two = identities
    bus = EventBus(authority)
    bus.register_schema("chat", "tick", EventSpec({"value": int}))
    bus.register_schema("companion", "tick", EventSpec({"value": int}))
    subscription = bus.subscribe(one, size=1)
    second = bus.subscribe(two)
    response = bus.begin(one)
    with pytest.raises(ValueError):
        bus.emit(one, response, "tick", {"value": "7"})
    with pytest.raises(ValueError):
        bus.emit(one, response, "tick", {"value": 7, "module_id": "companion"})
    assert bus.emit(one, response, "tick", {"value": 7})
    assert bus.poll(two, second) is None
    with pytest.raises(PermissionError):
        bus.poll(two, subscription)
    event = bus.poll(one, subscription)
    assert event is not None and event["module_id"] == "chat" and event["session_id"] == "a"
    assert event["version"] == 1 and event["sequence"] == 1
    assert bus.emit(one, response, "tick", {"value": 8})
    new_response = bus.begin(one)
    assert not bus.emit(one, response, "tick", {"value": 9})
    assert bus.poll(one, subscription) is None
    assert bus.emit(one, new_response, "tick", {"value": 10})
    bus.cancel(one, new_response)
    assert bus.poll(one, subscription) is None
    bus.release_owner(one)


def test_resource_fifo_quotas_cancel_and_cleanup(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, one, two = identities
    broker = ResourceBroker(authority, {"cpu_slots": 1, "ram_bytes": 100})
    broker.quota(one, {"cpu_slots": 1, "ram_bytes": 50})
    with pytest.raises(PermissionError):
        broker.acquire(one, {"ram_bytes": 51})
    first = broker.acquire(one, {"cpu_slots": 1})
    with pytest.raises(PermissionError):
        broker.release(two, first)
    entered, order = threading.Event(), []

    def acquire() -> None:
        entered.set()
        lease = broker.acquire(two, {"cpu_slots": 1}, timeout=2)
        order.append("two")
        broker.release(two, lease)

    thread = threading.Thread(target=acquire)
    thread.start()
    assert entered.wait(1)
    broker.release(one, first)
    thread.join(3)
    assert not thread.is_alive() and order == ["two"]
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        broker.acquire(two, {"cpu_slots": 1}, cancel=cancel)
    broker.release_owner(one)
    assert broker.status()["leases"] == broker.status()["queued"] == 0


def test_shared_llm_does_not_unload_other_active_owner(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, chat, character = identities
    entered, finish = threading.Event(), threading.Event()

    class Provider:
        capabilities = ProviderCapabilities()
        cancelled = 0

        def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
            entered.set()
            assert finish.wait(3)
            return {"text": "answer"}

        def cancel(self) -> None:
            self.cancelled += 1

        def unload(self) -> None:
            pass

    unloaded = []
    provider = Provider()
    model = SharedLLM(authority, provider, lambda: unloaded.append(True))
    model.retain(chat)
    model.retain(character)
    results = []
    thread = threading.Thread(target=lambda: results.append(model.complete(chat, {})))
    thread.start()
    assert entered.wait(1)
    model.release_owner(character)
    assert provider.cancelled == 0 and not unloaded
    finish.set()
    thread.join(3)
    assert results == [{"text": "answer"}]
    model.release_owner(chat)
    assert unloaded == [True]


def test_context_independent_permissions_revoke_and_provenance(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, one, two = identities
    grants = GrantStore(authority)

    class Adapter:
        def capture(
            self, app: str, window: str, kind: str, cancel: threading.Event
        ) -> dict[str, Any]:
            return {"text": "selected"} if kind == "ui" else {"image": "png-base64"}

    context = ContextProvider(authority, grants, Adapter(), min_interval=0)
    with pytest.raises(PermissionError):
        context.capture(one, "editor", "window")
    grants.allow(one, "context.ui", "editor")
    snapshot = context.capture(one, "editor", "window")
    with pytest.raises(PermissionError):
        context.authorize_egress(one, snapshot)
    grants.allow(one, "egress.text", "editor")
    assert context.authorize_egress(one, snapshot)["text"] == "selected"
    with pytest.raises(PermissionError):
        context.authorize_egress(two, snapshot)
    with pytest.raises(PermissionError):
        context.authorize_egress(one, replace(snapshot, text="forged"))
    grants.allow(one, "context.screen", "editor")
    with pytest.raises(NotImplementedError):
        context.capture(one, "editor", "window", kind="screen", capabilities=ProviderCapabilities())
    image = context.capture(
        one, "editor", "window", kind="screen", capabilities=ProviderCapabilities(image_input=True)
    )
    with pytest.raises(PermissionError):
        context.authorize_egress(one, image)
    grants.allow(one, "egress.image", "editor")
    assert context.authorize_egress(one, image)["image"]
    grants.revoke(one, "context.screen", "editor")
    with pytest.raises(PermissionError):
        context.authorize_egress(one, image)
    with pytest.raises(PermissionError):
        grants.require(one, "agent.control", "editor")


def test_update_failure_rolls_back_only_its_namespace(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=2)
    permissions = ("settings",)
    good = Manifest("one", "1.0.0", "ramvi.tests.module_fixture:create", permissions=permissions)
    host.register(good, approved_permissions=permissions)
    host.register(
        Manifest("two", "1.0.0", "ramvi.tests.module_fixture:create", permissions=permissions),
        approved_permissions=permissions,
    )
    try:
        host.activate("one")
        host.activate("two")
        host.invoke("one", "settings", {"value": "keep"})
        second_pid = host.status()["two"]["worker"]["pid"]
        with pytest.raises((RuntimeError, TimeoutError)):
            host.update(
                "one",
                Manifest(
                    "one", "2.0.0", "missing_module:create", permissions=permissions, data_schema=2
                ),
                migrations={1: lambda data: data.write_json("settings.json", {"value": "bad"})},
            )
        assert host.status()["one"]["version"] == "1.0.0"
        assert host.status()["one"]["state"] == "active"
        assert ScopedData(host.root, "one").read_json("settings.json")["value"] == "keep"
        assert host.status()["two"]["worker"]["pid"] == second_pid
        host.uninstall("one")
        assert ScopedData(host.root, "one").read_json("settings.json")["value"] == "keep"
    finally:
        host.close()


def test_import_does_not_load_optional_packages(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    script = """
import sys
from ramvi_service.modules.host import ModuleHost
from ramvi_service.contracts.modules import Manifest
assert not any(name in sys.modules for name in ['torch','numpy','triton','ramvi_character','sounddevice','pygame','human_brain'])
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(project / f"ramvi/{part}") for part in ("service", "engine", "storage")]
            ),
        },
        timeout=10,
        check=True,
        capture_output=True,
    )


def test_scope_gateway_owner_is_authenticated_not_payload(
    tmp_path: Path, identities: tuple[Authority, Owner, Owner]
) -> None:
    authority, one, two = identities
    grants = GrantStore(authority)
    events = EventBus(authority)
    resources = ResourceBroker(authority, {"cpu_slots": 1})
    gateway = ServiceGateway(authority)
    scopes = []
    for owner in (one, two):
        grants.allow(owner, "settings")
        scope = ScopedServices(
            owner,
            authority,
            grants,
            ScopedData(tmp_path, owner.module_id),
            ScopedSecrets(MemorySecrets(), owner.module_id),
            resources,
            events,
        )
        gateway.bind(scope)
        scopes.append(scope)
    url = gateway.start()
    try:
        client = ServiceClient(url, one.token)
        client.call("settings.set", {"value": {"only": "one"}})
        assert ServiceClient(url, two.token).call("settings.get", {}) == {"value": {}}
        with pytest.raises(PermissionError):
            client.call("settings.set", {"module_id": "companion", "value": {}})
        with pytest.raises(PermissionError):
            ServiceClient(url, "untrusted-model-output").call("settings.get", {})
        authority.revoke(one)
        with pytest.raises(PermissionError):
            client.call("settings.get", {})
    finally:
        gateway.close()


def test_immediate_context_revoke_signals_capture_and_transmission(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, one, _ = identities
    grants = GrantStore(authority)
    grants.allow(one, "context.ui", "editor")
    grants.allow(one, "egress.text", "editor")

    class Adapter:
        def capture(
            self, app: str, window: str, kind: str, cancel: threading.Event
        ) -> dict[str, Any]:
            return {"text": "selected"}

    context = ContextProvider(authority, grants, Adapter(), min_interval=0)
    snapshot = context.capture(one, "editor", "window")
    entered = threading.Event()
    outcomes = []

    def sender(data: dict[str, Any], cancel: threading.Event) -> None:
        entered.set()
        assert cancel.wait(2)

    def transmit() -> None:
        try:
            context.transmit(one, snapshot, sender)
        except PermissionError:
            outcomes.append("cancelled")

    worker = threading.Thread(target=transmit)
    worker.start()
    assert entered.wait(1)
    grants.revoke(one, "egress.text", "editor")
    worker.join(3)
    assert not worker.is_alive() and outcomes == ["cancelled"]


def test_shared_resource_estimate_is_used(identities: tuple[Authority, Owner, Owner]) -> None:
    from ramvi_engine.planning.resources import MemoryUsage, ResourceEstimate

    authority, _, _ = identities
    estimate = ResourceEstimate(weights=MemoryUsage(200, 100, 50), reserve=MemoryUsage(10, 20))
    broker = ResourceBroker.with_model_estimate(
        authority, ram=1000, vram=2000, estimate=estimate, cpu_slots=3
    )
    assert broker.capacity == {"ram_bytes": 830, "vram_bytes": 1790, "cpu_slots": 3}
    with pytest.raises(ValueError):
        ResourceBroker.with_model_estimate(authority, ram=10, vram=20, estimate=estimate)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:123@evil.invalid",
        "http://127.0.0.1:1/path",
        "http://127.0.0.1:1?next=evil",
        "http://127.0.0.1:1#fragment",
        "https://127.0.0.1:1",
        "http://evil.invalid:12",
    ],
)
def test_local_transport_rejects_credential_forwarding_urls(url: str) -> None:
    from ramvi_service.providers import LocalEngineProvider

    with pytest.raises(ValueError):
        LocalEngineProvider(url, "model")
    with pytest.raises(ValueError):
        ServiceClient(url, "secret")


def test_provider_capabilities_are_conservative() -> None:
    from ramvi_service.providers import LocalEngineProvider

    provider = LocalEngineProvider("http://127.0.0.1:12345/", "model")
    assert provider.capabilities.text
    assert not provider.capabilities.json_schema
    assert not provider.capabilities.tools
    assert not provider.capabilities.image_input
    assert not provider.capabilities.external


def test_event_queue_is_bounded(identities: tuple[Authority, Owner, Owner]) -> None:
    authority, one, _ = identities
    bus = EventBus(authority)
    bus.register_schema("chat", "tick", EventSpec({"value": int}))
    subscription = bus.subscribe(one, size=1)
    response = bus.begin(one)
    bus.emit(one, response, "tick", {"value": 1})
    bus.emit(one, response, "tick", {"value": 2})
    assert bus._subscriptions[subscription].dropped == 1
    assert bus.poll(one, subscription)["payload"] == {"value": 1}  # type: ignore[index]
    assert bus.poll(one, subscription) is None


def test_authority_rejects_forged_module_identity(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, one, _ = identities
    with pytest.raises(PermissionError):
        authority.require(replace(one, module_id="companion"))
    authority.revoke(one)
    with pytest.raises(PermissionError):
        authority.require(one)


def test_local_provider_targets_its_request_on_cancel() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ramvi_service.providers import LocalEngineProvider

    entered, release = threading.Event(), threading.Event()
    sent_ids, cancelled_ids = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result: dict[str, Any]
            if self.path == "/v1/chat/completions":
                sent_ids.append(self.headers.get("X-Request-ID"))
                entered.set()
                assert release.wait(3)
                result = {"answer": "done"}
            else:
                cancelled_ids.append(request["request_id"])
                release.set()
                result = {"requested": True}
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    provider = LocalEngineProvider(f"http://127.0.0.1:{server.server_port}", "model", timeout=3)
    cancel = threading.Event()
    outcomes = []

    def complete() -> None:
        try:
            provider.complete({"messages": []}, cancel)
        except InterruptedError:
            outcomes.append("cancelled")
        except Exception as error:
            outcomes.append(type(error).__name__ + ": " + str(error))

    worker = threading.Thread(target=complete)
    worker.start()
    try:
        assert entered.wait(2)
        cancel.set()
        provider.cancel()
        worker.join(4)
        assert not worker.is_alive()
        assert outcomes == ["cancelled"]
        assert sent_ids == cancelled_ids and sent_ids[0]
        provider.cancel()  # idle cancel must not hit a peer's request
        assert len(cancelled_ids) == 1
    finally:
        release.set()
        worker.join(4)
        server.shutdown()
        server.server_close()
        serving.join(4)
        assert not serving.is_alive()


def test_local_transport_does_not_follow_redirects() -> None:
    import urllib.error
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ramvi_service.providers import LocalEngineProvider

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            # Consume the request before closing: Windows otherwise resets the
            # connection before the client can parse the redirect response.
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://external.invalid/leak")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        provider = LocalEngineProvider(
            f"http://127.0.0.1:{server.server_port}", "model", api_key="secret", timeout=2
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            provider.complete({}, threading.Event())
        assert error.value.code == 302
        assert requests == ["/v1/chat/completions"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(4)
        assert not thread.is_alive()


def test_context_and_schema_versions_reject_malformed_values(
    tmp_path: Path, identities: tuple[Authority, Owner, Owner]
) -> None:
    authority, owner, _ = identities
    grants = GrantStore(authority)
    grants.allow(owner, "context.ui", "editor")

    class Adapter:
        def capture(
            self, app: str, window: str, kind: str, cancel: threading.Event
        ) -> dict[str, Any]:
            return {"text": 123}

    provider = ContextProvider(authority, grants, Adapter(), min_interval=0)
    with pytest.raises(ValueError):
        provider.capture(owner, "editor", "window")
    data = ScopedData(tmp_path, "module")
    with pytest.raises(ValueError):
        data.migrate(True, {})
    data.write_json("schema.json", [])
    with pytest.raises(ValueError):
        _ = data.schema_version


def test_external_egress_revocation_cancels_active_provider(
    tmp_path: Path, identities: tuple[Authority, Owner, Owner]
) -> None:
    authority, owner, _ = identities
    grants = GrantStore(authority)
    for permission in ("llm", "egress.text"):
        grants.allow(owner, permission)
    entered = threading.Event()

    class External:
        capabilities = ProviderCapabilities(external=True)

        def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
            entered.set()
            assert cancel.wait(3)
            return {"text": "late answer"}

        def cancel(self) -> None:
            pass

        def unload(self) -> None:
            pass

    model = SharedLLM(authority, External())
    scope = ScopedServices(
        owner,
        authority,
        grants,
        ScopedData(tmp_path, owner.module_id),
        ScopedSecrets(MemorySecrets(), owner.module_id),
        ResourceBroker(authority, {}),
        EventBus(authority),
        model,
    )
    scope.active = True
    outcomes = []

    def call() -> None:
        try:
            scope.call("llm.complete", {"request": {"messages": []}})
        except InterruptedError:
            outcomes.append("cancelled")

    worker = threading.Thread(target=call)
    worker.start()
    try:
        assert entered.wait(1)
        grants.revoke(owner, "egress.text")
        worker.join(4)
        assert not worker.is_alive() and outcomes == ["cancelled"]
    finally:
        grants.revoke(owner, "egress.text")
        worker.join(4)
        scope.stop()


def test_gateway_closes_partial_header_connections() -> None:
    import socket

    gateway = ServiceGateway(Authority())
    url = gateway.start()
    from urllib.parse import urlsplit

    port = urlsplit(url).port
    assert port is not None
    connection = socket.create_connection(("127.0.0.1", port), timeout=2)
    connection.sendall(b"POST / HTTP/1.1\r\nHost:")
    try:
        gateway.close()
        try:
            received = connection.recv(1)
        except ConnectionResetError:
            received = b""
        assert received == b""
    finally:
        connection.close()
    assert not any(thread.name == "ramvi-services" for thread in threading.enumerate())


def test_observed_worker_exit_releases_owner_without_silent_restart(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=3)
    host.register(Manifest("one", "1.0.0", "ramvi.tests.module_fixture:create"))
    try:
        host.activate("one")
        entry = host._entry("one")
        assert entry.owner is not None and entry.worker is not None
        owner = entry.owner
        entry.worker.stop()  # observed independent crash/exit, not a lifecycle RPC
        assert host.status()["one"]["state"] == "error"
        assert host.status()["one"]["worker"] is None
        with pytest.raises(PermissionError):
            host.authority.require(owner)
        with pytest.raises(RuntimeError):
            host.invoke("one", "value")
        host.activate("one")
        assert host.invoke("one", "value") == {"value": 1}
    finally:
        host.close()


def test_application_grants_are_explicit_and_survive_resume(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path, timeout=3)
    permission = "context.ui"
    host.register(
        Manifest("one", "1.0.0", "ramvi.tests.module_fixture:create", permissions=(permission,)),
        approved_permissions=(permission,),
    )
    try:
        host.activate("one")
        owner = host._entry("one").owner
        assert owner is not None
        with pytest.raises(PermissionError):
            host.grants.require(owner, permission, "editor")
        host.approve_application("one", permission, "editor")
        host.grants.require(owner, permission, "editor")
        host.suspend("one")
        host.resume("one")
        resumed = host._entry("one").owner
        assert resumed is not None and resumed != owner
        host.grants.require(resumed, permission, "editor")
        host.revoke_application("one", permission, "editor")
        with pytest.raises(PermissionError):
            host.grants.require(resumed, permission, "editor")
    finally:
        host.close()


def test_releasing_queued_llm_owner_does_not_wait_for_other_active_chat(
    identities: tuple[Authority, Owner, Owner],
) -> None:
    authority, chat, companion = identities
    entered, release = threading.Event(), threading.Event()

    class Provider:
        capabilities = ProviderCapabilities()

        def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
            entered.set()
            assert release.wait(4)
            return {"text": "chat"}

        def cancel(self) -> None:
            raise AssertionError("queued companion must not cancel active chat")

        def unload(self) -> None:
            pass

    model = SharedLLM(authority, Provider())
    model.retain(chat)
    model.retain(companion)
    chat_result, companion_result = [], []
    first = threading.Thread(target=lambda: chat_result.append(model.complete(chat, {})))

    def queued() -> None:
        try:
            model.complete(companion, {}, timeout=3)
        except InterruptedError:
            companion_result.append("cancelled")

    second = threading.Thread(target=queued)
    first.start()
    assert entered.wait(1)
    second.start()
    try:
        end = time.monotonic() + 1
        while model.status()["queue"]["queued"] != 1 and time.monotonic() < end:
            threading.Event().wait(0.001)
        assert model.status()["queue"]["queued"] == 1
        model.release_owner(companion)
        second.join(1)
        assert not second.is_alive() and companion_result == ["cancelled"]
        assert model.status()["active"]
    finally:
        release.set()
        first.join(5)
        second.join(5)
        model.release_owner(chat)
    assert chat_result == [{"text": "chat"}]
    assert not first.is_alive() and not second.is_alive()
