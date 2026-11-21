"""7.4 (API-04): сеть только по явному включению, ключ с правом remote, разрешённые адреса."""

import http.client
import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import settings as st
from ramvi_service.api_server import ApiOptions, allowed, is_local
from ramvi_service.app import build
from ramvi_service.engines import EngineManager
from ramvi_service.ui_api import UIServer
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402


def lan_address() -> str | None:
    """Свой адрес в сети (не loopback) — чтобы подключиться «как с другого устройства»."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # пакеты не отправляются
            address = str(probe.getsockname()[0])
    except OSError:
        return None
    return None if address.startswith("127.") or address == "0.0.0.0" else address


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get(host: str, port: int, path: str, key: str | None = None) -> tuple[int, Any]:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    return response.status, body


@pytest.fixture
def svc(tmp_path: Path) -> Iterator[tuple[UIServer, EngineManager, Path]]:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    st.save(app_dir, st.Settings(api_port=free_port()))
    data = tmp_path / "data"
    add_models(data, "model-a")
    server, engines = build(data, TOKEN, engine_command=ENGINE, app_dir=app_dir)
    server.start()
    server.api.start()
    engines.ensure("model-a")
    try:
        yield server, engines, app_dir
    finally:
        server.close()
        engines.stop()


def test_defaults_keep_the_api_on_this_computer(
    svc: tuple[UIServer, EngineManager, Path],
) -> None:
    server, _, _ = svc
    assert st.Settings().network_access is False
    assert ApiOptions.from_settings(st.Settings()).host == "127.0.0.1"
    assert server.api._server is not None
    assert server.api._server.server_address[0] == "127.0.0.1"
    discovery = json.loads(server.api.discovery_file.read_text(encoding="utf-8"))
    assert discovery["network"] is False
    address = lan_address()
    if address is not None:  # приёмка API-04: с сетевого адреса сервис недоступен
        with pytest.raises(OSError):
            socket.create_connection((address, server.api.port), timeout=3).close()


def test_address_rules() -> None:
    default = ApiOptions.from_settings(st.Settings()).allow
    assert allowed("192.168.1.20", default) and allowed("10.1.2.3", default)
    assert allowed("172.20.0.5", default) and not allowed("172.32.0.5", default)
    assert not allowed("8.8.8.8", default)
    assert allowed("::ffff:192.168.0.7", default)  # IPv4 через IPv6-сокет
    assert allowed("8.8.8.8", ("0.0.0.0/0",))  # «откуда угодно» — выбор пользователя
    assert not allowed("not-an-ip", default) and not allowed("1.2.3.4", ("garbage",))
    assert is_local("127.0.0.1") and is_local("::1") and not is_local("192.168.1.1")


def test_remote_needs_network_mode_allowed_address_and_remote_key(
    svc: tuple[UIServer, EngineManager, Path],
) -> None:
    server, _, _ = svc
    api = server.api
    _, local_key = api.keys.create("local tool", {"chat", "models"})
    _, remote_key = api.keys.create("laptop", {"chat", "models", "remote"})
    lan = "192.168.1.20"
    assert not api.admit(lan)  # сеть выключена — никто, кроме этого компьютера
    api.options = ApiOptions(host="0.0.0.0", allow=("192.168.0.0/16",))
    assert api.admit(lan) and not api.admit("8.8.8.8") and api.admit("127.0.0.1")
    headers = {"Authorization": f"Bearer {local_key}"}
    with pytest.raises(Exception) as error:
        api.authorize(headers, "/v1/models", lan)
    assert getattr(error.value, "code", "") == "api.unauthorized"
    assert api.authorize(headers, "/v1/models", "127.0.0.1").name == "local tool"
    remote = {"Authorization": f"Bearer {remote_key}"}
    assert api.authorize(remote, "/v1/models", lan).name == "laptop"


def test_settings_apply_at_once(svc: tuple[UIServer, EngineManager, Path]) -> None:
    server, _, app_dir = svc
    ui = Client(server.port)
    new_port = free_port()
    result = ui.ok("POST", "/settings", {"network_access": True, "api_port": new_port})
    assert result["api_error"] is None
    assert server.api.port == new_port and server.api.options.network
    assert server.api._server is not None and server.api._server.server_address[0] == "0.0.0.0"
    assert get("127.0.0.1", new_port, "/")[0] == 200  # с этого компьютера всё как было
    bad = ui.request("POST", "/settings", {"network_allow": "192.168.1.0/24, nonsense"})
    assert bad[0] == 400 and bad[1]["error"]["params"]["field"] == "network_allow"
    ui.ok("POST", "/settings", {"network_access": False})
    assert server.api._server.server_address[0] == "127.0.0.1"
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        result = ui.ok("POST", "/settings", {"api_port": taken})
        assert result["api_error"] == "api.port_busy"
    ui.ok("POST", "/settings", {"api_port": free_port()})
    assert server.api._server is not None
    assert (
        json.loads((app_dir / "settings.json").read_text(encoding="utf-8"))["network_access"]
        is False
    )


def test_real_connection_from_the_network_address(
    svc: tuple[UIServer, EngineManager, Path],
) -> None:
    address = lan_address()
    if address is None:
        pytest.skip("no network address on this machine")
    server, _, _ = svc
    api = server.api
    _, local_key = api.keys.create("local tool", {"chat", "models"})
    _, remote_key = api.keys.create("laptop", {"chat", "models", "remote"})
    port = free_port()
    assert api.apply(ApiOptions(host="0.0.0.0", port=port, allow=(f"{address}/32",))) is None
    assert get(address, port, "/")[0] == 401  # из сети даже «жив ли» — только по ключу
    assert get(address, port, "/v1/models", local_key)[0] == 401
    status, body = get(address, port, "/v1/models", remote_key)
    assert status == 200 and body["data"][0]["id"] == "model-a"
    assert api.apply(ApiOptions(host="0.0.0.0", port=port, allow=("203.0.113.0/24",))) is None
    assert get(address, port, "/v1/models", remote_key)[0] == 403  # адрес не в списке
