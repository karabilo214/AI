"""8.4 (раздел 11 ТЗ): секреты маскируются, шаблоны моделей не исполняют код, без
действия пользователя приложение не ходит в сеть, значения по умолчанию безопасны."""

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import settings as st
from ramvi_service.app import build
from ramvi_service.chat import ChatError
from ramvi_service.redact import mask_secrets, mask_value
from test_chat import ENGINE, TOKEN, Client, add_models  # noqa: E402


@pytest.mark.parametrize(
    ("text", "hidden", "kept"),
    [
        ("Incorrect API key provided: sk-proj-abcdef1234567890XYZ.", "abcdef1234567890", "sk-p…"),
        ("Authorization: Bearer abcDEF123456789xyz", "DEF123456789", "Bearer abcD…"),
        ("client ramvi-Qx8_Zk2LmN0pQrStUvWxYz123 failed", "LmN0pQrStUvW", "ramv…"),
        ("url?api_key=SECRETVALUE42&x=1", "SECRETVALUE42", "api_key=SECR…"),
        ('{"token": "tok_9f8e7d6c5b4a"}', "9f8e7d6c5b4a", '"token": "tok_…'),
        ("password=hunter2hunter2", "hunter2hunter2", "password=hunt…"),
        ("key AIzaSyA-1234567890abcdefghijk", "1234567890abcdef", "AIza…"),
    ],
)
def test_mask_secrets(text: str, hidden: str, kept: str) -> None:
    masked = mask_secrets(text)
    assert hidden not in masked and kept in masked


def test_plain_text_is_left_alone() -> None:
    for text in ("файл модели не найден: C:/models/qwen.gguf", "status 401", "sha256 mismatch"):
        assert mask_secrets(text) == text
    assert mask_value({"a": ["x token=abcdefgh1"], "n": 3}) == {"a": ["x token=abcd…"], "n": 3}


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Any]:
    add_models(tmp_path, "model-a")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        yield server
    finally:
        server.close()
        engines.stop()


def test_errors_sent_to_the_interface_are_masked(
    service: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ответ стороннего API с ключом и внутренняя ошибка с токеном не уходят в интерфейс."""

    def leaky(method: str, path: str, body: dict[str, Any]) -> Any:
        if path == "/voice/check":
            raise ChatError(
                "voice.api_error",
                host="api.example",
                status=401,
                detail="Incorrect API key provided: sk-proj-abcdef1234567890XYZ",
            )
        raise RuntimeError("upstream said token=supersecretvalue123")

    monkeypatch.setattr(service, "route", leaky)
    ui = Client(service.port)
    status, body = ui.request("POST", "/voice/check", {})
    assert status == 400 and "abcdef1234567890" not in str(body)
    assert body["error"]["params"]["detail"] == "Incorrect API key provided: sk-p…"
    status, body = ui.request("GET", "/status")
    assert status == 500 and "supersecretvalue123" not in str(body)


def test_model_template_cannot_reach_python() -> None:
    """Шаблон чата из файла модели исполняется в песочнице Jinja (transformers)."""
    templates = pytest.importorskip("transformers.utils.chat_template_utils")
    SecurityError = pytest.importorskip("jinja2.exceptions").SecurityError  # noqa: N806

    evil = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    with pytest.raises(SecurityError):
        templates._compile_jinja_template(evil).render(messages=[])
    escape = "{{ cycler.__init__.__globals__.os.system('echo pwned') }}"
    with pytest.raises((SecurityError, Exception)) as error:
        templates._compile_jinja_template(escape).render(messages=[])
    assert "pwned" not in str(error.value)


def test_no_outbound_network_without_user_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Телеметрии нет: запуск и обычные экраны не открывают соединений наружу."""
    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            attempts.append(address)
            raise OSError("outbound connection blocked by test")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    add_models(tmp_path, "model-a")
    server, engines = build(tmp_path, TOKEN, engine_command=ENGINE)
    server.start()
    try:
        ui = Client(server.port)
        for path in ("/status", "/models", "/catalog", "/chats", "/downloads", "/capabilities"):
            ui.request("GET", path)
        chat = ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
        assert ui.events(f"/chats/{chat}/send", {"text": "привет"})[-1]["finish"] == "stop"
    finally:
        server.close()
        engines.stop()
    assert attempts == []
    # Контроль: ловушка правда ловит соединение наружу (иначе пустой список ничего не значит).
    import urllib.request

    with pytest.raises(OSError):
        urllib.request.urlopen("http://192.0.2.1/", timeout=2)  # TEST-NET-1, адрес из документации
    assert attempts and attempts[0][0] == "192.0.2.1"


def test_safe_defaults() -> None:
    defaults = st.Settings()
    assert defaults.network_access is False  # внешний API только на этом компьютере
    assert defaults.voice_backend == "local"  # речь не уходит в сеть, пока не выбран API
    assert defaults.voice_api_url == ""
    assert not any("key" in name for name in vars(defaults))  # ключей в settings.json нет
