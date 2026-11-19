"""8.6, экзамен 4 ТЗ: без интернета чат, голос и файловые инструменты агента работают.

Все соединения наружу запрещены ловушкой на `socket.connect` (разрешена только петля
127.0.0.1 — так сервис говорит с движком и оболочкой). Контроль в конце: ловушка правда
ловит внешнее соединение, иначе «ноль попыток» ничего бы не значил. Распознавание речи —
подменённое (настоящая модель whisper здесь не скачана); то, что настоящий whisper читает
только локальный файл модели, видно в `voice/worker.py` (путь к файлу, не имя для загрузки).
"""

import socket
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.files import FileLimits, register_file_tools
from ramvi_service.agent.tools import ToolRunner
from ramvi_service.agent.workspace import Permissions, Workspace
from ramvi_service.app import build
from ramvi_service.voice import models as vm
from test_agent_tools import ListJournal  # noqa: E402
from test_chat import ENGINE, TOKEN, Client, add_models, text_of  # noqa: E402
from test_voice import FakeMicrophone, FakeTranscriber, tone  # noqa: E402

LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in LOOPBACK:
            attempts.append(address)
            raise OSError(101, "Network is unreachable (offline test)")
        return real_connect(self, address)

    def no_dns(*args: Any, **kwargs: Any) -> Any:
        host = args[0] if args else kwargs.get("host")
        if host not in LOOPBACK and host is not None:
            attempts.append(("dns", host))
            raise socket.gaierror(socket.EAI_NONAME, "offline test")
        return real_getaddrinfo(*args, **kwargs)

    real_getaddrinfo = socket.getaddrinfo
    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    yield attempts


def test_chat_voice_and_agent_files_work_offline(offline: list[Any], tmp_path: Path) -> None:
    app, data = tmp_path / "app", tmp_path / "app" / "data"
    add_models(data, "model-a")
    mic = FakeMicrophone()
    server, engines = build(
        data,
        TOKEN,
        engine_command=ENGINE,
        app_dir=app,
        voice_options={
            "microphone": mic,
            "make_transcriber": lambda path: FakeTranscriber(path, "привет без сети"),
            "memory": lambda: (None, None),
            "has_nvidia": lambda: False,
        },
    )
    server.start()
    try:
        ui = Client(server.port)
        # Чат
        chat = ui.ok("POST", "/chats", {"model_id": "model-a"})["chat"]["id"]
        events = ui.events(f"/chats/{chat}/send", {"text": "вопрос офлайн"})
        assert events[-1]["finish"] == "stop" and "вопрос офлайн" in text_of(events, "text")
        # Голос: локальная модель лежит на диске — распознавание без сети, текст в поле ввода
        model = vm.MODELS[vm.DEFAULT_MODEL].path(app / "models")
        model.parent.mkdir(parents=True)
        model.write_bytes(b"x")
        vm.VAD.path(app / "models").write_bytes(b"v")
        assert ui.ok("POST", "/voice/start", {})["state"] == "recording"
        mic.say(tone(0.5))
        assert ui.ok("POST", "/voice/stop", {})["text"] == "привет без сети"
        # Экраны, которые не должны требовать сети
        for path in ("/status", "/models", "/catalog", "/downloads", "/voice", "/capabilities"):
            assert ui.request("GET", path)[0] == 200, path
    finally:
        server.close()
        engines.stop()

    # Файловые инструменты агента: чтение, поиск, предложение изменения с diff и применение
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_bytes(b"def main():\n    return 41\n")
    workspace = Workspace(project, Permissions(write=True))
    runner = ToolRunner()
    register_file_tools(runner, workspace, FileLimits())
    journal = ListJournal()
    assert "return 41" in runner.execute("read_file", {"path": "main.py"}, journal).content
    assert runner.execute("grep", {"pattern": "return"}, journal).ok
    proposal = runner.execute(
        "edit_file",
        {"path": "main.py", "old_text": "return 41", "new_text": "return 42"},
        journal,
    )
    assert proposal.ok, proposal.error
    ChangeStore(tmp_path / "changes", workspace).apply(proposal.data["proposal"])
    assert (project / "main.py").read_bytes() == b"def main():\n    return 42\n"

    assert offline == [], f"попытки выйти в сеть: {offline}"
    # Контроль: ловушка ловит внешнее соединение (адрес из документации, TEST-NET-1)
    with pytest.raises(OSError):
        urllib.request.urlopen("http://192.0.2.1/", timeout=2)
    assert offline
