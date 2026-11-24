"""Вход через ChatGPT: подключение chatgpt работает через Codex CLI (здесь — поддельный codex)."""

import json
import os
import sys
import threading
from pathlib import Path

import pytest
from ramvi_service import codex_cli
from ramvi_service.chat import ChatError
from ramvi_service.cloud import CloudConnections, model_id
from ramvi_service.codex_cli import TEXT_ONLY, CodexProvider, prompt
from ramvi_storage.repositories import Models
from ramvi_storage.schema import open_core

FAKE = r"""#!{python}
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
log = Path({log!r})
if args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT"); sys.exit(0)
if args[:2] == ["debug", "models"]:
    print(json.dumps({{"models": [{{"slug": "gpt-a", "visibility": "list"}}, {{"slug": "hidden", "visibility": "hide"}}]}}))
    sys.exit(0)
if args[:1] == ["exec"]:
    text = sys.stdin.read()
    instructions = Path(args[args.index("-c", args.index("-c") + 1) + 1].split("=", 1)[1].strip('"')).read_text()
    log.write_text(json.dumps({{"args": args, "stdin": text, "instructions": instructions}}))
    if "SLOW" in text:
        time.sleep(30)
    if "FAIL" in text:
        print(json.dumps({{"type": "turn.failed", "error": {{"message": "usage limit reached"}}}})); sys.exit(1)
    print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "Ответ: " + text[-10:]}}}}))
    print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 120, "output_tokens": 7}}}}))
    sys.exit(0)
sys.exit(0)
"""
needs_shell = pytest.mark.skipif(sys.platform == "win32", reason="поддельный codex — скрипт с #!")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "codex.log"
    script = tmp_path / "codex"
    script.write_text(FAKE.format(python=sys.executable, log=str(log)), "utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("RAMVI_CODEX", str(script))
    return log


def test_prompt_puts_system_instead_of_codex_prompt_and_refuses_tools_and_images() -> None:
    instructions, body = prompt(
        {
            "messages": [
                {"role": "system", "content": "Ты RAMVI"},
                {"role": "user", "content": "Привет"},
            ]
        }
    )
    assert instructions == "Ты RAMVI" + TEXT_ONLY and body == "Привет"
    _, history = prompt(
        {
            "messages": [
                {"role": "user", "content": "Раз"},
                {"role": "assistant", "content": "Два"},
                {"role": "user", "content": [{"type": "text", "text": "Три"}]},
            ]
        }
    )
    assert history.index("Раз") < history.index("Два") < history.index("Три")
    with pytest.raises(ChatError) as tools:
        prompt({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]})
    assert tools.value.code == "chatgpt.no_tools"
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    with pytest.raises(ChatError):
        prompt({"messages": [{"role": "user", "content": image}]})


def test_missing_codex_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAMVI_CODEX", str(tmp_path / "nope"))
    assert codex_cli.status() == {"installed": False, "logged_in": False}
    with pytest.raises(ChatError) as error:
        codex_cli.models()
    assert error.value.code == "chatgpt.codex_missing"


@needs_shell
def test_status_models_and_reply_through_codex(fake: Path) -> None:
    assert codex_cli.status() == {"installed": True, "logged_in": True}
    assert codex_cli.models() == ["gpt-a"]  # скрытые модели каталога не показываются
    used: list[tuple[object, str]] = []
    provider = CodexProvider("gpt-a", after=lambda usage, state: used.append((usage, state)))
    request = {
        "messages": [
            {"role": "system", "content": "Ты RAMVI"},
            {"role": "user", "content": "Как дела?"},
        ]
    }
    reply = provider.complete(request, threading.Event())
    assert reply["choices"][0]["message"]["content"] == "Ответ: Как дела?"
    assert reply["usage"] == {"prompt_tokens": 120, "completion_tokens": 7}
    assert used == [({"prompt_tokens": 120, "completion_tokens": 7}, "done")]
    seen = json.loads(fake.read_text())
    assert seen["stdin"] == "Как дела?" and seen["instructions"].startswith("Ты RAMVI")
    args = seen["args"]
    assert (
        args[args.index("-m") + 1] == "gpt-a" and args[args.index("--sandbox") + 1] == "read-only"
    )
    assert "--ephemeral" in args and 'approval_policy="never"' in args
    chunks = list(provider.stream(request, threading.Event()))
    assert chunks[0] == {"type": "text", "text": "Ответ: Как дела?"} and chunks[-1]["type"] == "end"


@needs_shell
def test_failure_and_cancel(fake: Path) -> None:
    states: list[str] = []
    provider = CodexProvider("gpt-a", after=lambda usage, state: states.append(state))
    with pytest.raises(ChatError) as error:
        provider.complete({"messages": [{"role": "user", "content": "FAIL"}]}, threading.Event())
    assert (
        error.value.code == "cloud.service_error" and "usage limit" in error.value.params["detail"]
    )
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(InterruptedError):
        provider.complete({"messages": [{"role": "user", "content": "SLOW"}]}, cancel)
    assert states == ["error", "cut"]


@needs_shell
def test_chatgpt_connection_needs_no_key_and_lists_subscription_models(
    fake: Path, tmp_path: Path
) -> None:
    db = open_core(tmp_path)
    try:
        store = CloudConnections(tmp_path, None, Models(db))
        saved = store.save({"kind": "chatgpt", "name": "ChatGPT"})
        assert saved["url"] == "https://chatgpt.com" and saved["has_key"] is False
        assert store.view()["connections"][0]["has_key"] is True  # ключ не нужен
        with pytest.raises(ChatError):
            store.save({"kind": "chatgpt", "name": "ChatGPT", "key": "sk-x"})
        assert store.remote_models({"id": saved["id"]}) == ["gpt-a"]
        store.enable(saved["id"], "gpt-a", True)
        provider = store.provider(store.models.get(model_id(saved["id"], "gpt-a")))
        assert isinstance(provider, CodexProvider)
        reply = provider.complete(
            {"messages": [{"role": "user", "content": "Привет"}]}, threading.Event()
        )
        assert reply["choices"][0]["message"]["content"].startswith("Ответ")
        assert store.spending()["recent"][0]["completion_tokens"] == 7  # учёт ответов как у облака
    finally:
        db.close()
    assert os.environ["RAMVI_CODEX"]
