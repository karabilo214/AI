"""Text runtime exercises cancellation, fallback and model-independent persistence."""

import copy
import json
import threading
import time
from typing import Any

import pytest
from ramvi_character.context import Counter
from ramvi_character.module import CharacterModule
from test_character_response import records


class Services:
    def __init__(self, output: str | None = None, streaming: bool = True) -> None:
        self.data: dict[str, Any] = {}
        self.requests: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.output = output or "\n".join(json.dumps(r) for r in records())
        self.streaming = streaming
        self.sent = False
        self.block = threading.Event()
        self.block.set()

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "data.get":
            return {"value": copy.deepcopy(self.data.get(payload["path"]))}
        if method == "data.set":
            self.data[payload["path"]] = copy.deepcopy(payload["value"])
            return {"ok": True}
        if method == "llm.capabilities":
            return {"available": True, "capabilities": {"streaming": self.streaming}}
        if method == "llm.stream":
            self.requests.append(payload["request"])
            self.sent = False
            return {"stream_id": "stream"}
        if method == "llm.cancel":
            self.cancelled.append(payload["stream_id"])
            return {"ok": True}
        if method == "llm.read":
            self.block.wait(timeout=2)
            if self.sent:
                return {"done": True, "chunks": []}
            self.sent = True
            return {
                "done": False,
                "chunks": [
                    {"type": "text", "text": self.output},
                    {"type": "finish", "finish_reason": "stop"},
                ],
            }
        if method == "llm.complete":
            self.requests.append(payload["request"])
            self.block.wait(timeout=2)
            return {"choices": [{"message": {"content": self.output}, "finish_reason": "stop"}]}
        raise AssertionError(method)


def active(services: Services) -> CharacterModule:
    m = CharacterModule()
    m.initialize({"services": services})
    m.activate()
    # Deterministic test tokenizer, no dependency on model weights.
    m._dialogue.counter = Counter(
        lambda s: len(s) // 10, lambda ms: sum(len(m["content"]) // 10 for m in ms), "test"
    )
    return m


def finish(m: CharacterModule) -> dict[str, Any]:
    assert m._dialogue.thread is not None
    m._dialogue.thread.join(timeout=4)
    assert not m._dialogue.thread.is_alive()
    result = m.dialogue()
    result["job"] = m.dialogue("poll")["job"]
    return result


def test_streamed_text_without_renderer_and_restart_model_switch() -> None:
    svc = Services()
    m = active(svc)
    m.select_profile("friendly")
    m.set_boundary("teasing", False)
    response = m.dialogue("start", {"text": "Привет", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed"
    assert result["history"][0]["assistant"] == "Привет"
    assert result["job"]["events"][1]["response_id"] == response["response_id"]
    assert not m.status()["devices_open"] and m.status()["presentation"] is None
    checkpoint = m.checkpoint()
    m.dispose()
    restored = CharacterModule()
    restored.initialize({"services": svc, "checkpoint": checkpoint})
    restored.activate()
    assert restored.dialogue()["history"] == result["history"]
    assert restored.status()["personality"]["base"]["id"] == "friendly"
    assert not restored.status()["personality"]["boundaries"]["teasing"]
    restored._dialogue.counter = m._dialogue.counter
    restored.dialogue("start", {"text": "Дальше", "model": "two"})
    result = finish(restored)
    assert result["job"]["state"] == "completed" and len(result["history"]) == 2
    assert svc.requests[-1]["ramvi_model_id"] == "two"
    assert any("Привет" in msg["content"] for msg in svc.requests[-1]["messages"])
    restored.dispose()


@pytest.mark.parametrize("streaming", [True, False])
def test_cancel_drops_late_provider_output(streaming: bool) -> None:
    svc = Services(streaming=streaming)
    if not streaming:
        svc.output = json.dumps(records())
    svc.block.clear()
    m = active(svc)
    m.dialogue("start", {"text": "Привет", "model": "one"})
    deadline = time.monotonic() + 2
    while not svc.requests and time.monotonic() < deadline:
        time.sleep(0.005)
    m.dialogue("cancel")
    svc.block.set()
    result = finish(m)
    assert result["job"]["state"] == "cancelled" and result["job"]["events"] == []
    assert result["history"][0]["status"] == "cancelled"
    m.dispose()


def test_invalid_partial_does_not_fallback_or_publish_memory() -> None:
    svc = Services(output="\n".join(json.dumps(r) for r in records()[:2]) + '\n{"type":"bad"}\n')
    m = active(svc)
    m.dialogue("start", {"text": "Ответ", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "error" and len(svc.requests) == 1
    assert all(e["type"] != "end" for e in result["job"]["events"])
    assert "fact" not in svc.data
    m.dispose()


def test_nonstream_and_candidate_remains_unconfirmed() -> None:
    r = records()
    r[-1]["memory_candidates"] = [
        {"id": "m", "kind": "fact", "text": "proposal", "confidence": 1, "evidence": ["current"]}
    ]
    svc = Services(output=json.dumps(r), streaming=False)
    m = active(svc)
    m.dialogue("start", {"text": "Тест", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed"
    assert result["job"]["events"][-1]["memory_candidates"][0]["status"] == "candidate"
    assert "proposal" not in json.dumps(svc.data)
    m.dispose()


def test_policy_change_invalidates_inflight_response() -> None:
    svc = Services()
    svc.block.clear()
    m = active(svc)
    m.dialogue("start", {"text": "Тест", "model": "one"})
    m.set_boundary("teasing", False)
    svc.block.set()
    result = finish(m)
    assert result["job"]["state"] == "cancelled" and not result["job"]["events"]
    # Bad owner controls return errors instead of crashing the host worker.
    assert "error" in m.dialogue("profile", {"id": "unknown"})
    assert m.status()["state"] == "active"
    m.dispose()


def test_bad_structure_gets_one_plain_text_fallback() -> None:
    svc = Services(output="not JSON")
    original = svc.call

    def call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "llm.stream" and svc.requests:
            svc.output = "Обычный ответ без управляющих действий."
        return original(method, payload)

    svc.call = call  # type: ignore[method-assign]
    m = active(svc)
    m.dialogue("start", {"text": "Привет", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed"
    assert result["job"]["fallback"] and len(svc.requests) == 2
    segments = [e for e in result["job"]["events"] if e["type"] == "segment"]
    assert len(segments) == 1 and segments[0]["animation"] == "none"
    assert not result["job"]["events"][-1]["memory_candidates"]
    m.dispose()


def test_real_worker_scoped_llm_and_authenticated_ui_route(tmp_path: Any, monkeypatch: Any) -> None:
    # CI remains lightweight; this integration also runs where Engine's tokenizer is installed.
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from pathlib import Path
    from typing import cast

    from ramvi_service.contracts.modules import Authority, ProviderCapabilities
    from ramvi_service.modules.host import ModuleHost
    from ramvi_service.resources import SharedLLM
    from ramvi_service.sections import SectionRegistry
    from ramvi_service.ui_api import UIServer
    from test_chat import TOKEN, Client

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = '{% for message in messages %}{{ message["role"] + ":" + message["content"] + "\\n" }}{% endfor %}assistant:'
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    monkeypatch.setenv("RAMVI_TOKENIZER", str(tmp_path / "tokenizer"))

    class Provider:
        capabilities = ProviderCapabilities(streaming=True)

        def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
            raise AssertionError("stream expected")

        def stream(self, request: dict[str, Any], cancel: threading.Event) -> Any:
            for record in records():
                if cancel.is_set():
                    raise InterruptedError("cancelled")
                yield {"type": "text", "text": json.dumps(record) + "\n"}
            yield {"type": "finish", "finish_reason": "stop"}

        def cancel(self) -> None:
            pass

        def unload(self) -> None:
            pass

    authority = Authority()
    host = ModuleHost(tmp_path / "data", llm=SharedLLM(authority, Provider()))
    registry = SectionRegistry(host)
    registry.discover(Path(__file__).resolve().parents[1] / "modules")
    server = UIServer(cast(Any, None), TOKEN, sections=registry).start()
    try:
        client = Client(server.port)
        assert client.ok("POST", "/sections/character/open")["section"]["state"] == "active"
        path = "/sections/character/commands/dialogue"
        assert (
            Client(server.port, "wrong-token").request("POST", path, {"operation": "status"})[0]
            == 401
        )
        client.ok("POST", path, {"operation": "profile", "arguments": {"id": "friendly"}})
        result = client.ok(
            "POST", path, {"operation": "start", "arguments": {"text": "Hello", "model": "first"}}
        )
        assert "response_id" in result, result
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            packet = client.ok("POST", path, {"operation": "poll"})
            if packet["job"]["state"] != "running":
                break
            time.sleep(0.02)
        assert packet["job"]["state"] == "completed", packet
        assert any(e.get("text") == "Привет" for e in packet["job"]["events"])
        assert host.status("character")["character"]["state"] == "active"
        assert client.request("POST", "/sections/character/commands/dispose", {})[0] == 404
        host.suspend("character")
        host.resume("character")
        restored = client.ok("POST", path, {"operation": "status"})
        assert restored["character"]["profile_id"] == "friendly"
        assert restored["history"][0]["assistant"] == "Привет"
    finally:
        server.close()
        host.close()
