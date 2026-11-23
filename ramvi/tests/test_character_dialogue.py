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


def test_streamed_reply_survives_a_copied_schema_reference_in_memory() -> None:
    """9.22: the real model wrote {"$ref": ...} as evidence; the reply must stay completed."""
    r = records()
    r[-1]["memory_candidates"] = [
        {
            "id": "m",
            "kind": "fact",
            "text": "x",
            "confidence": 1,
            "evidence": [{"$ref": "#/$defs/source"}],
        },
        {"id": "k", "kind": "fact", "text": "kept", "confidence": 1, "evidence": ["current"]},
    ]
    svc = Services(output="\n".join(json.dumps(x) for x in r) + "\n")
    m = active(svc)
    m.dialogue("start", {"text": "Тест", "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed" and not result["job"].get("fallback")
    assert result["job"]["memory_rejected"] == 1 and len(svc.requests) == 1
    assert [c["id"] for c in result["job"]["events"][-1]["memory_candidates"]] == ["k"]
    assert "$ref" not in svc.requests[0]["messages"][0]["content"]
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
        # B.15: файл больше прежнего потолка команды модулю (64 КБ) доходит до модуля целиком.
        big = {"name": "big.txt", "text": "я" * 150_000}  # 300 КБ в UTF-8
        args = {"text": "Смотри", "model": "first", "files": [big]}
        assert "response_id" in client.ok("POST", path, {"operation": "start", "arguments": args})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            packet = client.ok("POST", path, {"operation": "poll"})
            if packet["job"]["state"] != "running":
                break
            time.sleep(0.02)
        assert packet["job"]["state"] == "completed", packet
        status = client.ok("POST", path, {"operation": "status"})
        assert status["history"][-1]["user"] == "Смотри\n[file: big.txt]"
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


def test_segments_carry_the_pack_face_and_a_gesture_for_strong_emotion() -> None:
    """anger → лицо furious; сильная злость без движения — персонаж сам кричит (shout)."""
    r = records()
    r[1].update({"emotion": "anger", "intensity": 0.9, "animation": "none"})
    svc = Services(output="\n".join(json.dumps(x) for x in r) + "\n")
    m = active(svc)
    m.dialogue("start", {"text": "Тест", "model": "one"})
    result = finish(m)
    segment = next(e for e in result["job"]["events"] if e["type"] == "segment")
    assert (segment["expression"], segment["animation"]) == ("furious", "shout")
    prompt = svc.requests[0]["messages"][0]["content"]
    assert "turn_away" in prompt and '"furious"' in prompt  # модель видит жесты и глаза набора
    r[1].update({"emotion": "anger", "intensity": 0.2, "animation": "nod"})
    svc = Services(output="\n".join(json.dumps(x) for x in r) + "\n")
    m2 = active(svc)
    m2.dialogue("start", {"text": "Тест", "model": "one"})
    segment = next(e for e in finish(m2)["job"]["events"] if e["type"] == "segment")
    assert segment["animation"] == "nod"  # выбор модели не подменяется
    m.dispose()
    m2.dispose()


def test_owner_allows_profanity_and_the_model_is_told_to_swear() -> None:
    """Настройка «Разрешить нецензурную лексику»: выключено — запрет, включено — просьба ругаться."""
    svc = Services(output="\n".join(json.dumps(x) for x in records()) + "\n")
    m = active(svc)
    assert m.dialogue("status", {})["character"]["profanity"] is False
    m.dialogue("start", {"text": "Тест", "model": "one"})
    finish(m)
    first = svc.requests[-1]["messages"][0]["content"]
    assert "No profanity." in first and "мат" not in first.split("Swear")[0][-200:]
    m.dialogue("boundary", {"name": "profanity", "allowed": True})
    assert m.dialogue("status", {})["character"]["profanity"] is True
    m.dialogue("start", {"text": "Ещё", "model": "one"})
    finish(m)
    second = svc.requests[-1]["messages"][0]["content"]
    assert "No profanity." not in second and "allowed swearing" in second
    assert "скажи «триста»" in second  # узнаёт ловушку
    assert "обломался" not in second  # но готового ответа нет: модель не повторяет его по кругу
    assert "Never repeat your earlier replies" in second
    assert "Absurd or impossible requests" in second  # не «гавкает», а отвечает по-человечески
    assert "never how you sound" in second  # особые случаи не отменяют характер
    assert "гав" not in second.lower()  # без готовой реплики, которую модель повторяла бы
    assert m.dialogue("status", {})["character"]["profanity_level"] == 2
    assert "Swear moderately" in second  # по умолчанию «средне»
    prompts = {}
    for level in (1, 3):
        m.dialogue("profanity_level", {"level": level})
        m.dialogue("start", {"text": "Ещё", "model": "one"})
        finish(m)
        prompts[level] = svc.requests[-1]["messages"][0]["content"]
    assert "Swear rarely" in prompts[1] and "Swear moderately" not in prompts[1]
    assert "Swear a lot" in prompts[3] and "Swear rarely" not in prompts[3]
    assert m.dialogue("profanity_level", {"level": 4})["error"] == "ValueError"
    restored = CharacterModule()  # ползунок переживает перезапуск
    restored.initialize({"services": svc, "checkpoint": m.checkpoint()})
    restored.activate()
    assert restored.dialogue("status", {})["character"]["profanity_level"] == 3
    m.dialogue("boundary", {"name": "profanity", "allowed": False})
    m.dialogue("start", {"text": "Ещё", "model": "one"})
    finish(m)
    off = svc.requests[-1]["messages"][0]["content"]
    assert "No profanity." in off and "allowed swearing" not in off  # галочка выключена
    restored.dispose()
    m.dispose()


def test_reply_parts_carry_their_own_voice_style_to_the_voice() -> None:
    """9.11: ответ делится на части по эмоции; подача голоса части выбирается из списка."""
    r = records()
    r[1].update({"emotion": "joy", "voice_style": "excited", "text": "Ок!"})
    two = dict(
        r[1], id="s2", seq=2, text="Давай попробуем…", emotion="interest", voice_style="intriguing"
    )
    r = [r[0], r[1], two, dict(r[2], last_seq=2)]
    svc = Services(output="\n".join(json.dumps(x) for x in r) + "\n")
    m = active(svc)
    m.dialogue("start", {"text": "Тест", "model": "one"})
    segments = [e for e in finish(m)["job"]["events"] if e["type"] == "segment"]
    assert [(s["text"], s["voice_style"]) for s in segments] == [
        ("Ок!", "excited"),
        ("Давай попробуем…", "intriguing"),
    ]
    prompt = svc.requests[0]["messages"][0]["content"]
    assert "intriguing" in prompt and "into segments wherever the emotion" in prompt
    assert "startled" in prompt and "reproachful" in prompt  # вскрик и укор — среди подач
    m.dispose()


def test_attached_file_goes_to_this_reply_only_and_history_keeps_its_name() -> None:
    svc = Services()
    m = active(svc)
    secret = "Проверочное число 700021"
    m.dialogue(
        "start",
        {
            "text": "Что скажешь?",
            "model": "one",
            "files": [{"name": "C:\\docs\\note.txt", "text": secret}],
        },
    )
    result = finish(m)
    assert result["job"]["state"] == "completed"
    prompt = "\n".join(msg["content"] for msg in svc.requests[-1]["messages"])
    assert secret in prompt and "note.txt" in prompt and "docs" not in prompt
    # В истории — слова владельца и имя файла, без содержимого.
    assert result["history"][0]["user"] == "Что скажешь?\n[file: note.txt]"
    m.dialogue("start", {"text": "А теперь?", "model": "one"})
    finish(m)
    assert secret not in "\n".join(msg["content"] for msg in svc.requests[-1]["messages"])
    # Только файл, без слов, — тоже реплика; содержимое модель видит.
    m.dialogue("start", {"text": "", "model": "one", "files": [{"name": "a.txt", "text": "42"}]})
    assert finish(m)["job"]["state"] == "completed"
    current = json.loads(svc.requests[-1]["messages"][-1]["content"])
    assert current["source_id"] == "current" and current["text"].endswith(
        '<file name="a.txt">\n42\n</file>'
    )
    m.dispose()


@pytest.mark.parametrize(
    "files",
    [
        [{"name": "bin.txt", "text": "a\x00b"}],
        [{"name": "a.txt", "text": "x", "path": "/etc"}],
        [{"name": "", "text": "x"}],
        "not a list",
    ],
)
def test_bad_files_are_refused_before_the_model(files: Any) -> None:
    svc = Services()
    m = active(svc)
    with pytest.raises(ValueError):
        m._dialogue.start("Привет", "one", files)
    assert not svc.requests
    m.dispose()


def test_file_that_does_not_fit_reports_tokens_instead_of_a_generic_error() -> None:
    svc = Services()
    m = active(svc)
    # Тестовый счётчик: 10 символов — 1 токен; окно локальной модели 8192 (B.16: без лимитов разделов).
    m.dialogue(
        "start",
        {"text": "Вот", "model": "one", "files": [{"name": "big.txt", "text": "я" * 90_000}]},
    )
    job = finish(m)["job"]
    assert job["state"] == "error" and job["error"] == "ContextOverflow"
    assert (
        job["overflow"]["section"] == "required"
        and job["overflow"]["needed"] > job["overflow"]["available"]
    )
    assert not svc.requests  # до модели не дошло
    m.dispose()


def test_long_reply_text_is_limited_by_tokens_and_kept_whole_in_history() -> None:
    svc = Services()
    m = active(svc)
    long_text = "Слово " * 2000  # ~24 КБ: больше прежних 4096 байт, но помещается в бюджет
    m.dialogue("configure", {"draft": long_text})
    m.dialogue("start", {"text": long_text, "model": "one"})
    result = finish(m)
    assert result["job"]["state"] == "completed"
    assert result["history"][-1]["user"] == long_text
    # Ещё больше — не помещается в бюджет реплики: понятная ошибка с числами.
    m.dialogue("start", {"text": "я" * 90_000, "model": "one"})
    job = finish(m)["job"]
    assert (
        job["error"] == "ContextOverflow"
        and job["overflow"]["needed"] > job["overflow"]["available"]
    )
    checkpoint = m.checkpoint()
    m.dispose()
    restored = CharacterModule()
    restored.initialize({"services": svc, "checkpoint": checkpoint})
    restored.activate()  # история с большим последним ходом читается обратно
    assert restored.dialogue()["history"][-1]["user"] == "я" * 90_000
    restored.dispose()


class WindowServices(Services):
    """Сервис, который сообщает окно модели (B.16): внешняя (облако) или локальная."""

    def __init__(self, *, external: bool, context: int | None) -> None:
        super().__init__()
        self.external, self.context = external, context

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "llm.capabilities":
            assert payload == {"model": "one"}  # окно — именно выбранной модели
            return {
                "available": True,
                "capabilities": {"streaming": True, "external": self.external},
                "context": self.context,
            }
        return super().call(method, payload)


@pytest.mark.parametrize(
    ("external", "context", "fits", "minimum"),
    [
        (True, None, True, None),  # облако: окно неизвестно — общего потолка нет
        (False, 32768, True, 32768),  # профиль задачи с длинным контекстом
        (False, None, False, 8192),  # нет сведений — прежний минимум персонажа
    ],
)
def test_request_size_follows_the_model_window(
    external: bool, context: int | None, fits: bool, minimum: int | None
) -> None:
    svc = WindowServices(external=external, context=context)
    m = active(svc)
    # ~12 000 токенов тестового счётчика: больше 8192, меньше 32768.
    m.dialogue("start", {"text": "я" * 120_000, "model": "one"})
    job = finish(m)["job"]
    assert (job["state"] == "completed") is fits, job.get("error_detail")
    if fits:
        assert svc.requests[-1].get("ramvi_min_context") == minimum
    else:
        assert job["overflow"]["available"] == 8192 - 1536 - 256
    m.dispose()


def test_local_window_is_the_profile_or_the_loaded_context() -> None:
    from ramvi_service.engines import SwitchableProvider

    class Manager:
        model_id = "local-a"
        _loaded_context = 16384

        def context(self) -> int:
            return 2048

    remote = object()
    switch = SwitchableProvider(Manager(), lambda m: remote if m == "cloud:c:m" else None)  # type: ignore[arg-type]
    assert switch.context_for({"ramvi_model_id": "local-a"}) == 16384  # уже загружена больше
    assert switch.context_for({"ramvi_model_id": "local-b"}) == 2048  # загрузится по профилю
    assert switch.context_for({"ramvi_model_id": "cloud:c:m"}) is None  # облако: неизвестно


def test_voice_gets_the_delivery_in_words() -> None:
    """9.11: облачному голосу уходит описание подачи, а не голый ярлык."""
    from ramvi_character.speech import VOICE_DELIVERY, voice_style

    style = voice_style("anger", 0.9, "startled")
    assert "startled exclamation" in style and "0.90" in style and "anger" in style
    from ramvi_character.dialogue import VOICE_STYLES

    assert set(VOICE_STYLES) <= set(VOICE_DELIVERY)  # у каждой подачи есть описание
