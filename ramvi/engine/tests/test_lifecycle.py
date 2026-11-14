"""Жизненный цикл: реальные владельцы Python, гонки и HTTP без GPU."""

import gc
import json
import threading
import weakref
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_engine.lifecycle import EngineSession, GenerationCancelled, ModelBundle, ModelUnavailable
from ramvi_engine.server.app import App, make_handler


class FakeEngine:
    maxt = 32
    ck = SimpleNamespace(eos=set())

    def __init__(self) -> None:
        self.closed = 0
        self.finalized = False

    def close(self) -> None:
        self.closed += 1

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            yield from range(kwargs.get("max_new", 5))
        finally:
            self.finalized = True


def make_session(**kwargs: Any) -> tuple[EngineSession, list[weakref.ReferenceType[FakeEngine]]]:
    refs: list[weakref.ReferenceType[FakeEngine]] = []

    def factory() -> ModelBundle:
        eng, draft = FakeEngine(), FakeEngine()
        refs.extend((weakref.ref(eng), weakref.ref(draft)))
        return ModelBundle(eng, SimpleNamespace(), draft)

    return EngineSession(factory, **kwargs), refs


def test_load_warmup_restart_and_release_all_owners() -> None:
    session, refs = make_session()
    assert session.status()["state"] == "unloaded"
    assert session.load()["state"] == "ready"
    session.load()
    assert len(refs) == 2  # idempotent load
    assert session.warmup([1], max_new=2)["generated"] == 2
    assert session.restart()["state"] == "ready"
    assert refs[0]() is refs[1]() is None
    assert len(refs) == 4
    assert list(session.generate([1])) == [0, 1, 2, 3, 4]
    session.unload()
    session.unload()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert session.status()["state"] == "unloaded"
    with pytest.raises(ModelUnavailable):
        list(session.generate([1]))
    assert session.status()["state"] == "unloaded"


def test_cancel_active_only_and_close_generator() -> None:
    session, refs = make_session()
    session.load()
    assert not session.cancel()["requested"]
    gen = session.generate([1])
    assert next(gen) == 0
    assert session.cancel()["requested"]
    with pytest.raises(GenerationCancelled):
        next(gen)
    assert refs[0]().finalized  # type: ignore[union-attr]
    assert not session.status()["active"]
    assert list(session.generate([1], max_new=1)) == [0]
    gen = session.generate([1])
    next(gen)
    gen.close()
    assert session.status()["state"] == "ready"
    session.close()


def test_failure_keeps_unsafe_resources_until_close_succeeds() -> None:
    eng = FakeEngine()
    fail = True

    def quiesce() -> None:
        if fail:
            raise TimeoutError("GPU still busy")

    session = EngineSession(lambda: ModelBundle(eng, object()), quiesce=quiesce)
    session.load()
    with pytest.raises(TimeoutError):
        session.unload()
    assert session.status()["state"] == "error"
    assert session.status()["loaded"]
    with pytest.raises(ModelUnavailable):
        session.load()
    fail = False
    session.unload()
    assert not session.status()["loaded"]


def test_unload_waits_for_active_step_and_blocks_new_work() -> None:
    entered, release = threading.Event(), threading.Event()

    class Slow(FakeEngine):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert release.wait(5)
            yield 1

    session = EngineSession(lambda: ModelBundle(Slow(), object()), timeout=0.03)
    session.load()
    errors = []

    def run() -> None:
        try:
            list(session.generate([1]))
        except GenerationCancelled:
            errors.append("cancelled")

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(TimeoutError):
            session.unload()
        assert session.status()["loaded"]  # never frees a running kernel's buffers
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        session.unload()
    assert errors == ["cancelled"]


def test_factory_failure_then_success_and_invalid_warmup() -> None:
    calls = 0

    def factory() -> ModelBundle:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("load failed")
        return ModelBundle(FakeEngine(), object())

    session = EngineSession(factory)
    with pytest.raises(ValueError):
        session.load()
    assert session.status()["state"] == "error"
    session.load()
    invalid: list[Any] = [[], [-1], [True]]
    for tokens in invalid:
        with pytest.raises(ValueError):
            session.warmup(tokens)
    session.close()


def test_lifecycle_http_auth_and_unloaded_requests() -> None:
    session, _ = make_session()
    app = App(session, session.tokenizer, "test", draft=session.draft_handle, api_key="secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def request(method: str, path: str, body: Any = None, auth: bool = True) -> tuple[int, Any]:
        headers = {"Authorization": "Bearer secret"} if auth else {}
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.request(method, path, json.dumps(body) if body is not None else None, headers)
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    try:
        assert request("POST", "/engine/load", {}, False)[0] == 401
        assert request("GET", "/engine/status", auth=False)[0] == 401
        assert request("POST", "/v1/completions", {"prompt": "x"})[0] == 503
        assert request("POST", "/engine/load", {})[1]["state"] == "ready"
        assert request("POST", "/engine/warmup", {"tokens": [1], "max_new": 2})[1]["generated"] == 2
        assert request("POST", "/engine/restart", {"model": "arbitrary"})[0] == 400
        assert request("POST", "/engine/restart", {})[1]["state"] == "ready"
        assert request("POST", "/engine/unload", {})[1]["state"] == "unloaded"
        assert request("GET", "/engine/status")[1]["loaded"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        session.close()
        assert not thread.is_alive()


def test_construct_failure_closes_partial_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from ramvi_engine import model_loader

    calls = []

    class Broken:
        def __init__(self) -> None:
            self.buffer = object()
            raise ValueError("partial")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(model_loader, "quiesce_cuda", lambda: calls.append("quiesce"))
    monkeypatch.setattr(model_loader, "cleanup_cuda", lambda: calls.append("cleanup"))
    with pytest.raises(ValueError, match="partial"):
        model_loader._construct(Broken)
    assert calls == ["close", "quiesce", "cleanup"]


@pytest.mark.parametrize("gguf", [False, True])
def test_loader_preserves_model_options(gguf: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from ramvi_engine import model_loader

    opened: list[tuple[str, dict[str, Any]]] = []
    checked = []

    def open_model(name: str, **kwargs: Any) -> Any:
        opened.append((name, kwargs))
        return SimpleNamespace(ck=SimpleNamespace(dir="draft-folder"), close=lambda: None)

    tokenizer = SimpleNamespace()
    modules = {
        "ramvi_engine.formats.model": SimpleNamespace(find_model=lambda x: x),
        "ramvi_engine.formats.tokenizer_check": SimpleNamespace(
            validate_tokenizer=lambda *args: checked.append("main"),
            validate_draft=lambda *args: checked.append("draft"),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(model_loader, "_open", open_model)
    monkeypatch.setattr(
        model_loader,
        "import_module",
        lambda name: (
            SimpleNamespace(set_grad_enabled=lambda _: None)
            if name == "torch"
            else SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer))
        ),
    )
    bundle = model_loader.load_bundle(
        model_loader.ModelConfig("main.gguf" if gguf else "main", "draft", maxt=1234, resident=1)
    )
    assert bundle.tokenizer is tokenizer
    assert opened == [
        ("draft", {"maxt": 1234, "cpu": False}),
        (
            "main.gguf" if gguf else "main",
            {"maxt": 1234, "resident": 1, "cpu": "hybrid" if gguf else True},
        ),
    ]
    assert checked == ["main", "draft"]


def test_same_thread_unload_cannot_free_suspended_generator() -> None:
    session, refs = make_session()
    session.load()
    gen = session.generate([1])
    next(gen)
    with pytest.raises(TimeoutError):
        session.unload()
    assert refs[0]() is not None
    gen.close()
    session.unload()
    assert all(ref() is None for ref in refs)


def test_partial_failure_retains_owner_for_retry() -> None:
    from ramvi_engine.lifecycle import LoadFailure

    eng = FakeEngine()

    def factory() -> ModelBundle:
        raise LoadFailure("partial model still owns buffers", ModelBundle(eng, None))

    session = EngineSession(factory)
    with pytest.raises(LoadFailure):
        session.load()
    assert session.status()["loaded"]
    session.unload()
    assert eng.closed == 1
    assert not session.status()["loaded"]


def test_nested_generation_does_not_corrupt_active_owner() -> None:
    session, _ = make_session()
    session.load()
    gen = session.generate([1])
    next(gen)
    with pytest.raises(ModelUnavailable):
        list(session.generate([1]))
    assert session.status()["active"]
    assert session.cancel()["requested"]
    with pytest.raises(GenerationCancelled):
        next(gen)
    session.close()


def test_app_lease_covers_final_event_until_generator_closed() -> None:
    session = EngineSession(
        lambda: ModelBundle(
            FakeEngine(), SimpleNamespace(decode=lambda ids, **kwargs: "".join(str(t) for t in ids))
        )
    )
    session.load()
    app = App(session, session.tokenizer, "test")
    gen = app._run([1])
    for kind, _ in gen:
        if kind == "end":
            break
    entered, done = threading.Event(), threading.Event()

    def unload() -> None:
        entered.set()
        session.unload()
        done.set()

    thread = threading.Thread(target=unload)
    thread.start()
    try:
        assert entered.wait(5)
        assert session.status()["loaded"]
        assert not done.is_set()
    finally:
        gen.close()
        thread.join(5)
    assert done.is_set()
    assert not thread.is_alive()
    assert not session.status()["loaded"]


def test_execution_cache_released_only_after_last_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from ramvi_engine import model_loader

    scratch = {"buffer": object()}
    monkeypatch.setitem(sys.modules, "ramvi_engine.kernels.wcomp", SimpleNamespace(_SK=scratch))
    monkeypatch.setattr(model_loader, "load_bundle", lambda _: ModelBundle(FakeEngine(), object()))
    monkeypatch.setattr(model_loader, "quiesce_cuda", lambda: None)
    monkeypatch.setattr(model_loader, "cleanup_cuda", lambda: None)
    first = model_loader.create_session(model_loader.ModelConfig("one"))
    second = model_loader.create_session(model_loader.ModelConfig("two"))
    try:
        first.load()
        second.load()
        first.unload()
        first.unload()  # idempotent release must not decrement twice
        assert scratch
        second.unload()
        assert not scratch
    finally:
        first.close()
        second.close()
    assert model_loader._CACHE_OWNERS == 0


def test_targeted_cancel_does_not_cancel_foreign_and_rejects_late_start() -> None:
    session, _ = make_session()
    session.load()
    generator = session.generate([1], _generation_id="owner_a")
    next(generator)
    session.cancel("owner_b")
    assert next(generator) == 1
    session.cancel("owner_a")
    with pytest.raises(GenerationCancelled):
        next(generator)
    with pytest.raises(GenerationCancelled):
        list(session.generate([1], _generation_id="owner_b"))
    assert list(session.generate([1], max_new=1, _generation_id="fresh_owner")) == [0]
    session.close()


def test_cancel_waiting_request_does_not_wait_for_unrelated_owner() -> None:
    session, _ = make_session()
    session.load()
    active = session.generate([1], _generation_id="active_owner")
    next(active)
    entered, stopped = threading.Event(), threading.Event()

    def queued() -> None:
        entered.set()
        try:
            with session.request(request_id="waiting_owner"):
                raise AssertionError("cancelled queued work must not execute")
        except GenerationCancelled:
            stopped.set()

    worker = threading.Thread(target=queued)
    worker.start()
    try:
        assert entered.wait(2)
        session.cancel("waiting_owner")
        assert stopped.wait(2)
        assert next(active) == 1
    finally:
        active.close()
        worker.join(2)
        session.close()
    assert not worker.is_alive()
