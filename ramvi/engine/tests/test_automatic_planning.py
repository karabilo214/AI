"""Bounded resource planning, explicit/manual limits, OOM retry and real control flow."""

import gc
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ramvi_engine.lifecycle import EngineSession, LoadFailure, ModelBundle
from ramvi_engine.planning.automatic import (
    NoMemoryPlan,
    PlanJournal,
    PlanningController,
    PlanningDecision,
    PlanningOptions,
    apply_decision,
    automatic_reserve,
    batch_limit,
    choose_plan,
    placement,
    prefill_enabled,
)
from ramvi_engine.planning.resources import MemoryUsage, ResourceEstimate


def estimate(route: list[str], use_draft: bool, reserve: int) -> ResourceEstimate:
    return ResourceEstimate(
        weights=MemoryUsage(vram=10 + route.count("gpu") * 30, ram=5, mapped=10),
        draft=MemoryUsage(vram=15 if use_draft else 0),
        reserve=MemoryUsage(vram=reserve),
    )


def choose(**kwargs: Any) -> PlanningDecision:
    args: dict[str, Any] = dict(
        layers=3,
        estimate=estimate,
        vram_available=100,
        ram_available=100,
        maxt=2048,
        architecture="qwen3",
        model_format="gguf",
        cpu=True,
        draft_requested=True,
        options=PlanningOptions(reserve_bytes=0, max_batch=16),
    )
    args.update(kwargs)
    return choose_plan(**args)


def test_hand_solved_memory_choice_and_manual_override() -> None:
    selected = choose()
    assert selected.resident == 2  # 10 + 2*30 + 15 = 85; three residents need115.
    assert selected.draft_enabled
    assert selected.resources.required.vram == 85
    assert selected.prefill_mode == "legacy"  # no speculative blocked-prefill default
    manual = choose(requested_resident=3)
    assert manual.mode == "manual" and manual.resident == 3
    assert manual.assessment["vram_shortfall"] == 15
    no_draft = choose(options=PlanningOptions(reserve_bytes=0, draft="off"))
    assert no_draft.resident == 3 and not no_draft.draft_enabled


def test_paging_is_explicit_and_owned_ram_cannot_be_ignored() -> None:
    def mapped(route: list[str], use: bool, reserve: int) -> ResourceEstimate:
        return ResourceEstimate(weights=MemoryUsage(vram=10, ram=50, mapped=1000))

    assert choose(estimate=mapped).assessment["ram_shortfall"] == 950
    with pytest.raises(NoMemoryPlan):
        choose(estimate=mapped, options=PlanningOptions(reserve_bytes=0, allow_paging=False))
    with pytest.raises(NoMemoryPlan):
        choose(estimate=mapped, ram_available=49)
    with pytest.raises(NoMemoryPlan):
        choose(vram_available=0)


def test_moe_placement_and_bounded_search() -> None:
    assert placement(
        4, 2, architecture="qwen3moe", model_format="gguf", cpu=True, draft_enabled=False
    ) == ("cpu", "cpu", "gpu", "gpu")
    with pytest.raises(NotImplementedError):
        placement(
            4, 2, architecture="qwen3moe", model_format="gguf", cpu=False, draft_enabled=False
        )
    calls = []

    def impossible(route: list[str], draft: bool, reserve: int) -> ResourceEstimate:
        calls.append((route, draft))
        return ResourceEstimate(weights=MemoryUsage(vram=1000))

    with pytest.raises(NoMemoryPlan):
        choose(estimate=impossible)
    assert len(calls) == 2 * (3 + 1)
    with pytest.raises(NotImplementedError):
        choose(architecture="qwen3moe", options=PlanningOptions(reserve_bytes=0, prefill="blocked"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"draft": "unknown"},
        {"prefill": "unknown"},
        {"reserve_bytes": -1},
        {"reserve_bytes": True},
        {"max_batch": 3},
        {"max_batch": 4.0},
        {"max_load_attempts": 0},
        {"max_load_attempts": 1.5},
        {"allow_paging": 1},
    ],
)
def test_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PlanningOptions(**kwargs)


def test_explicit_runtime_caps_and_prefill_policy() -> None:
    assert automatic_reserve(8 * 2**30) == 2**30
    assert automatic_reserve(2 * 2**30) == 256 * 2**20
    assert batch_limit(object()) == 16
    with pytest.raises(ValueError):
        batch_limit(SimpleNamespace(max_batch=True))
    assert not prefill_enabled(SimpleNamespace(prefill_mode="legacy"), 10000, lambda _: True)
    assert prefill_enabled(SimpleNamespace(prefill_mode="blocked"), 256, lambda _: False)
    assert not prefill_enabled(SimpleNamespace(prefill_mode="blocked"), 255, lambda _: True)
    assert prefill_enabled(object(), 1, lambda _: True)
    assert not choose(options=PlanningOptions(reserve_bytes=0, max_batch=1)).draft_enabled
    with pytest.raises(ValueError):
        choose(draft_requested=False, options=PlanningOptions(draft="on"))


def test_bounded_atomic_journal_and_actual_placement(tmp_path: Path) -> None:
    journal = PlanJournal(tmp_path / "plan.jsonl", limit=2)
    for index in range(3):
        journal.append("decision", {"index": index})
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [row["payload"]["index"] for row in rows] == [1, 2]
    assert list(tmp_path.iterdir()) == [journal.path]
    decision = choose()
    engine = SimpleNamespace(place=list(decision.placement))
    apply_decision(engine, decision, journal)
    engine.place[0] = "stream"
    engine.planning_controller.observe(engine)
    report = engine.placement_decision
    assert report["actual_resident"] == 1
    assert report["estimate_status"] == "stale_after_runtime_change"
    assert json.loads(journal.path.read_text().splitlines()[-1])["event"] == "actual"


def test_journal_concurrent_rows_and_io_failure_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = PlanJournal(tmp_path / "plan.jsonl")

    def writer() -> None:
        for index in range(10):
            journal.append("actual", {"index": index})

    threads = [threading.Thread(target=writer) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(journal.path.read_text().splitlines()) == 30

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(journal, "append", fail)
    engine = SimpleNamespace(place=["gpu"] * 3)
    controller = PlanningController(choose(), journal)
    controller.observe(engine, "error")
    assert engine.placement_decision["journal_error"] == "OSError"


class FakeCore:
    V = 100
    maxt = 64

    def __init__(self, resident: int) -> None:
        self.place = ["gpu"] * resident + ["stream"] * (4 - resident)
        self.ctx: list[int] = []
        self.closed = False
        self.warmups = 0

    def close(self) -> None:
        self.closed = True

    def process(self, *args: Any, **kwargs: Any) -> list[int]:
        self.warmups += 1
        return [7]


class FakeOOM(RuntimeError):
    pass


def test_auto_load_oom_retry_downgrades_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ramvi_engine import model_loader

    options = PlanningOptions(reserve_bytes=0, max_batch=16)
    decisions = []
    opened = []

    def resolve(config: Any, opts: PlanningOptions, cap: int | None) -> PlanningDecision:
        decisions.append((opts, cap))
        resident = 4 if cap is None else cap
        route = ("gpu",) * resident + ("stream",) * (4 - resident)
        return PlanningDecision(
            "auto",
            resident,
            0,
            opts.max_batch or 16,
            opts.draft != "off",
            "legacy",
            route,
            ResourceEstimate(),
            {},
            (),
        )

    def load(config: Any, decision: PlanningDecision) -> ModelBundle:
        opened.append(config)
        if len(opened) == 1:
            raise FakeOOM("startup OOM")
        return ModelBundle(FakeCore(config.resident), object())

    monkeypatch.setattr(model_loader, "_resolve_plan", resolve)
    monkeypatch.setattr(model_loader, "_load_bundle_once", load)
    monkeypatch.setattr(model_loader, "PlanJournal", lambda: PlanJournal(tmp_path / "plan.jsonl"))
    monkeypatch.setattr(model_loader, "cleanup_cuda", lambda: None)
    monkeypatch.setattr(
        model_loader,
        "import_module",
        lambda _: SimpleNamespace(
            OutOfMemoryError=FakeOOM, cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None)
        ),
    )
    bundle = model_loader.load_bundle(model_loader.ModelConfig("model", "draft", planning=options))
    assert len(opened) == 2
    assert opened[0].resident == 4 and opened[1].resident == 3
    assert opened[1].draft is None
    assert decisions[1][0].max_batch == 8
    assert bundle.engine.max_batch == 8 and bundle.engine.warmups == 2
    assert bundle.engine.placement_decision["actual_resident"] == 3
    assert "load_oom" in [
        json.loads(line)["event"] for line in (tmp_path / "plan.jsonl").read_text().splitlines()
    ]


def test_manual_load_oom_never_changes_explicit_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ramvi_engine import model_loader

    opened = []

    def load(config: Any, decision: PlanningDecision) -> ModelBundle:
        opened.append(config.resident)
        raise FakeOOM("forced plan")

    monkeypatch.setattr(
        model_loader, "_resolve_plan", lambda *args: replace(choose(), mode="manual", resident=2)
    )
    monkeypatch.setattr(model_loader, "_load_bundle_once", load)
    monkeypatch.setattr(model_loader, "PlanJournal", lambda: PlanJournal(tmp_path / "plan.jsonl"))
    monkeypatch.setattr(model_loader, "cleanup_cuda", lambda: None)
    monkeypatch.setattr(
        model_loader, "import_module", lambda _: SimpleNamespace(OutOfMemoryError=FakeOOM)
    )
    with pytest.raises(FakeOOM):
        model_loader.load_bundle(
            model_loader.ModelConfig("model", resident=2, planning=PlanningOptions(mode="manual"))
        )
    assert opened == [2]


def test_partial_oom_cleanup_failure_retains_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ramvi_engine import model_loader

    class Unsafe(FakeCore):
        def close(self) -> None:
            raise RuntimeError("DMA still active")

    engine = Unsafe(2)
    monkeypatch.setattr(model_loader, "_resolve_plan", lambda *args: choose())
    monkeypatch.setattr(
        model_loader, "_load_bundle_once", lambda *args: ModelBundle(engine, object())
    )
    monkeypatch.setattr(
        model_loader, "_planning_warmup", lambda *args: (_ for _ in ()).throw(FakeOOM("warmup"))
    )
    monkeypatch.setattr(model_loader, "PlanJournal", lambda: PlanJournal(tmp_path / "plan.jsonl"))
    monkeypatch.setattr(
        model_loader,
        "import_module",
        lambda _: SimpleNamespace(
            OutOfMemoryError=FakeOOM, cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None)
        ),
    )
    with pytest.raises(LoadFailure) as failure:
        model_loader.load_bundle(model_loader.ModelConfig("model", planning=PlanningOptions()))
    assert failure.value.bundle.engine is engine
    assert engine.place  # buffers were not cleared after unsafe close


def test_observer_cannot_mask_model_error_or_hold_lifecycle_lock() -> None:
    def fail_observer(*args: Any) -> None:
        raise OSError("journal unavailable")

    controller = SimpleNamespace(observe=fail_observer, journal_error=None)

    class Broken:
        planning_controller = controller

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("model failure")
            yield 0

    session = EngineSession(lambda: ModelBundle(Broken(), object()))
    session.load()
    with pytest.raises(ValueError, match="model failure"):
        list(session.generate([1]))
    assert controller.journal_error == "OSError"
    assert not session.status()["active"]
    session.unload()
    gc.collect()


def test_native_generate_caps_preserve_control_tokens() -> None:
    pytest.importorskip("numpy")
    from .legacy.test_o02 import Toy

    for cap in (1, 2, 4, 8, 16):
        engine, draft = Toy(), Toy()  # type: ignore[no-untyped-call]
        engine.max_batch = cap
        observed: list[int] = []
        original = draft.draft

        def record(
            seq: Any,
            k: int,
            cum: Any = None,
            _observed: list[int] = observed,
            _original: Any = original,
        ) -> Any:
            _observed.append(k)
            return _original(seq, k, cum)

        draft.__dict__["draft"] = record
        assert list(engine.generate([1, 2, 3], 53, draft=draft, k=15)) == [7] * 53
        assert all(k <= cap - 1 for k in observed)
        if cap == 1:
            assert not observed


def test_manual_dense_runtime_does_not_silently_demote() -> None:
    import ast

    path = Path(__file__).resolve().parents[1] / "ramvi_engine/core/gengine.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GGUFEngine"
    )
    process = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "process"
    )
    namespace: dict[str, Any] = {"torch": SimpleNamespace(OutOfMemoryError=FakeOOM)}
    exec(compile(ast.Module(body=[process], type_ignores=[]), str(path), "exec"), namespace)
    demoted = []

    def fail(*args: Any, **kwargs: Any) -> None:
        raise FakeOOM("runtime")

    engine = SimpleNamespace(
        ctx=[1, 2],
        place=["gpu"],
        L=1,
        allow_oom_demote=False,
        _process=fail,
        _demote=lambda layer: demoted.append(layer),
    )
    with pytest.raises(FakeOOM):
        namespace["process"](engine, [7], 0)
    assert not demoted
    assert engine.place == ["gpu"] and engine.ctx == []
