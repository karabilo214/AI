"""Загрузка native-моделей и создание управляемой сессии."""

import gc
import os
import sys
import threading
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

from ramvi_engine.lifecycle import EngineSession, LoadFailure, ModelBundle
from ramvi_engine.model_location import is_gguf_model, resolve_gguf_name
from ramvi_engine.planning.automatic import (
    PlanJournal,
    PlanningDecision,
    PlanningOptions,
    apply_decision,
    choose_plan,
)


@dataclass(frozen=True)
class ModelConfig:
    model: str
    draft: str | None = None
    maxt: int = 2048
    cpu: bool = True
    resident: int | None = None
    tokenizer: str = "Qwen3-8B"
    planning: PlanningOptions | None = None

    def __post_init__(self) -> None:
        # dataclass config crossing a JSON process boundary.
        if isinstance(self.planning, dict):
            object.__setattr__(self, "planning", PlanningOptions(**self.planning))


def quiesce_cuda() -> None:
    """Перед освобождением владельца дождаться завершения GPU-команд."""
    torch = import_module("torch")

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    """Удалить циклы владельцев и освободить неиспользуемый CUDA allocator cache."""
    torch = import_module("torch")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _construct(cls: Any, *args: Any, **kwargs: Any) -> Any:
    # Сохраняем partial instance при ошибке __init__, включая запущенный Loader.
    engine = cls.__new__(cls)
    try:
        cls.__init__(engine, *args, **kwargs)
        return engine
    except BaseException:
        try:
            engine.close()
            quiesce_cuda()
            vars(engine).clear()
            cleanup_cuda()
        except Exception as cleanup_error:
            raise LoadFailure(str(cleanup_error), ModelBundle(engine, None)) from cleanup_error
        raise


def _open(name: str, **kwargs: Any) -> Any:
    from ramvi_engine.core.engine import Engine

    if not is_gguf_model(name):
        return _construct(Engine, name, **kwargs)
    from ramvi_engine.core.gengine import GGUFEngine
    from ramvi_engine.formats.gguf import GGUF, find_ollama

    g = GGUF(resolve_gguf_name(name, find_ollama))  # type: ignore[no-untyped-call]
    if g.meta.get("general.architecture") == "qwen3moe":
        from ramvi_engine.core.moe import MoEEngine

        return _construct(MoEEngine, name, **kwargs)
    return _construct(GGUFEngine, name, **kwargs)


def _load_bundle_once(config: ModelConfig, decision: PlanningDecision | None = None) -> ModelBundle:
    torch = import_module("torch")
    AutoTokenizer = import_module("transformers").AutoTokenizer

    from ramvi_engine.formats.model import find_model
    from ramvi_engine.formats.tokenizer_check import validate_draft, validate_tokenizer

    torch.set_grad_enabled(False)
    engine = draft = None
    try:
        if config.draft:
            gguf_draft = is_gguf_model(config.draft)
            draft = _open(
                config.draft,
                maxt=config.maxt,
                **({"vram_reserve": 0} if gguf_draft else {"cpu": False}),
            )
        gguf = is_gguf_model(config.model)
        cpu: bool | str = (
            False if not config.cpu else ("hybrid" if gguf and draft is not None else True)
        )
        engine_options: dict[str, Any] = {
            "maxt": config.maxt,
            "resident": config.resident,
            "cpu": cpu,
        }
        if decision is not None:
            engine_options["vram_reserve"] = decision.reserve_bytes
        engine = _open(config.model, **engine_options)
        tokenizer = AutoTokenizer.from_pretrained(
            find_model(config.tokenizer if gguf else config.model)  # type: ignore[no-untyped-call]
        )
        from ramvi_engine.model_template import configure_model_template

        configure_model_template(engine, tokenizer)
        if decision is None and gguf and config.resident is None:
            engine.autofit(tokenizer=tokenizer) if hasattr(engine, "gexp") else engine.autofit()
        validate_tokenizer(engine, tokenizer)  # type: ignore[no-untyped-call]
        if draft is not None:
            draft_tk = (
                tokenizer if hasattr(draft, "g") else AutoTokenizer.from_pretrained(draft.ck.dir)
            )
            validate_draft(engine, draft, tokenizer, draft_tk)  # type: ignore[no-untyped-call]
        return ModelBundle(engine, tokenizer, draft)
    except BaseException as failure:
        if isinstance(failure, LoadFailure):
            engine = failure.bundle.engine
        try:
            for owned in (engine, draft):
                if owned is not None:
                    owned.close()
            quiesce_cuda()
            for owned in (engine, draft):
                if owned is not None:
                    vars(owned).clear()
            cleanup_cuda()
        except Exception as cleanup_error:
            raise LoadFailure(str(cleanup_error), ModelBundle(engine, None, draft)) from failure
        raise


def _metadata_estimator(name: str, maxt: int) -> tuple[str, str, int, Any]:
    """Read metadata/mmap headers only; no inference or GPU weight allocation."""
    from ramvi_engine.planning.resources import estimate_gguf, estimate_hf

    if is_gguf_model(name):
        from ramvi_engine.formats.gguf import GGUF, find_ollama

        model = GGUF(resolve_gguf_name(name, find_ollama))  # type: ignore[no-untyped-call]
        architecture = model.meta["general.architecture"]
        if architecture not in ("qwen3", "qwen3moe"):
            raise NotImplementedError(architecture)
        return (
            "gguf",
            architecture,
            int(model.meta[f"{architecture}.block_count"]),
            (lambda route: estimate_gguf(model, route, maxt, batch=16)),
        )
    from ramvi_engine.formats.model import Ckpt

    checkpoint = Ckpt(name)  # type: ignore[no-untyped-call]
    return (
        "safetensors",
        checkpoint.cfg["model_type"],
        int(checkpoint.L),
        (lambda route: estimate_hf(checkpoint, route, maxt, batch=16)),
    )


def _resolve_plan(
    config: ModelConfig, options: PlanningOptions, resident_cap: int | None = None
) -> PlanningDecision:
    from ramvi_engine.config import env
    from ramvi_engine.planning.plan import ram_info
    from ramvi_engine.planning.resources import MemoryUsage

    torch = import_module("torch")
    model_format, architecture, layers, estimate_main = _metadata_estimator(
        config.model, config.maxt
    )
    draft_usage = MemoryUsage()
    if config.draft and options.draft != "off":
        _, _, draft_layers, estimate_draft = _metadata_estimator(config.draft, config.maxt)
        draft_usage = estimate_draft(["gpu"] * draft_layers).allocated

    def estimate(route: list[str], use_draft: bool, reserve: int) -> Any:
        value = estimate_main(route)
        return replace(
            value,
            draft=draft_usage if use_draft else MemoryUsage(),
            reserve=MemoryUsage(vram=reserve),
        )

    return choose_plan(
        layers,
        estimate,
        vram_available=int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0,
        ram_available=int(ram_info()[1]),  # type: ignore[no-untyped-call]
        maxt=config.maxt,
        architecture=architecture,
        model_format=model_format,
        cpu=config.cpu,
        draft_requested=config.draft is not None,
        options=options,
        requested_resident=config.resident,
        resident_cap=resident_cap,
        cpu_fraction=float(env("RAMVI_CPU_FRAC", "0.58")) if model_format == "gguf" else 0.58,
    )


def preview_plan(config: ModelConfig) -> dict[str, Any]:
    """Shared CLI/catalog preview; only metadata and current memory availability."""
    options = config.planning or PlanningOptions()
    return _resolve_plan(config, options).to_dict()


def _planning_warmup(bundle: ModelBundle, decision: PlanningDecision) -> None:
    """Bounded actual prefill/verify warmup, before accepting user requests."""
    engine = bundle.engine
    vocabulary = int(engine.V if hasattr(engine, "V") else engine.head.V)
    cap = decision.max_batch
    warm = min(32, max(1, engine.maxt - cap))
    tokens = [(1000 + index * 7919) % vocabulary for index in range(min(engine.maxt, warm + cap))]
    engine.process(tokens[:warm], 0, last_only=True)
    if len(tokens) > warm:
        engine.process(tokens[warm:], warm)
    engine.ctx = []
    if bundle.draft is not None:
        bundle.draft.ctx = []


def load_bundle(config: ModelConfig, *, _journal: PlanJournal | None = None) -> ModelBundle:
    """Legacy path or explicit bounded automatic plan, with safe pre-output OOM retry."""
    if config.planning is None:
        return _load_bundle_once(config)
    options = config.planning
    journal = _journal or PlanJournal()
    resident_cap = None
    attempts = (
        1 if options.mode == "manual" or config.resident is not None else options.max_load_attempts
    )
    torch = import_module("torch")
    for attempt in range(attempts):
        decision = _resolve_plan(config, options, resident_cap)
        journal.safe_append(
            "decision",
            {
                "attempt": attempt + 1,
                "model": os.path.basename(config.model),
                "decision": decision.to_dict(),
            },
        )
        effective = replace(
            config,
            resident=decision.resident,
            draft=config.draft if decision.draft_enabled else None,
            planning=None,
        )
        bundle = None
        try:
            bundle = _load_bundle_once(effective, decision)
            apply_decision(bundle.engine, decision, journal)
            if bundle.draft is not None:
                bundle.draft.max_batch = decision.max_batch
                bundle.draft.prefill_mode = "legacy"
            if decision.mode == "auto":
                torch.cuda.reset_peak_memory_stats()
                _planning_warmup(bundle, decision)
            bundle.engine.planning_controller.observe(bundle.engine)
            journal.safe_append(
                "loaded",
                {
                    "attempt": attempt + 1,
                    "placement": list(bundle.engine.place),
                    "max_batch": decision.max_batch,
                    "draft_enabled": bundle.draft is not None,
                },
            )
            return bundle
        except torch.OutOfMemoryError as failure:
            # No token has been published, so loading can be retried without sampling replay.
            if bundle is not None:
                try:
                    for owned in (bundle.engine, bundle.draft):
                        if owned is not None:
                            owned.close()
                    quiesce_cuda()
                    for owned in (bundle.engine, bundle.draft):
                        if owned is not None:
                            vars(owned).clear()
                except Exception as cleanup_error:
                    raise LoadFailure(str(cleanup_error), bundle) from failure
            bundle = None
            import traceback

            traceback.clear_frames(failure.__traceback__)
            cleanup_cuda()
            journal.safe_append(
                "load_oom",
                {
                    "attempt": attempt + 1,
                    "resident": decision.resident,
                    "max_batch": decision.max_batch,
                },
            )
            exhausted_knobs = (
                decision.resident == 0
                and decision.max_batch == 1
                and not decision.draft_enabled
                and decision.prefill_mode == "legacy"
            )
            if attempt + 1 == attempts or exhausted_knobs:
                raise
            resident_cap = max(0, decision.resident - max(1, decision.resident // 4))
            options = replace(
                options,
                max_batch=max(1, decision.max_batch // 2),
                prefill="legacy",
                draft="off" if options.draft == "auto" else options.draft,
            )
        except BaseException as failure:
            if bundle is not None:
                try:
                    for owned in (bundle.engine, bundle.draft):
                        if owned is not None:
                            owned.close()
                    quiesce_cuda()
                    for owned in (bundle.engine, bundle.draft):
                        if owned is not None:
                            vars(owned).clear()
                except Exception as cleanup_error:
                    raise LoadFailure(str(cleanup_error), bundle) from failure
            raise
    raise AssertionError("unreachable planning retry")


# Кэши ядра общие для main/draft и всех управляемых сессий процесса.
_CACHE_LOCK = threading.Lock()
_CACHE_OWNERS = 0


def _clear_execution_caches() -> None:
    for module_name, attributes in (
        ("ramvi_engine.kernels.wcomp", ("_SK",)),
        ("ramvi_engine.kernels.gkernels", ("_XP",)),
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            for attribute in attributes:
                getattr(module, attribute).clear()
    grammar = sys.modules.get("ramvi_engine.generation.constrain")
    if grammar is not None:
        lock = grammar._COMPILE_LOCK
        if not lock.acquire(timeout=30):
            raise TimeoutError("компилятор грамматики ещё занят")
        try:
            grammar._GRAMMARS.clear()
            grammar._COMPILERS.clear()
        finally:
            lock.release()


def create_session(config: ModelConfig, *, timeout: float = 30.0) -> EngineSession:
    leased = False
    journal = PlanJournal() if config.planning is not None else None

    def factory() -> ModelBundle:
        nonlocal leased
        global _CACHE_OWNERS
        with _CACHE_LOCK:
            if not leased:
                _CACHE_OWNERS += 1
                leased = True
        return load_bundle(config) if journal is None else load_bundle(config, _journal=journal)

    def release() -> None:
        nonlocal leased
        global _CACHE_OWNERS
        with _CACHE_LOCK:
            if leased:
                _CACHE_OWNERS -= 1
                leased = False
            if not _CACHE_OWNERS:
                _clear_execution_caches()
            cleanup_cuda()

    return EngineSession(factory, cleanup=release, quiesce=quiesce_cuda, timeout=timeout)
