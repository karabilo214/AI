"""Conservative bounded memory planning and actual-placement journal.

This module imports no numerical backend. Estimates are budgets, not measured speed.
"""

import copy
import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ramvi_engine.config import data_dir
from ramvi_engine.planning.resources import ResourceEstimate

GiB = 2**30
BATCHES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PlanningOptions:
    """None in ModelConfig preserves legacy loading; explicit options enable planning."""

    mode: Literal["auto", "manual"] = "auto"
    reserve_bytes: int | None = None
    max_batch: int | None = None
    draft: Literal["auto", "on", "off"] = "auto"
    prefill: Literal["auto", "legacy", "blocked"] = "auto"
    max_load_attempts: int = 4
    allow_paging: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "manual") or self.draft not in ("auto", "on", "off"):
            raise ValueError("unknown planning mode/draft policy")
        if self.prefill not in ("auto", "legacy", "blocked"):
            raise ValueError("unknown prefill mode")
        if self.reserve_bytes is not None and (
            isinstance(self.reserve_bytes, bool)
            or not isinstance(self.reserve_bytes, int)
            or self.reserve_bytes < 0
        ):
            raise ValueError("reserve_bytes must be a nonnegative integer")
        if self.max_batch is not None and (
            isinstance(self.max_batch, bool)
            or not isinstance(self.max_batch, int)
            or self.max_batch not in BATCHES
        ):
            raise ValueError("max_batch must be 1, 2, 4, 8 or 16")
        if (
            isinstance(self.max_load_attempts, bool)
            or not isinstance(self.max_load_attempts, int)
            or not 1 <= self.max_load_attempts <= 16
        ):
            raise ValueError("max_load_attempts must be between 1 and 16")
        if not isinstance(self.allow_paging, bool):
            raise ValueError("allow_paging must be boolean")


@dataclass(frozen=True)
class PlanningDecision:
    mode: str
    resident: int
    reserve_bytes: int
    max_batch: int
    draft_enabled: bool
    prefill_mode: str
    placement: tuple[str, ...]
    resources: ResourceEstimate
    assessment: dict[str, Any]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "resident": self.resident,
            "reserve_bytes": self.reserve_bytes,
            "max_batch": self.max_batch,
            "batch_scope": "speculative_verification_and_prefill_chunks",
            "draft_enabled": self.draft_enabled,
            "prefill_mode": self.prefill_mode,
            "placement": list(self.placement),
            "resources": self.resources.to_dict(),
            "assessment": copy.deepcopy(self.assessment),
            "reasons": list(self.reasons),
            "estimate_status": "preload_estimate",
            "optimality": "not_claimed",
        }


class NoMemoryPlan(RuntimeError):
    """No feasible estimated GPU/owned-RAM budget; never silently switch CPU-only."""


CPU_FRACTION_QUANTIZED = 0.58


def offload_cpu_fraction(unquantized: bool, override: str | None = None) -> float:
    """3.18: share of non-GPU layers computed by the CPU (the rest stream over PCIe).

    A streamed layer is read from RAM and then sent over the bus for every pass; a CPU layer
    is read from RAM once. For unquantized F16/F32 weights the bus copy is the larger cost, so
    all non-GPU layers go to the CPU (RTX 5060 8 GB laptop, qwen3-8b-f16: 2.35 tok/s vs 1.47 with streaming).
    Quantized layers keep the earlier measured split. An explicit RAMVI_CPU_FRAC wins.
    """
    if override is not None and override != "":
        value = float(override)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("RAMVI_CPU_FRAC must be within 0..1")
        return value
    return 1.0 if unquantized else CPU_FRACTION_QUANTIZED


LAYER_MATRICES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def unquantized_layers(gguf: Any) -> bool:
    """3.18: dense Qwen3 GGUF whose layer matrices are F16/F32 (metadata only)."""
    if gguf.meta.get("general.architecture") != "qwen3":
        return False
    return all(gguf.raw(f"blk.0.{name}.weight")[3] in ("F16", "F32") for name in LAYER_MATRICES)


def automatic_reserve(vram_available: int) -> int:
    if (
        isinstance(vram_available, bool)
        or not isinstance(vram_available, int)
        or vram_available < 0
    ):
        raise ValueError("vram_available must be nonnegative bytes")
    return max(64 * 2**20, min(GiB, vram_available // 8))


def placement(
    layers: int,
    resident: int,
    *,
    architecture: str,
    model_format: str,
    cpu: bool,
    draft_enabled: bool,
    cpu_fraction: float = 0.58,
    hybrid: bool | None = None,
) -> tuple[str, ...]:
    """hybrid: слои процессора — «hyb» (по умолчанию — когда включён угадыватель)."""
    if hybrid is None:
        hybrid = draft_enabled
    if architecture not in ("qwen3", "qwen3moe") or model_format not in ("gguf", "safetensors"):
        raise NotImplementedError("unsupported model architecture/format")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be positive")
    if isinstance(resident, bool) or not isinstance(resident, int) or not 0 <= resident <= layers:
        raise ValueError("resident outside layer range")
    if (
        isinstance(cpu_fraction, bool)
        or not isinstance(cpu_fraction, (int, float))
        or not math.isfinite(cpu_fraction)
        or not 0 <= cpu_fraction <= 1
    ):
        raise ValueError("cpu_fraction outside 0..1")
    if architecture == "qwen3moe":
        if model_format != "gguf" or not cpu and resident < layers:
            raise NotImplementedError("MoE supports GGUF resident/CPU layers, no streaming")
        return ("cpu",) * (layers - resident) + ("gpu",) * resident
    result = ["gpu"] * resident
    accumulator = 0.0
    for index in range(layers - resident):
        if model_format == "safetensors":
            result.append("cpu" if cpu and index % 2 else "stream")
        else:
            accumulator += cpu_fraction if cpu else 0.0
            if accumulator >= 1.0 - 1e-9:
                result.append("hyb" if hybrid else "cpu")
                accumulator -= 1.0
            else:
                result.append("stream")
    return tuple(result)


def choose_plan(
    layers: int,
    estimate: Callable[[list[str], bool, int], ResourceEstimate],
    *,
    vram_available: int,
    ram_available: int,
    maxt: int,
    architecture: str,
    model_format: str,
    cpu: bool,
    draft_requested: bool,
    options: PlanningOptions,
    requested_resident: int | None = None,
    cpu_fraction: float = 0.58,
    resident_cap: int | None = None,
    draft_with_offload: bool = True,
    hybrid_prefill: bool = False,
) -> PlanningDecision:
    """At most 2*(layers+1) candidates (3*(layers+1) with hybrid_prefill); fixed native
    buffers are still estimated at16.

    3.18: draft_with_offload=False skips draft candidates that leave layers outside the GPU
    (with a draft those layers become "hyb" and every verification batch streams them).

    B.6: hybrid_prefill=True — and without a draft, layers outside the GPU are first tried as
    "hyb": a single step stays on the CPU, but a long prompt streams them over the bus and
    computes them on the GPU (CPU layers dominated prefill time). If the extra pinned RAM and
    GPU KV do not fit, the same resident count is tried with plain "cpu" layers.

    Draft is considered only if a name was supplied. No speed benefit is inferred.
    Mapped-file working-set deficit can be allowed explicitly; owned RAM cannot.
    """
    for name, value in (("vram_available", vram_available), ("ram_available", ram_available)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be nonnegative integer bytes")
    if isinstance(maxt, bool) or not isinstance(maxt, int) or maxt < 1:
        raise ValueError("maxt must be positive")
    if requested_resident is not None and (
        isinstance(requested_resident, bool) or not 0 <= requested_resident <= layers
    ):
        raise ValueError("resident outside layer range")
    manual = options.mode == "manual" or requested_resident is not None
    if options.mode == "manual" and requested_resident is None:
        raise ValueError("manual planning requires resident")
    if options.draft == "on" and not draft_requested:
        raise ValueError("draft=on requires an explicit draft model")
    reserve = (
        automatic_reserve(vram_available)
        if options.reserve_bytes is None
        else options.reserve_bytes
    )
    batches = [b for b in BATCHES if b <= min(maxt, options.max_batch or 16)]
    batch = batches[-1]
    draft_modes = [draft_requested and options.draft != "off" and batch > 1]
    if options.draft == "on" and batch == 1:
        raise ValueError("draft=on requires max_batch >= 2")
    if draft_modes[0] and options.draft == "auto" and not manual:
        draft_modes.append(False)
    residents = (
        [requested_resident] if requested_resident is not None else list(range(layers, -1, -1))
    )
    if resident_cap is not None:
        residents = [r for r in residents if r is not None and r <= resident_cap]
    if architecture == "qwen3moe" and not cpu:
        residents = [r for r in residents if r == layers]
    attempts = 0
    for use_draft in draft_modes:
        for resident in residents:
            assert resident is not None
            # An explicit draft=on (or manual plan) wins over the rule, like RAMVI_CPU_FRAC.
            if (
                use_draft
                and not draft_with_offload
                and options.draft == "auto"
                and not manual
                and resident < layers
            ):
                continue
            hybrids = [use_draft]
            if (
                hybrid_prefill
                and not use_draft
                and cpu
                and resident < layers
                and architecture == "qwen3"
                and model_format == "gguf"
            ):
                hybrids = [True, False]
            for hybrid in hybrids:
                route = placement(
                    layers,
                    resident,
                    architecture=architecture,
                    model_format=model_format,
                    cpu=cpu,
                    draft_enabled=use_draft,
                    cpu_fraction=cpu_fraction,
                    hybrid=hybrid,
                )
                resources = estimate(list(route), use_draft, reserve)
                assessment = resources.assess(
                    vram_available=vram_available, ram_available=ram_available
                )
                attempts += 1
                required = resources.required
                fits_owned = required.vram <= vram_available and required.ram <= ram_available
                fits = fits_owned and (options.allow_paging or assessment["estimated_fit"])
                if manual or fits or hybrid is False or len(hybrids) == 1:
                    break
            if manual or fits:
                reasons = [
                    f"metadata_candidates_tested={attempts}",
                    "fixed native allocation buffers estimated at batch=16",
                    "prefill auto retains validated legacy default; blocked is opt-in",
                ]
                if not assessment["estimated_fit"]:
                    reasons.append(
                        "file working set exceeds RAM or manual budget; paging/OOM possible"
                    )
                if "hyb" in route and not use_draft:
                    reasons.append(
                        "hyb without draft: long prompts compute offloaded layers on GPU"
                    )
                if draft_requested and not use_draft:
                    reasons.append("draft disabled by policy, batch cap or available budget")
                # The full batch16 persistent allocation is already in the budget.
                # Do not penalize throughput from unmeasured residual-headroom heuristics;
                # shrink only for context/caller cap or an observed pre-output OOM.
                prefill = "blocked" if options.prefill == "blocked" else "legacy"
                if architecture == "qwen3moe" and prefill == "blocked":
                    raise NotImplementedError("MoE has only its existing chunked prefill")
                return PlanningDecision(
                    "manual" if manual else "auto",
                    resident,
                    reserve,
                    batch,
                    use_draft,
                    prefill,
                    route,
                    resources,
                    assessment,
                    tuple(reasons),
                )
    raise NoMemoryPlan(f"no estimated GPU/owned-RAM fit after {attempts} candidates")


class PlanJournal:
    """Per-session bounded JSONL; no prompts/tokens/secrets and no shared-file race."""

    def __init__(self, path: Path | None = None, *, limit: int = 256) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("journal limit must be positive")
        self.path = path or data_dir() / "planning" / f"session-{uuid.uuid4().hex}.jsonl"
        self.limit = limit
        self.error: str | None = None
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []

    def append(self, event: str, payload: dict[str, Any]) -> None:
        if event not in {"decision", "load_oom", "loaded", "actual", "error"}:
            raise ValueError("unknown planning event")
        row = {
            "version": 1,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "payload": copy.deepcopy(payload),
        }
        json.dumps(row, allow_nan=False)
        with self._lock:
            self._rows.append(row)
            self._rows = self._rows[-self.limit :]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=self.path.name + ".",
                    delete=False,
                ) as handle:
                    temporary = handle.name
                    for item in self._rows:
                        handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)

    def safe_append(self, event: str, payload: dict[str, Any]) -> bool:
        try:
            self.append(event, payload)
            return True
        except OSError as exc:
            self.error = type(exc).__name__
            return False


class PlanningController:
    """Contains only numeric/copyable decisions; never retains engine/tensor owners."""

    def __init__(self, decision: PlanningDecision, journal: PlanJournal) -> None:
        self.decision = decision
        self.journal = journal
        self.last_place = decision.placement
        self.journal_error: str | None = None

    def observe(self, engine: Any, outcome: str = "completed") -> None:
        actual = tuple(getattr(engine, "place", ()))
        report = self.decision.to_dict()
        report["actual_placement"] = list(actual)
        report["actual_resident"] = actual.count("gpu")
        report["max_batch"] = batch_limit(engine)
        if actual != self.decision.placement:
            report["estimate_status"] = "stale_after_runtime_change"
        if actual != self.last_place or outcome != "completed":
            self.journal.safe_append(
                "actual",
                {"outcome": outcome, "placement": list(actual), "max_batch": batch_limit(engine)},
            )
            self.last_place = actual
        self.journal_error = self.journal.error
        report["journal_error"] = self.journal_error
        engine.placement_decision = report


def batch_limit(engine: Any) -> int:
    value = getattr(engine, "max_batch", 16)
    if isinstance(value, bool) or not isinstance(value, int) or value not in BATCHES:
        raise ValueError("runtime max_batch must be 1, 2, 4, 8 or 16")
    return int(value)


def prefill_enabled(engine: Any, size: int, legacy: Callable[[int], bool]) -> bool:
    mode = getattr(engine, "prefill_mode", None)
    if mode is None:
        return legacy(size)
    if mode not in ("legacy", "blocked"):
        raise ValueError("runtime prefill mode must be legacy or blocked")
    return mode == "blocked" and size >= 256


def apply_decision(engine: Any, decision: PlanningDecision, journal: PlanJournal) -> None:
    engine.max_batch = decision.max_batch
    engine.prefill_mode = decision.prefill_mode
    engine.allow_oom_demote = decision.mode != "manual"
    engine.planning_controller = PlanningController(decision, journal)
    engine.planning_controller.observe(engine)


def options_dict(options: PlanningOptions) -> dict[str, Any]:
    return asdict(options)
