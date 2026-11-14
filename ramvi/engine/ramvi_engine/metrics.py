"""Метрики генерации: честные области измерения и bounded RSS sampling."""

import copy
import ctypes
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from typing import Any


def process_rss() -> int | None:
    """Текущий RSS/working set процесса; не lifetime high-water mark."""
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "peak_working_set",
                        "working_set",
                        "peak_paged",
                        "paged",
                        "peak_nonpaged",
                        "nonpaged",
                        "pagefile",
                        "peak_pagefile",
                    )
                ]

            data = Counters()
            data.cb = ctypes.sizeof(data)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(data), data.cb
            ):
                return None
            return int(data.working_set)
        if sys.platform == "darwin":

            class Info(ctypes.Structure):
                _fields_ = [
                    ("virtual", ctypes.c_uint64),
                    ("resident", ctypes.c_uint64),
                    ("resident_max", ctypes.c_uint64),
                    ("times", ctypes.c_int32 * 4),
                    ("policy", ctypes.c_int32),
                    ("suspend", ctypes.c_int32),
                ]

            lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            task = ctypes.c_uint32.in_dll(lib, "mach_task_self_").value
            data = Info()
            count = ctypes.c_uint32(ctypes.sizeof(data) // 4)
            if lib.task_info(task, 20, ctypes.byref(data), ctypes.byref(count)) != 0:
                return None
            return int(data.resident)
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError, IndexError):
        return None


def plan_snapshot(engine: Any) -> dict[str, Any]:
    place = getattr(engine, "place", None)
    layers = list(place) if isinstance(place, (list, tuple)) else None
    return {
        "source": "engine.place",
        "layers": layers,
        "counts": dict(Counter(layers)) if layers is not None else None,
        "context_tokens": getattr(engine, "maxt", None),
        "planning_decision": copy.deepcopy(getattr(engine, "placement_decision", None)),
    }


class MemoryMonitor:
    """RAM-пик выборочный; VRAM-пик — CUDA allocator всего процесса на устройстве."""

    INTERVAL = 0.05

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._torch: Any = None
        self._device: Any = None
        self._peak_rss: int | None = None
        self._samples = 0
        self._closed = False
        self.data: dict[str, Any] = {
            "ram_scope": "sampled_process_rss",
            "ram_sample_interval_ms": self.INTERVAL * 1000,
            "ram_start_bytes": None,
            "ram_end_bytes": None,
            "peak_ram_bytes": None,
            "vram_scope": "process_device_allocator",
            "vram_device": None,
            "vram_start_allocated_bytes": None,
            "vram_end_allocated_bytes": None,
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
            "driver_memory_included": False,
        }

    def _sample(self) -> int | None:
        rss = process_rss()
        if rss is not None:
            self._samples += 1
            self._peak_rss = max(self._peak_rss or 0, rss)
        return rss

    def start(self, engine: Any) -> None:
        self.data["ram_start_bytes"] = self._sample()
        if self._peak_rss is not None:

            def sample() -> None:
                while not self._stop.wait(self.INTERVAL):
                    self._sample()

            self._thread = threading.Thread(target=sample, name="ramvi-rss", daemon=True)
            try:
                self._thread.start()
            except RuntimeError:
                self._thread = None
                self.data["ram_sampler_error"] = True
        # Не импортируем torch и не инициализируем CUDA для лёгкого/пустого движка.
        device = getattr(getattr(engine, "pos", None), "device", None)
        torch = sys.modules.get("torch")
        if torch is not None and getattr(device, "type", None) == "cuda":
            try:
                self._device = str(device)
                torch.cuda.reset_peak_memory_stats(self._device)
                self.data["vram_start_allocated_bytes"] = torch.cuda.memory_allocated(self._device)
                self.data["vram_device"] = self._device
                self._torch = torch
            except (RuntimeError, ValueError, AttributeError):
                self.data["vram_probe_error"] = True

    def stop(self) -> dict[str, Any]:
        if self._closed:
            return self.data
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.data["ram_end_bytes"] = self._sample()
        self.data["peak_ram_bytes"] = self._peak_rss
        self.data["ram_samples"] = self._samples
        if self._torch is not None:
            try:
                cuda = self._torch.cuda
                self.data["vram_end_allocated_bytes"] = cuda.memory_allocated(self._device)
                self.data["peak_vram_allocated_bytes"] = cuda.max_memory_allocated(self._device)
                self.data["peak_vram_reserved_bytes"] = cuda.max_memory_reserved(self._device)
            except (RuntimeError, ValueError, AttributeError):
                self.data["vram_probe_error"] = True
        return self.data


def number(stats: dict[str, Any], key: str) -> int | float | None:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


class GenerationMetrics:
    def __init__(
        self, generation_id: str, started: float | None = None, model: str | None = None
    ) -> None:
        self.id = generation_id
        self.model = model
        self.started_at = time.time()
        self.started = time.perf_counter() if started is None else started
        self.first_output_ms: float | None = None
        self.first_text_ms: float | None = None

    def observe(self, kind: str, value: Any) -> None:
        now = (time.perf_counter() - self.started) * 1000
        visible = kind in ("text", "reasoning") and bool(value)
        if kind == "end" and value.get("tool_calls"):
            visible = True
        if visible and self.first_output_ms is None:
            self.first_output_ms = now
        if kind == "text" and value and self.first_text_ms is None:
            self.first_text_ms = now

    def finish(
        self,
        stats: dict[str, Any],
        final: dict[str, Any] | None,
        memory: dict[str, Any],
        plan: dict[str, Any],
        outcome: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        count = final["gen_n"] if final is not None else number(stats, "gen_n")
        decode = max(0, count - 1) if count is not None else None
        decode_ms = number(stats, "gen_ms")
        drafted, accepted = number(stats, "drafted"), number(stats, "accepted")
        return {
            "schema_version": 1,
            "generation_id": self.id,
            "model": self.model,
            "started_at_unix_s": self.started_at,
            "finished_at_unix_s": time.time(),
            "outcome": outcome,
            "timing_scope": "generation_worker_including_model_wait",
            "ttft_ms": self.first_output_ms,
            "first_text_ms": self.first_text_ms,
            "engine_first_token_ms": number(stats, "first_token_ms") if count else None,
            "prefill_ms": number(stats, "prompt_ms"),
            "decode_ms": decode_ms,
            "worker_total_ms": (time.perf_counter() - self.started) * 1000,
            "input_tokens": final["prompt_n"] if final is not None else None,
            "prefill_tokens": number(stats, "prompt_n"),
            "output_tokens": count,
            "decode_tokens": decode,
            "decode_tokens_per_second": decode * 1000 / decode_ms if decode and decode_ms else None,
            "draft": {
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "acceptance_rate": accepted / drafted if accepted is not None and drafted else None,
                "calls": number(stats, "draft_calls"),
                "skips": number(stats, "draft_skips"),
                "probes": number(stats, "draft_probes"),
                "passes": number(stats, "passes"),
                "single_steps": number(stats, "steps1"),
            },
            "memory": copy.deepcopy(memory),
            "plan": copy.deepcopy(plan),
            "error_type": error_type,
        }


class MetricsStore:
    """Только последний числовой снимок; тексты/промпты не сохраняются."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._counts: Counter[str] = Counter()
        self._last: dict[str, Any] | None = None

    def begin(self, model: str | None = None) -> GenerationMetrics:
        with self._lock:
            self._active += 1
        return GenerationMetrics(uuid.uuid4().hex, model=model)

    def finish(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._active -= 1
            self._counts[record["outcome"]] += 1
            self._last = copy.deepcopy(record)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "active": self._active,
                "counts": dict(self._counts),
                "last": copy.deepcopy(self._last),
            }


def with_metrics(body: dict[str, Any], final: dict[str, Any] | None) -> dict[str, Any]:
    """Расширение API в namespace RAMVI, без изменения стандартных usage-полей."""
    if final is not None:
        body["ramvi_metrics"] = final.get("metrics")
    return body


def with_error_metrics(body: dict[str, Any], error: BaseException) -> dict[str, Any]:
    record = getattr(error, "ramvi_metrics", None)
    if isinstance(record, dict):
        body["ramvi_metrics"] = copy.deepcopy(record)
    return body
