"""4.7 AUTO-05: контрольный прогон модели на этом компьютере.

После скачивания (или по запросу) модель загружается так же, как для чата
(контекст профиля задачи), и отвечает на короткий прогрев и несколько
синтетических запросов: коротких и длинного — около половины контекста. Меряются
время до первого токена и ток/с (на стороне сервиса, от запроса до ответа), пики
RAM и VRAM и план слоёв (из `ramvi_metrics` движка, если он их прислал), page
faults процесса движка.

Результат сохраняется в `model_profiles` под ключом модель + отпечаток железа +
контекст; в нём же SHA-256 файла — заменённый файл свой замер теряет. Отпечаток
железа (`hardware_key`) — только локально, никуда не отправляется.

- **Нестабильно** (ошибка, или скорость коротких запросов гуляет больше чем на 25 %)
  — модель получает пометку и опускается в выборе (4.6).
- **Нехватка памяти во время генерации** — безопасный откат: модель загружается
  заново с меньшим числом слоёв в видеопамяти (×0,75) и прогон повторяется. Удачный
  безопасный план сохраняется, и следующие загрузки этой модели при том же железе и
  контексте берут его (`resident_for`).

Модель общая: прогон занимает её слот (`SharedLLM.exclusive`) — идущий ответ не
обрывается. Если до прогона была загружена другая модель, она возвращается.
"""

import hashlib
import json
import statistics
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_storage.repositories import Model, ModelProfile, Models, NotFound

from ramvi_service.engines import EngineError, EngineManager, resolve_model
from ramvi_service.hardware import HardwareProfile
from ramvi_service.resources import SharedLLM

WARMUP = "Reply with one word: ready"
# Просьбы о тексте ~100 слов: ответ дольше 64 токенов, скорость меряется на полной длине.
SHORT = (
    "Write a short story about a cat in about 100 words.",
    "Explain how a refrigerator works in about 100 words.",
)
OUTPUT_TOKENS = 64
UNSTABLE_SPREAD = 0.25  # разброс скорости коротких запросов (max−min)/медиана
SAFE_FACTOR = 0.75  # при нехватке памяти — столько от прежних слоёв в видеопамяти
PRINCIPAL = "local_user"


class OutOfMemory(RuntimeError):
    """Нехватка памяти при генерации; `plan` — план движка в этот момент (если пришёл)."""

    def __init__(self, message: str, plan: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.plan = plan


def hardware_key(hw: HardwareProfile) -> str:
    """Отпечаток железа для замеров: процессор, ядра, RAM, видеокарты и драйвер."""
    parts = {
        "cpu": hw.cpu_name,
        "cores": hw.cores_logical,
        "ram_gb": round((hw.ram_total_mb or 0) / 1024),
        "gpus": sorted(
            (g.name, g.vendor, g.vram_total_mb or 0, g.driver or "")
            for g in hw.gpus
            if g.vendor == "NVIDIA"
        ),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def long_prompt(context: int) -> str:
    """Синтетический текст примерно на половину контекста (≈ 2 токена на «слово»)."""
    words = max(16, min(context // 2, 4096) // 2)
    return "Summarize in one sentence: " + " ".join(f"item{i}" for i in range(words))


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(pid: int) -> list[int]:
    """Процесс и все его потомки (на Windows движок запускается через процесс-привратник)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("usage", wintypes.DWORD),
                ("pid", wintypes.DWORD),
                ("heap", ctypes.c_size_t),
                ("module", wintypes.DWORD),
                ("threads", wintypes.DWORD),
                ("parent", wintypes.DWORD),
                ("priority", ctypes.c_long),
                ("flags", wintypes.DWORD),
                ("name", ctypes.c_wchar * 260),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        parents: dict[int, int] = {}
        try:
            entry = Entry()
            entry.size = ctypes.sizeof(Entry)
            ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                parents[int(entry.pid)] = int(entry.parent)
                ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
    else:
        parents = {}
        for folder in Path("/proc").glob("[0-9]*"):
            try:
                parents[int(folder.name)] = int(
                    (folder / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[1]
                )
            except (OSError, ValueError, IndexError):
                continue
    found, frontier = [pid], [pid]
    while frontier:
        current = frontier.pop()
        children = [
            child for child, parent in parents.items() if parent == current and child != pid
        ]
        found += children
        frontier += children
    return found


def page_faults(pid: int | None) -> int | None:
    """Page faults дерева процессов движка; на этой ОС недоступно — None."""
    if pid is None or sys.platform not in ("win32",) and not sys.platform.startswith("linux"):
        return None
    try:
        counts = [_faults(p) for p in _tree(pid)]
    except (OSError, ValueError, AttributeError):
        return None
    known = [c for c in counts if c is not None]
    return sum(known) if known else None


def _faults(pid: int) -> int | None:
    try:
        if sys.platform.startswith("linux"):
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            return int(fields[9])  # majflt — чтения с диска
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f", "g", "h")
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED | VM_READ
            if not handle:
                return None
            try:
                data = Counters()
                data.cb = ctypes.sizeof(data)
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                psapi.GetProcessMemoryInfo.argtypes = [
                    wintypes.HANDLE,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                ]
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(data), data.cb):
                    return None
                return int(data.faults)  # мягкие + жёсткие
            finally:
                kernel.CloseHandle(handle)
    except (OSError, ValueError, IndexError, AttributeError):
        return None
    return None


class ControlRun:
    """Прогоны по одному за раз; состояние каждого — в `status()`."""

    def __init__(
        self,
        engines: EngineManager,
        llm: SharedLLM,
        models: Models,
        hardware: Callable[[], HardwareProfile],
        context: Callable[[], int],
        *,
        timeout: float = 600.0,
        output_tokens: int = OUTPUT_TOKENS,
    ) -> None:
        self.engines, self.llm, self.models = engines, llm, models
        self.hardware, self.context, self.timeout = hardware, context, timeout
        self.output_tokens = output_tokens
        self._owner = llm.authority.issue("core", PRINCIPAL, "control_run")
        llm.retain(self._owner)
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._threads: list[threading.Thread] = []

    # --- запуск --------------------------------------------------------------------

    def start(self, model_id: str) -> dict[str, Any]:
        """Запустить в фоне; уже идёт — то же состояние."""
        self.models.get(model_id)  # NotFound — нет такой модели
        with self._lock:
            current = self._state.get(model_id)
            if current is not None and current["state"] in ("queued", "running"):
                return dict(current)
            self._state[model_id] = {"state": "queued", "model_id": model_id}
            thread = threading.Thread(
                target=self._run_safely, args=(model_id,), name="ramvi-control-run", daemon=True
            )
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()
        return self.status(model_id)

    def status(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.get(model_id) or {"state": "none", "model_id": model_id})

    def wait(self, timeout: float = 30) -> None:
        for thread in list(self._threads):
            thread.join(timeout)

    def _set(self, model_id: str, **values: Any) -> None:
        with self._lock:
            self._state[model_id] = {"model_id": model_id, **values}

    def _run_safely(self, model_id: str) -> None:
        try:
            result = self.run(model_id)
            self._set(model_id, state="done", result=result)
        except Exception as error:  # поток прогона не роняет сервис
            code = getattr(error, "code", "control.failed")
            params = getattr(error, "params", {"detail": type(error).__name__})
            self._set(model_id, state="failed", error={"code": code, "params": params})

    # --- сам прогон ----------------------------------------------------------------

    def run(self, model_id: str) -> dict[str, Any]:
        """Прогнать модель и сохранить результат; блокирует до конца."""
        self._set(model_id, state="running")
        model = self._with_sha(self.models.get(model_id))
        hw = self.hardware()
        key, context = hardware_key(hw), self.context()
        with self.llm.exclusive(self._owner, timeout=self.timeout):
            previous = self.engines.model_id
            try:
                result = self._measure(model, context, resident=None)
                if result.get("oom"):
                    safe = self._safe_resident(result)
                    retry = self._measure(model, context, resident=safe)
                    retry["rolled_back_from"] = result.get("resident")
                    result = retry
            finally:
                if previous is not None and previous != model_id:
                    try:
                        self.engines.ensure(previous, timeout=self.timeout)
                    except EngineError:
                        pass  # прежняя модель вернётся при следующем сообщении
        result.update(context=context, sha256=model.sha256, hardware_key=key)
        if not result.get("oom") and result.get("runs"):
            self.models.add_profile(
                ModelProfile(
                    model_id=model.id,
                    hardware_hash_local=key,
                    context=context,
                    plan=result,
                    tested_at=time.time(),
                    ttft=result["ttft_s"],
                    tokps=result["tok_s"],
                    peak_ram=result.get("peak_ram"),
                    peak_vram=result.get("peak_vram"),
                )
            )
        return result

    def _with_sha(self, model: Model) -> Model:
        """Ключ замера — хеш файла: у импортированной модели считается один раз."""
        if model.sha256:
            return model
        path = resolve_model(model, self.engines.model_root).path
        updated = replace(model, sha256=file_sha256(path))
        self.models.upsert(updated)
        return updated

    @staticmethod
    def _safe_resident(result: dict[str, Any]) -> int:
        resident = result.get("resident")
        layers = result.get("layers") or 0
        base = resident if isinstance(resident, int) else layers
        return max(0, int(base * SAFE_FACTOR))

    def _measure(self, model: Model, context: int, resident: int | None) -> dict[str, Any]:
        """Загрузить (при `resident` — с этим числом слоёв в VRAM) и прогнать запросы."""
        self.engines.unload()  # загрузка с нуля: план под этот прогон
        self.engines.ensure(model.id, timeout=self.timeout, resident=resident)
        pid = self.engines.engine_pid()
        faults_before = page_faults(pid)
        runs: list[dict[str, Any]] = []
        plan: dict[str, Any] | None = None
        peaks = {"ram": 0, "vram": 0}
        prompts = [WARMUP, SHORT[0], long_prompt(context), SHORT[1]]
        for index, prompt in enumerate(prompts):
            try:
                one = self._ask(prompt, 8 if index == 0 else self.output_tokens)
            except OutOfMemory as error:
                plan = error.plan or plan
                return {
                    "oom": True,
                    "resident": resident if resident is not None else _resident(plan),
                    "layers": _layers(plan),
                }
            metrics = one.pop("metrics") or {}
            plan = metrics.get("plan") or plan
            memory = metrics.get("memory") or {}
            peaks["ram"] = max(peaks["ram"], int(memory.get("peak_ram_bytes") or 0))
            vram = memory.get("peak_vram_reserved_bytes") or memory.get("peak_vram_allocated_bytes")
            peaks["vram"] = max(peaks["vram"], int(vram or 0))
            if index:  # прогрев не считается
                runs.append({"kind": "long" if index == 2 else "short", **one})
        faults_after = page_faults(pid)
        return _summary(runs, plan, peaks, faults_before, faults_after, resident)

    def _ask(self, prompt: str, tokens: int) -> dict[str, Any]:
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0,
        }
        provider = self.engines.provider()
        started = time.perf_counter()
        first: float | None = None
        end: dict[str, Any] = {}
        try:
            for chunk in provider.stream(request, threading.Event(), with_metrics=True):
                if chunk["type"] in ("text", "reasoning") and first is None:
                    first = time.perf_counter()
                elif chunk["type"] == "end":
                    end = chunk
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                metrics = getattr(error, "ramvi_metrics", None) or {}
                raise OutOfMemory(str(error), metrics.get("plan")) from error
            raise
        finished = time.perf_counter()
        tokens_out = int((end.get("usage") or {}).get("completion_tokens") or 0)
        decode = finished - (first or finished)
        return {
            "ttft_s": (first or finished) - started,
            "tokens": tokens_out,
            "tok_s": (tokens_out - 1) / decode if tokens_out > 1 and decode > 0 else None,
            "metrics": end.get("metrics"),
        }

    # --- для оценки и загрузки -----------------------------------------------------

    def latest(self, model: Model, hw: HardwareProfile, context: int) -> dict[str, Any] | None:
        """Сохранённый результат для этой модели, железа и контекста (файл тот же)."""
        try:
            found = self.models.latest_profile(model.id, hardware_key(hw))
        except NotFound:
            return None
        if found is None or found.context != context:
            return None
        if model.sha256 and found.plan.get("sha256") not in (None, model.sha256):
            return None  # файл заменён — замер чужой
        return found.plan

    def resident_for(self, model: Model, context: int) -> int | None:
        """Безопасный план после отката (если был) — для `EngineManager.resident_for`."""
        found = self.latest(model, self.hardware(), context)
        if found is None or found.get("rolled_back_from") is None:
            return None
        resident = found.get("resident")
        return resident if isinstance(resident, int) else None


def _resident(plan: dict[str, Any] | None) -> int | None:
    counts = (plan or {}).get("counts") or {}
    value = counts.get("gpu")
    return int(value) if isinstance(value, int) else None


def _layers(plan: dict[str, Any] | None) -> int | None:
    counts = (plan or {}).get("counts") or {}
    total = sum(v for v in counts.values() if isinstance(v, int))
    return total or None


def _summary(
    runs: list[dict[str, Any]],
    plan: dict[str, Any] | None,
    peaks: dict[str, int],
    faults_before: int | None,
    faults_after: int | None,
    resident: int | None,
) -> dict[str, Any]:
    speeds = [r["tok_s"] for r in runs if r["tok_s"]]
    short = [r["tok_s"] for r in runs if r["kind"] == "short" and r["tok_s"]]
    median = statistics.median(speeds) if speeds else None
    spread = (max(short) - min(short)) / statistics.median(short) if len(short) >= 2 else 0.0
    stable = bool(speeds) and len(speeds) == len(runs) and spread <= UNSTABLE_SPREAD
    faults = (
        faults_after - faults_before
        if faults_before is not None and faults_after is not None
        else None
    )
    return {
        "oom": False,
        "runs": runs,
        "tok_s": median,
        "tok_s_range": [min(speeds), max(speeds)] if speeds else None,
        "ttft_s": statistics.median(r["ttft_s"] for r in runs) if runs else None,
        "spread": round(spread, 3),
        "stable": stable,
        "peak_ram": peaks["ram"] or None,
        "peak_vram": peaks["vram"] or None,
        "page_faults": faults,
        "resident": resident if resident is not None else _resident(plan),
        "layers": _layers(plan),
        "plan_counts": (plan or {}).get("counts"),
    }
