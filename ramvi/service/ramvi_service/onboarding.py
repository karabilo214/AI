"""3.6 (UI-01): мастер первого запуска — проверка компьютера, импорт и выбор модели.

- `system()` — профиль железа (4.1) и список проверок со статусом ok/warn/fail/unknown.
  Бэкенд (torch + CUDA) проверяется в отдельном процессе: сервис сам torch не
  загружает (лёгкий wheel), результат кешируется на время работы сервиса.
- `import_model(path)` — подключить файл GGUF без копирования: файл остаётся на
  месте, в базе хранится путь. Заголовок читается стандартной библиотекой и
  проверяется до регистрации (архитектура, которую умеет движок).
- Выбор модели и признак «мастер пройден» хранятся в `settings.json` папки
  данных: после перезапуска мастер не показывается, новые чаты берут модель.

Пользовательских строк нет: проверки и ошибки — ключи перевода с параметрами.
"""

import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

from ramvi_storage.repositories import Model, Models, NotFound

from ramvi_service import settings as app_settings
from ramvi_service.chat import ChatError
from ramvi_service.hardware import HardwareProfile, collect

GIB_MB = 1024
DISK_LOW_MB = 2 * GIB_MB  # меньше — история и база ещё влезут, модель уже нет
SUPPORTED_ARCHITECTURES = ("qwen3",)  # GGUF-движок: ramvi_engine.core.gengine

# llama.cpp LLAMA_FTYPE: общий тип квантования файла (general.file_type).
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
}  # fmt: skip

_PROBE = """
import json
r = {"torch": None, "cuda": False, "cuda_version": None, "devices": [], "error": None}
try:
    import torch
    r["torch"] = torch.__version__
    r["cuda"] = bool(torch.cuda.is_available())
    r["cuda_version"] = torch.version.cuda
    if r["cuda"]:
        r["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as error:
    r["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(r))
"""


def probe_backends(timeout: float = 120.0) -> dict[str, Any]:
    """torch и CUDA так, как их увидит движок (тот же интерпретатор)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        value = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(value, dict):
            raise ValueError("ожидался объект JSON")
        return value
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
        return {"torch": None, "cuda": False, "error": f"{type(error).__name__}: {error}"}


def _gb(mb: int | None) -> float | None:
    return None if mb is None else round(mb / GIB_MB, 1)


def _check(item: str, status: str, code: str, **params: Any) -> dict[str, Any]:
    return {"id": item, "status": status, "code": code, "params": params}


def checks(hw: HardwareProfile, backends: dict[str, Any]) -> list[dict[str, Any]]:
    """Проверки для экрана мастера. Порядок: ОС, CPU, RAM, GPU, диск, бэкенд."""
    result = []
    system = hw.os.split()[0] if hw.os else ""
    if system in ("Windows", "Linux"):
        result.append(_check("os", "ok", "system.os", name=hw.os, arch=hw.arch))
    else:
        result.append(_check("os", "warn", "system.os_no_cuda", name=hw.os, arch=hw.arch))
    if hw.cpu_name or hw.cores_logical:
        cores = hw.cores_physical or hw.cores_logical
        result.append(
            _check(
                "cpu",
                "ok",
                "system.cpu",
                name=hw.cpu_name or "",
                cores=cores,
                threads=hw.cores_logical,
            )
        )
    else:
        result.append(_check("cpu", "unknown", "system.cpu_unknown"))
    if hw.ram_total_mb is not None:
        result.append(
            _check(
                "ram",
                "ok",
                "system.ram",
                total_gb=_gb(hw.ram_total_mb),
                available_gb=_gb(hw.ram_available_mb) if hw.ram_available_mb is not None else "?",
            )
        )
    else:
        result.append(_check("ram", "unknown", "system.ram_unknown"))
    nvidia = [g for g in hw.gpus if g.vendor == "NVIDIA"]
    if nvidia:
        best = max(nvidia, key=lambda g: g.vram_total_mb or 0)
        result.append(
            _check(
                "gpu",
                "ok",
                "system.gpu",
                name=best.name,
                vram_gb=_gb(best.vram_total_mb) if best.vram_total_mb else "?",
                free_gb=_gb(best.vram_free_mb) if best.vram_free_mb is not None else "?",
            )
        )
    elif hw.gpus:
        names = ", ".join(g.name for g in hw.gpus)
        result.append(_check("gpu", "fail", "system.gpu_no_nvidia", names=names))
    else:
        result.append(_check("gpu", "fail", "system.gpu_none"))
    if hw.disk_free_mb is None:
        result.append(_check("disk", "unknown", "system.disk_unknown"))
    else:
        status = "warn" if hw.disk_free_mb < DISK_LOW_MB else "ok"
        code = "system.disk_low" if status == "warn" else "system.disk"
        result.append(_check("disk", status, code, free_gb=_gb(hw.disk_free_mb), path=hw.disk_path))
    if backends.get("cuda"):
        result.append(
            _check(
                "backend",
                "ok",
                "system.backend_cuda",
                cuda=backends.get("cuda_version") or "?",
                torch=backends.get("torch") or "?",
            )
        )
    elif backends.get("torch"):
        # Совет про драйвер имеет смысл, только если карта NVIDIA есть.
        code = "system.backend_no_cuda" if nvidia else "system.backend_no_gpu"
        result.append(_check("backend", "fail", code, torch=backends["torch"]))
    else:
        result.append(
            _check("backend", "fail", "system.backend_missing", detail=backends.get("error") or "")
        )
    return result


# --- GGUF -------------------------------------------------------------------------------

_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9
_WANTED = ("general.architecture", "general.file_type", "general.name")
MAX_KV = 1_000_000
MAX_STRING = 64 * 2**20


class NotGGUF(ValueError):
    """Файл не GGUF или заголовок повреждён."""


def _read(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise NotGGUF("файл обрывается внутри заголовка")
    return data


def _string(file: BinaryIO) -> str:
    (size,) = struct.unpack("<Q", _read(file, 8))
    if size > MAX_STRING:
        raise NotGGUF("слишком длинная строка в заголовке")
    return _read(file, size).decode("utf-8", errors="replace")


def _value(file: BinaryIO, kind: int) -> Any:
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(file, struct.calcsize(fmt)))[0]
    if kind == _STRING:
        return _string(file)
    if kind == _ARRAY:
        item, count = struct.unpack("<IQ", _read(file, 12))
        if count > 64 * 2**20:
            raise NotGGUF("слишком длинный массив в заголовке")
        if item in _SCALARS:  # пропустить, не разбирая
            file.seek(struct.calcsize(_SCALARS[item]) * count, os.SEEK_CUR)
            return None
        for _ in range(count):
            _value(file, item)
        return None
    raise NotGGUF(f"неизвестный тип значения {kind}")


def read_gguf_header(path: Path) -> dict[str, Any]:
    """Архитектура, тип квантования и имя из заголовка GGUF (без numpy и torch)."""
    with path.open("rb") as file:
        if _read(file, 4) != b"GGUF":
            raise NotGGUF("нет сигнатуры GGUF")
        version, _tensors, count = struct.unpack("<IQQ", _read(file, 20))
        if version not in (2, 3) or count > MAX_KV:
            raise NotGGUF(f"версия GGUF {version} не поддерживается")
        found: dict[str, Any] = {}
        for _ in range(count):
            key = _string(file)
            (kind,) = struct.unpack("<I", _read(file, 4))
            value = _value(file, kind)
            if key in _WANTED:
                found[key] = value
                if len(found) == len(_WANTED):
                    break
    architecture = found.get("general.architecture")
    if not isinstance(architecture, str):
        raise NotGGUF("в заголовке нет general.architecture")
    file_type = found.get("general.file_type")
    quant = FILE_TYPES.get(file_type, f"type {file_type}") if isinstance(file_type, int) else None
    name = found.get("general.name")
    return {
        "architecture": architecture,
        "quant": quant,
        "name": name if isinstance(name, str) else None,
    }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-.")
    return slug[:64] or "model"


# --- мастер -------------------------------------------------------------------------------


class Onboarding:
    def __init__(
        self,
        data_dir: Path,
        models: Models,
        *,
        hardware: Callable[[Path], HardwareProfile] = collect,
        backends: Callable[[], dict[str, Any]] = probe_backends,
    ) -> None:
        self.data_dir, self.models = Path(data_dir), models
        self._hardware, self._probe = hardware, backends
        self._backends: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def system(self) -> dict[str, Any]:
        hw = self._hardware(self.data_dir)
        with self._lock:  # первая проверка бэкенда долгая; повторные — из кеша
            if self._backends is None:
                self._backends = self._probe()
            backends = self._backends
        return {"hardware": hw.to_dict(), "backends": backends, "checks": checks(hw, backends)}

    def _settings(self) -> app_settings.Settings:
        return app_settings.load(self.data_dir)

    def state(self) -> dict[str, Any]:
        current = self._settings()
        return {"done": current.onboarding_done, "default_model": self.default_model()}

    def default_model(self) -> str | None:
        """Сохранённый выбор, если такая модель ещё установлена."""
        model_id = self._settings().default_model
        if model_id is None:
            return None
        try:
            self.models.get(model_id)
        except NotFound:
            return None
        return model_id

    def complete(self, model_id: str | None) -> dict[str, Any]:
        """Мастер пройден: с выбранной моделью или пропуском (модель выберут позже)."""
        if model_id is not None:
            self._require(model_id)
        current = self._settings()
        app_settings.save(
            self.data_dir,
            replace(current, onboarding_done=True, default_model=model_id or current.default_model),
        )
        return self.state()

    def remember(self, model_id: str) -> None:
        """Последний явный выбор модели — модель новых чатов после перезапуска."""
        current = self._settings()
        if current.default_model != model_id:
            app_settings.save(self.data_dir, replace(current, default_model=model_id))

    def _require(self, model_id: str) -> None:
        try:
            self.models.get(model_id)
        except NotFound:
            raise ChatError("chat.model_not_found", model_id=model_id) from None

    def import_model(self, path_text: str) -> dict[str, Any]:
        """Подключить локальный GGUF без копирования; повтор того же файла — та же запись."""
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            raise ChatError("import.not_absolute", path=path_text)
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            raise ChatError("import.not_found", path=path_text) from None
        except OSError as error:
            raise ChatError("import.no_access", path=path_text, detail=str(error)) from None
        if not path.is_file():
            raise ChatError("import.not_a_file", path=str(path))
        for existing in self.models.all():
            if existing.path == str(path):
                return model_json(existing)
        try:
            header = read_gguf_header(path)
            size = path.stat().st_size
        except NotGGUF as error:
            raise ChatError("import.not_gguf", path=str(path), detail=str(error)) from None
        except PermissionError as error:
            raise ChatError("import.no_access", path=str(path), detail=str(error)) from None
        except OSError as error:
            raise ChatError("import.no_access", path=str(path), detail=str(error)) from None
        if header["architecture"] not in SUPPORTED_ARCHITECTURES:
            raise ChatError(
                "import.unsupported_architecture",
                path=str(path),
                architecture=header["architecture"],
                supported=", ".join(SUPPORTED_ARCHITECTURES),
            )
        base = _slug(path.stem)
        taken = {m.id for m in self.models.all()}
        model_id, n = base, 2
        while model_id in taken:
            model_id, n = f"{base}-{n}", n + 1
        model = self.models.upsert(
            Model(
                model_id,
                "local",
                header["architecture"],
                str(path),
                time.time(),
                quant=header["quant"],
                bytes=size,
                capabilities={"format": "gguf", "imported": True, "copied": False},
            )
        )
        return model_json(model)


def model_json(m: Model) -> dict[str, Any]:
    return {
        "id": m.id,
        "architecture": m.architecture,
        "quant": m.quant,
        "bytes": m.bytes,
        "capabilities": m.capabilities,
    }
