"""8.7: ядра Triton без ptxas — PTX компилирует драйвер NVIDIA.

ptxas.exe нет в списке распространяемых файлов CUDA EULA (приложение A), поэтому в комплект он
не входит. Triton по умолчанию собирает PTX в cubin через ptxas; здесь вместо cubin загрузчику
отдаётся сам PTX: cuModuleLoadData принимает PTX и компилирует его JIT-компилятором драйвера
(результат драйвер кеширует). Клиенту нужен только драйвер NVIDIA с CUDA 13.x, без CUDA Toolkit.
Включается, когда ptxas нет (или RAMVI_DRIVER_JIT=1); RAMVI_DRIVER_JIT=0 — всегда ptxas.
"""

import ctypes
import os
import sys
from pathlib import Path


def driver_cuda_version() -> str:
    """Версия CUDA, которую поддерживает установленный драйвер (cuDriverGetVersion), «13.2»."""
    lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    value = ctypes.c_int()
    if lib.cuDriverGetVersion(ctypes.byref(value)) != 0:
        raise RuntimeError("cuDriverGetVersion не ответил")
    return f"{value.value // 1000}.{value.value % 1000 // 10}"


def ptxas_available() -> bool:
    try:
        from triton import knobs

        path = knobs.nvidia.ptxas.path
    except Exception:  # Triton не нашёл ptxas (его нет в комплекте)
        return False
    return bool(path) and Path(path).is_file()


def enable(force: bool | None = None) -> bool:
    """Включить сборку ядер драйвером, если ptxas нет. Возвращает, включена ли она."""
    setting = os.environ.get("RAMVI_DRIVER_JIT")
    if force is None:
        force = None if setting is None else setting == "1"
    if force is False or (force is None and ptxas_available()):
        return False
    from triton.backends.nvidia import compiler

    cuda = driver_cuda_version()
    tag = f"driver-jit-cuda-{cuda}"

    def get_ptx_version_from_options(options, arch):  # type: ignore[no-untyped-def]
        return options.ptx_version or compiler.ptx_get_version(cuda)

    def make_cubin(self, src, metadata, opt, capability):  # type: ignore[no-untyped-def]
        return src.encode("utf-8") + b"\0"  # PTX вместо cubin: соберёт драйвер

    compiler.get_ptxas_version = lambda arch=80: tag  # входит в ключ кеша ядер Triton
    compiler.get_ptx_version_from_options = get_ptx_version_from_options
    compiler.CUDABackend.make_cubin = make_cubin
    return True
