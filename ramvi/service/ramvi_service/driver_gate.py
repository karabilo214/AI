"""8.7: перед запуском движка — есть ли драйвер NVIDIA и поддерживает ли он нужную CUDA.

CUDA Toolkit и ptxas в комплект не входят (их нельзя распространять), ядра собирает драйвер
(engine/ramvi_engine/kernels/driver_jit.py). Поэтому клиенту нужен только драйвер NVIDIA с
поддержкой CUDA не ниже REQUIRED_CUDA; если его нет или он старый — понятное сообщение,
что установить, вместо непонятного сбоя движка.
"""

import ctypes
import sys
from collections.abc import Callable

REQUIRED_CUDA = (13, 2)  # CUDA библиотек комплекта (torch cu132)
DRIVERS_URL = "https://www.nvidia.com/drivers"


def driver_cuda() -> tuple[int, int] | None:
    """Версия CUDA, которую поддерживает драйвер (cuDriverGetVersion); None — драйвера нет."""
    name = {"win32": "nvcuda.dll", "linux": "libcuda.so.1"}.get(sys.platform)
    if name is None:
        return None
    try:
        lib = ctypes.CDLL(
            name
        )  # на Windows cuDriverGetVersion — __stdcall, но без аргументов-стека x64 одинаково
    except OSError:
        return None
    value = ctypes.c_int()
    if lib.cuDriverGetVersion(ctypes.byref(value)) != 0 or value.value <= 0:
        return None
    return value.value // 1000, value.value % 1000 // 10


def check(read: Callable[[], tuple[int, int] | None] = driver_cuda) -> tuple[str, str] | None:
    """(ключ, подробность) — движок не запускать; None — драйвер подходит."""
    have = read()
    need = "{}.{}".format(*REQUIRED_CUDA)
    if have is None:
        return "engine.no_nvidia_driver", need
    if have < REQUIRED_CUDA:
        return "engine.driver_cuda_old", "{}.{}".format(*have)
    return None
