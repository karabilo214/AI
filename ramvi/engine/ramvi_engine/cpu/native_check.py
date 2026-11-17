"""3.18: есть ли ядро C++ (qmm) для счёта процессором — без загрузки DLL и Numba.

Те же условия, что проверяет `gcpu._dll` до загрузки библиотеки: не отключено через
`RAMVI_NO_CPP`, Windows/Linux, процессор с AVX-512 (X86_V4) и F16C, файл библиотеки на месте.
Нужно оценке памяти: с ядром C++ матрицы F16 считаются прямо из файла модели, без копии
в float32. Целостность библиотеки проверяет `_dll` при первом вызове; если она не пройдёт,
движок остановится с ошибкой, а не будет тихо держать неучтённую копию.
"""

import sys
from collections.abc import Mapping
from pathlib import Path

from ramvi_engine.config import env


def cpu_features() -> Mapping[str, bool]:
    try:
        import numpy as np

        return np._core._multiarray_umath.__cpu_features__  # type: ignore[attr-defined,no-any-return]
    except (ImportError, AttributeError):
        return {}


def native_cpu_available(
    features: Mapping[str, bool] | None = None, platform: str | None = None
) -> bool:
    platform = sys.platform if platform is None else platform
    if env("RAMVI_NO_CPP") == "1" or platform not in ("win32", "linux"):
        return False
    features = cpu_features() if features is None else features
    if not (features.get("X86_V4") and features.get("F16C")):
        return False
    filename = "qmm.dll" if platform == "win32" else "qmm.so"
    return (Path(__file__).resolve().parents[1] / "native" / filename).exists()
