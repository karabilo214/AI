"""Совместимость старых импортов на время переноса RAMVI Engine."""

import sys
from pathlib import Path

# Только переходный пакет знает расположение старого рабочего дерева.
_engine_checkout = Path(__file__).resolve().parents[1] / "ramvi" / "engine"
if (_engine_checkout / "ramvi_engine" / "__init__.py").is_file():
    _location = str(_engine_checkout)
    if _location not in sys.path:
        sys.path.insert(0, _location)
