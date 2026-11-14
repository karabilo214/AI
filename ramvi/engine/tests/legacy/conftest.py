"""Лёгкое окружение не собирает унаследованные вычислительные тесты."""

from importlib.util import find_spec

collect_ignore_glob = (
    ["test_*.py"]
    if any(find_spec(name) is None for name in ("numpy", "torch", "numba", "xgrammar"))
    else []
)
