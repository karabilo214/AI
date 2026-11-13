"""Переходный импорт; реализация находится в RAMVI Engine."""

from llm_app._compat import alias_module

alias_module(__name__, "ramvi_engine.core.gengine")
