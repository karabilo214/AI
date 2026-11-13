"""Единый переходный маршрут старых имён к перенесённым модулям."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
import sys

MODULES: dict[str, str] = {'head': 'ramvi_engine.generation.head', 'engine': 'ramvi_engine.core.engine', 'gengine': 'ramvi_engine.core.gengine', 'moe': 'ramvi_engine.core.moe', 'reference': 'ramvi_engine.core.reference', 'tkernels': 'ramvi_engine.kernels.tkernels', 'wcomp': 'ramvi_engine.kernels.wcomp', 'gkernels': 'ramvi_engine.kernels.gkernels', 'vocab8': 'ramvi_engine.kernels.vocab8', 'q4': 'ramvi_engine.kernels.q4', 'prefill': 'ramvi_engine.kernels.prefill', 'prefill_kernels': 'ramvi_engine.kernels.prefill_kernels', 'cpu_layer': 'ramvi_engine.cpu.cpu_layer', 'gcpu': 'ramvi_engine.cpu.gcpu', 'gguf': 'ramvi_engine.formats.gguf', 'model': 'ramvi_engine.formats.model', 'tokenizer_check': 'ramvi_engine.formats.tokenizer_check', 'plan': 'ramvi_engine.planning.plan', 'tune': 'ramvi_engine.planning.tune', 'measured_plan': 'ramvi_engine.planning.measured_plan', 'constrain': 'ramvi_engine.generation.constrain', 'grammar_worker': 'ramvi_engine.generation.grammar_worker', 'server': 'ramvi_engine.server.app', 'request_validation': 'ramvi_engine.server.validation'}


def alias_module(old_name: str, new_name: str) -> ModuleType:
    """Сохранить идентичность модуля для импортов и monkeypatch старого API."""
    module = import_module(new_name)
    sys.modules[old_name] = module
    return module


def source_path(filename: str) -> Path:
    """Найти исходник для старых AST-проверок, не импортируя GPU-модуль."""
    target = MODULES[Path(filename).stem]
    spec = find_spec(target)
    if spec is None or spec.origin is None:
        raise ImportError(f"Не найден исходник {target}")
    return Path(spec.origin)
