"""Загрузка реальных методов без CUDA/Triton для проверок управления состоянием."""
import ast
from pathlib import Path
from importlib.util import find_spec
from ramvi_engine.config import env, cache_dir, data_dir, work_dir, results_dir, ensure_dir, tuning_cache_for_read

MODULES = {'head': 'ramvi_engine.generation.head', 'engine': 'ramvi_engine.core.engine', 'gengine': 'ramvi_engine.core.gengine', 'moe': 'ramvi_engine.core.moe', 'reference': 'ramvi_engine.core.reference', 'tkernels': 'ramvi_engine.kernels.tkernels', 'wcomp': 'ramvi_engine.kernels.wcomp', 'gkernels': 'ramvi_engine.kernels.gkernels', 'vocab8': 'ramvi_engine.kernels.vocab8', 'q4': 'ramvi_engine.kernels.q4', 'prefill': 'ramvi_engine.kernels.prefill', 'prefill_kernels': 'ramvi_engine.kernels.prefill_kernels', 'cpu_layer': 'ramvi_engine.cpu.cpu_layer', 'gcpu': 'ramvi_engine.cpu.gcpu', 'gguf': 'ramvi_engine.formats.gguf', 'model': 'ramvi_engine.formats.model', 'tokenizer_check': 'ramvi_engine.formats.tokenizer_check', 'plan': 'ramvi_engine.planning.plan', 'tune': 'ramvi_engine.planning.tune', 'measured_plan': 'ramvi_engine.planning.measured_plan', 'constrain': 'ramvi_engine.generation.constrain', 'grammar_worker': 'ramvi_engine.generation.grammar_worker', 'server': 'ramvi_engine.server.app', 'request_validation': 'ramvi_engine.server.validation', '__main__': 'ramvi_engine.cli'}
ROOT = Path(__file__).resolve().parents[2] / "ramvi_engine"

def source_path(filename):
    spec = find_spec(MODULES[Path(filename).stem])
    if spec is None or spec.origin is None:
        raise ImportError(filename)
    return Path(spec.origin)

def definition(file, name, namespace):
    for key, value in dict(env=env, cache_dir=cache_dir, data_dir=data_dir, work_dir=work_dir, results_dir=results_dir, ensure_dir=ensure_dir, tuning_cache_for_read=tuning_cache_for_read).items():
        namespace.setdefault(key, value)
    path = source_path(file) if Path(file).stem in MODULES else ROOT / file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]
