"""Загрузка реальных методов без CUDA/Triton для проверок управления состоянием."""
import ast
from pathlib import Path
from llm_app._compat import MODULES, source_path
ROOT = Path(__file__).resolve().parents[1]
def definition(file, name, namespace):
    path = source_path(file) if Path(file).stem in MODULES and Path(file).name == file else ROOT / file
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]
