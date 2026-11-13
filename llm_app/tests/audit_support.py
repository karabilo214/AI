"""Загрузка реальных методов без CUDA/Triton для проверок управления состоянием."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def definition(file, name, namespace):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / file), 'exec'), namespace)
    return namespace[name]
