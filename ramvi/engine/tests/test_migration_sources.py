"""Повторяемая структурная сверка переноса с последним исходным коммитом."""

import ast
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "ramvi/engine/ramvi_engine"
BASELINE = "7e3330232b4b83e13f1456cb05c579fef52ef9a9"
# Карта зафиксирована для экзамена переноса; не строится из проверяемого пакета.
MODULES: dict[str, str] = {
    "head": "generation.head",
    "engine": "core.engine",
    "gengine": "core.gengine",
    "moe": "core.moe",
    "reference": "core.reference",
    "tkernels": "kernels.tkernels",
    "wcomp": "kernels.wcomp",
    "gkernels": "kernels.gkernels",
    "vocab8": "kernels.vocab8",
    "q4": "kernels.q4",
    "prefill": "kernels.prefill",
    "prefill_kernels": "kernels.prefill_kernels",
    "cpu_layer": "cpu.cpu_layer",
    "gcpu": "cpu.gcpu",
    "gguf": "formats.gguf",
    "model": "formats.model",
    "tokenizer_check": "formats.tokenizer_check",
    "plan": "planning.plan",
    "tune": "planning.tune",
    "measured_plan": "planning.measured_plan",
    "constrain": "generation.constrain",
    "grammar_worker": "generation.grammar_worker",
    "server": "server.app",
    "request_validation": "server.validation",
    "__main__": "cli",
}


class WithoutImports(ast.NodeTransformer):
    """Исключить перемещённые импорты из сравнения вычислительных тел."""

    def visit_Import(self, node: ast.Import) -> None:
        """Удалить импорт из сравниваемого дерева."""
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Удалить импорт из сравниваемого дерева."""
        return None


def source_before_move(relative_path: str) -> bytes:
    """Получить прежние байты из git без checkout и запуска старого кода."""
    return subprocess.run(
        ["git", "show", f"{BASELINE}:llm_app/{relative_path}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        timeout=15,
        check=True,
    ).stdout


def allowed_layout_changes(name: str, source: str) -> str:
    """Применить только согласованные различия размещения, hash и CLI."""
    if name == "gcpu":
        source = (
            source.replace(
                'path = os.path.join(os.path.dirname(__file__), "cpp", filename)',
                'path = str(Path(__file__).resolve().parents[1] / "native" / filename)',
            )
            .replace(
                'Path(__file__).with_name("cpp") / "qmm.cpp"',
                'Path(__file__).resolve().parents[1] / "native" / "qmm.cpp"',
            )
            .replace("cpp/build.sh", "native/build.sh")
        )
    if name == "server":
        source = source.replace(
            '        for file in sorted(Path(__file__).parent.glob("*.py")):\n'
            "            digest.update(file.name.encode()); digest.update(file.read_bytes())",
            "        source_root = Path(__file__).resolve().parents[1]\n"
            '        for file in sorted(source_root.rglob("*.py")):\n'
            "            digest.update(file.relative_to(source_root).as_posix().encode())\n"
            "            digest.update(file.read_bytes())",
        )
    if name == "__main__":
        source = (
            source.replace("python -m llm_app", "python -m ramvi_engine")
            .replace(
                'argparse.ArgumentParser(prog="llm_app")',
                'argparse.ArgumentParser(prog="ramvi-engine")',
            )
            .replace(
                "def main():\n",
                'def main() -> None:\n    """Запустить serve, chat или plan для RAMVI Engine."""\n',
                1,
            )
        )
    if name == "__main__":
        source = source.replace(
            '    ap = argparse.ArgumentParser(prog="ramvi-engine")',
            "    from ramvi_engine.console import configure_output\n"
            "    configure_output()\n"
            '    ap = argparse.ArgumentParser(prog="ramvi-engine")',
            1,
        )
    return source


def computational_tree(source: str) -> str:
    """Нормализовать AST без исполнения проверяемого модуля."""
    return ast.dump(WithoutImports().visit(ast.parse(source)), include_attributes=False)


@pytest.mark.parametrize("name", MODULES)
def test_migrated_computation_matches_baseline(name: str) -> None:
    """Каждый перенесённый модуль сохраняет прежние вычислительные тела."""
    old = source_before_move(f"{name}.py").decode("utf-8")
    new = (PACKAGE_ROOT / (MODULES[name].replace(".", "/") + ".py")).read_text(encoding="utf-8")
    assert computational_tree(allowed_layout_changes(name, old)) == computational_tree(new)


@pytest.mark.parametrize(
    "name", ["wcomp", "gkernels", "tkernels", "q4", "vocab8", "prefill_kernels"]
)
def test_independent_kernels_keep_exact_bytes(name: str) -> None:
    """Перенос не меняет исходные байты ядер и ключи автонастройки."""
    assert (
        source_before_move(f"{name}.py") == (PACKAGE_ROOT / "kernels" / f"{name}.py").read_bytes()
    )


@pytest.mark.parametrize("name", ["qmm.cpp", "build.sh", "write_manifest.py"])
def test_native_sources_keep_exact_bytes(name: str) -> None:
    """C++ и правила сборки совпадают с исходной версией."""
    assert source_before_move(f"cpp/{name}") == (PACKAGE_ROOT / "native" / name).read_bytes()


def test_comparison_detects_arithmetic_change() -> None:
    """Контроль: переезд импорта допустим, изменение вычисления обнаруживается."""
    old = "import old\ndef forward(x):\n    return x * 2\n"
    moved = "import new\ndef forward(x):\n    return x * 2\n"
    broken = moved.replace("x * 2", "x * 3")
    assert computational_tree(old) == computational_tree(moved)
    assert computational_tree(old) != computational_tree(broken)
