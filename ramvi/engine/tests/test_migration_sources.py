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
    "__main__": "cli",
}


class WithoutImports(ast.NodeTransformer):
    """Исключить перемещённые импорты из сравнения вычислительных тел."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | None:
        if node.name == "_planning_config":
            return None  # separately exercised CLI configuration boundary
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_argument"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value
            in {
                "--plan-mode",
                "--reserve-gib",
                "--max-batch",
                "--draft-policy",
                "--prefill",
                "--no-paging",
                "--load-attempts",
            }
        ):
            return None
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_argument"
        ):  # текст справки не влияет на вычисления
            node.value.keywords = [k for k in node.value.keywords if k.arg != "help"]
        self.generic_visit(node)
        return node

    def visit_Try(self, node: ast.Try) -> ast.Try | None:
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
            and ast.unparse(node.body[0].value) == "_planning_config(a)"
        ):
            return None
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.If | None:
        if ast.unparse(node.test) == "a.cmd == 'plan' and config.planning is not None":
            return None
        self.generic_visit(node)
        return node

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
    if name in {"gengine", "moe", "plan"}:
        source = source.replace(
            'GGUF(find_ollama(name) if ":" in name and not name.endswith(".gguf") else name)',
            "GGUF(resolve_gguf_name(name, find_ollama))",
        )
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
        source = source.replace(
            'gguf = ":" in a.model or a.model.endswith(".gguf")', "gguf = is_gguf_model(a.model)"
        )
        source = source.replace(
            'ap.add_argument("--api-key", default=None,',
            'ap.add_argument("--api-key", default=os.environ.get("RAMVI_API_KEY"),',
        )
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
    if name == "__main__":
        source = source.replace(
            "  plan   — показать план памяти и выйти.",
            "  plan   — показать план памяти и выйти;\n"
            "  capabilities — JSON-карта возможностей без загрузки модели.",
        ).replace("Запустить serve, chat или plan", "Запустить serve, chat, plan или capabilities")
        # ENG-02: только новая ветка discovery; прежнюю загрузку/генерацию сверяем.
        source = source.replace(
            'choices=["serve", "chat", "plan"]',
            'choices=["serve", "chat", "plan", "capabilities"]',
        ).replace('ap.add_argument("--model", required=True)', 'ap.add_argument("--model")')
        source = source.replace(
            "    if a.maxt < 1",
            '    if a.cmd == "capabilities":\n'
            "        import json\n"
            "        from ramvi_engine.capabilities import capability_catalog\n"
            "        print(json.dumps(capability_catalog(), ensure_ascii=False, indent=2))\n"
            "        return\n"
            "    if not a.model:\n"
            '        ap.error("--model обязателен для serve, chat и plan")\n'
            "    if a.maxt < 1",
        )
    if name in {"engine", "gengine", "moe", "gcpu", "prefill", "measured_plan"}:
        source = (
            source.replace('os.environ.get("LLM_', 'env("RAMVI_')
            .replace("os.environ.get('LLM_", "env('RAMVI_")
            .replace("LLM_", "RAMVI_")
        )
    if name == "tune":
        source = (
            source.replace(
                "Результат сохраняется в ~/.cache/llm_app/tune.json",
                "Результат сохраняется в каталоге данных RAMVI/cache/tune.json",
            )
            .replace(
                'CACHE = os.path.join(os.path.expanduser("~"), ".cache", "llm_app", "tune.json")',
                'CACHE = str(cache_dir() / "tune.json")',
            )
            .replace(
                '        if os.path.exists(CACHE):\n            with open(CACHE, encoding="utf-8") as handle:',
                "        read_cache = tuning_cache_for_read(CACHE)\n"
                "        if os.path.exists(read_cache):\n"
                '            with open(read_cache, encoding="utf-8") as handle:',
            )
        )
    if name == "__main__":
        # Lifecycle заменяет только загрузку; маршруты/терминальный цикл прежние.
        start = source.index("    from .engine import Engine\n")
        end = source.index("    from .server import App, serve\n", start)
        source = (
            source[:start]
            + (
                "    eng = create_session(config)\n"
                "    eng.load()\n"
                "    tk = eng.tokenizer\n"
                "    draft = eng.draft_handle if a.draft else None\n"
            )
            + source[end:]
        )
        start = source.index('    if a.cmd == "serve":\n')
        end = source.index("\n\nif __name__", start)
        body = source[start:end]
        source = (
            source[:start]
            + "    try:\n"
            + "".join("    " + line + "\n" for line in body.splitlines())
            + "    finally:\n        eng.close()\n"
            + source[end:]
        )
        start = source.index("    eng.load()\n")
        end = source.index("    try:\n", start)
        prefix = source[start:end]
        source = (
            source[:start]
            + "    try:\n"
            + "".join("    " + line + "\n" for line in prefix.splitlines())
            + source[end + len("    try:\n") :]
        )
    if name == "plan":
        # ENG-ResourceEstimate: отчёт добавлен после выбора размещения.
        # Исходные решения планировщика и measured_schedule сверяются далее.
        source = source.replace(
            '    log(f"план: видеокарта свободно',
            "    resources = estimate_hf(ck, place, maxt, draft=MemoryUsage(vram=int(draft_bytes)),\n"
            "                            reserve=MemoryUsage(vram=int(vram_reserve)))\n"
            '    plan["resources"] = resources.to_dict()\n'
            '    plan["resource_fit"] = resources.assess(vram_available=int(free), ram_available=int(avail))\n'
            '    if not plan["resource_fit"]["estimated_fit"]:\n'
            '        log("  внимание: подробная оценка ресурсов превышает доступный бюджет; нужен меньший план/контекст")\n'
            '    log(f"план: видеокарта свободно',
        ).replace(
            '    log(f"GGUF {arch}:',
            "    resources = estimate_gguf(g, place, maxt, reserve=MemoryUsage(vram=int(.9 * GiB)))\n"
            '    plan["resources"] = resources.to_dict()\n'
            '    plan["resource_fit"] = resources.assess(vram_available=int(free), ram_available=int(plan["ram_avail"]))\n'
            '    log(f"GGUF {arch}:',
        )
    if name in {"engine", "gengine", "moe"}:
        # 2.6 changes only configured chunk limits/reserve/manual OOM policy.
        if name == "engine":
            source = source.replace(
                "def __init__(self, name, maxt=2048, cpu=True, resident=None, log=print):",
                "def __init__(self, name, maxt=2048, cpu=True, resident=None, log=print, vram_reserve=None):",
            )
            source = source.replace(
                "make_plan(ck, maxt, cpu=cpu, force_resident=resident, log=log)",
                "make_plan(ck, maxt, cpu=cpu, force_resident=resident, log=log, vram_reserve=0.8 * GiB if vram_reserve is None else vram_reserve)",
            )
            source = source.replace(
                '        gate_enabled = env("RAMVI_DRAFT_GATE", "0") == "1"',
                '        gate_enabled = env("RAMVI_DRAFT_GATE", "0") == "1"\n        max_batch = batch_limit(self)',
            )
            source = source.replace(
                "kmax = min(k or 15, 15,", "kmax = min(k or 15, max_batch - 1, 15,"
            )
        start = source.index("    def _prefill(self, tokens, start):")
        end = source.find("\n    def ", start + 5)
        if end < 0:
            end = len(source)
        part = source[start:end]
        if name == "moe":
            part = part.replace(
                "        n = len(tokens)",
                "        batch = batch_limit(self)\n        n = len(tokens)",
            )
            part = (
                part.replace("range(0, n, 16)", "range(0, n, batch)")
                .replace("tokens[c:c + 16]", "tokens[c:c + batch]")
                .replace(
                    "min(16, n - (n - 1) // 16 * 16)", "min(batch, n - (n - 1) // batch * batch)"
                )
            )
        else:
            part = part.replace(
                "blocked = enabled(len(tokens))",
                "blocked = prefill_enabled(self, len(tokens), enabled)\n        batch = batch_limit(self)",
            )
            part = (
                part.replace("range(0, n, 16)", "range(0, n, batch)")
                .replace("min(16, n - c)", "min(batch, n - c)")
                .replace("X[c:c + 16]", "X[c:c + batch]")
            )
        source = source[:start] + part + source[end:]
        if name == "gengine":
            source = source.replace(
                "                original = original or exc\n                self.ctx = []",
                "                original = original or exc\n                self.ctx = []\n                if not getattr(self, 'allow_oom_demote', True):\n                    raise",
            )
    if name == "constrain":
        source = source.replace(
            'with tempfile.TemporaryDirectory(prefix="llm-grammar-") as folder:',
            'with tempfile.TemporaryDirectory(prefix="ramvi-grammar-", dir=ensure_dir(work_dir())) as folder:',
        )
    if name == "__main__":
        # 3.3: только сторож родителя перед загрузкой; загрузка/serve прежние.
        assert source.count("    torch.set_grad_enabled(False)") == 1
        source = source.replace(
            "    torch.set_grad_enabled(False)",
            '    if a.cmd == "serve":\n'
            "        from ramvi_engine.processes import ParentWatch\n"
            "        ParentWatch.from_env()\n"
            "    torch.set_grad_enabled(False)",
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
