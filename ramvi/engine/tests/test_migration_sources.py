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
    return mapped_weights_318(name, source)


# 3.18 ENG-MAPPED-01: слои вне видеокарты читаются из файла модели без собственных копий.
# Арифметика слоёв прежняя: меняется только источник байт подкачки (memmap + закреплённое
# кольцо загрузчика) и отсутствие неиспользуемой копии F16 в float32 при ядре C++.
MAPPED_318: dict[str, tuple[tuple[str, str], ...]] = {
    "engine": (
        (
            '                pend = []\n                for src, dst in job["pairs"]:\n                    for a in range(0, src.numel(), CH):\n                        if self._stop.is_set():\n                            raise RuntimeError("подкачка закрыта")\n                        with torch.cuda.stream(self.cs):\n                            dst[a:a + CH].copy_(src[a:a + CH], non_blocking=True)\n                            event = torch.cuda.Event(); event.record(self.cs)\n                        pend.append(event)\n',
            '                pend = []\n                for src, dst in job["pairs"]:\n                    staged = not src.is_pinned()\n                    for a in range(0, src.numel(), CH):\n                        if self._stop.is_set():\n                            raise RuntimeError("подкачка закрыта")\n                        part = src[a:a + CH]\n                        if staged:\n                            slot = self._stage_slot(part.numel())\n                            slot.copy_(part)  # из файла (memmap): чтение с диска, если страниц нет в RAM\n                            part = slot\n                        with torch.cuda.stream(self.cs):\n                            dst[a:a + CH].copy_(part, non_blocking=True)\n                            event = torch.cuda.Event(); event.record(self.cs)\n                        if staged:\n                            self._stage_events[self._stage_index] = event\n                        pend.append(event)\n',
        ),
        (
            "    def _event(self, event):",
            '    STAGE_SLOTS = 4\n    STAGE_BYTES = 10 * 2**20\n\n    def _stage_slot(self, n):\n        """3.18: закреплённый буфер для куска из обычной памяти/файла (кольцо из 4 по 10 МиБ).\n\n        Буфер берётся снова только после того, как его прежняя копия в видеокарту завершилась."""\n        if not hasattr(self, "_stage"):\n            self._stage = [torch.empty(self.STAGE_BYTES, dtype=torch.uint8).pin_memory()\n                           for _ in range(self.STAGE_SLOTS)]\n            self._stage_events = [None] * self.STAGE_SLOTS\n            self._stage_index = -1\n        self._stage_index = (self._stage_index + 1) % self.STAGE_SLOTS\n        previous = self._stage_events[self._stage_index]\n        if previous is not None:\n            self._event(previous)\n        return self._stage[self._stage_index][:n]\n\n    def _event(self, event):',
        ),
    ),
    "gengine": (
        (
            'GM = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}\n',
            'GM = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}\nMAPPED_KINDS = ("F16", "F32")\n\n\ndef host_weights(raw, arena):\n    """3.18: источник подкачки матрицы слоя вне видеокарты — (байты, раскладка для _unpack).\n\n    F16/F32 в видеокарте лежат как в файле, поэтому подкачиваются прямо из файла модели (memmap):\n    своей копии в RAM нет, прочитанное держит в памяти система, а не поместившееся дочитывает с диска.\n    Q4_K/Q6_K в видеокарте раскладываются иначе (repack) — их копия по-прежнему в закреплённой RAM."""\n    buf, N, K, t = raw\n    if t in MAPPED_KINDS and N > 0 and K > 0 and K % 256 == 0:\n        import warnings\n        with warnings.catch_warnings():  # memmap только для чтения: torch предупреждает о записи\n            warnings.simplefilter("ignore", UserWarning)\n            src = torch.from_numpy(np.asarray(buf).reshape(-1))\n        dtype = torch.float16 if t == "F16" else torch.float32\n        return src, [("t", None, t, None), ("N", None, N, None), ("K", None, K, None), ("W", 0, dtype, (N, K))]\n    b, lay = _pack(repack(buf, N, K, t))\n    return arena.put(b.cpu()), lay\n\n\n',
        ),
        (
            '            ws = {n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight")) for n in GM}\n            if self.place[i] == "gpu":\n                self.W.append(ws)\n            else:\n                pk = {}\n                for n, w in ws.items():\n                    b, lay = _pack(w)\n                    pk[n] = (arena.put(b.cpu()), lay)\n                if self.place[i] == "hyb":\n                    self.HW[i] = pk\n                else:\n                    self.W.append(pk)\n                del ws\n',
            '            if self.place[i] == "gpu":\n                self.W.append({n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight")) for n in GM})\n            else:\n                pk = {n: host_weights(g.raw(f"blk.{i}.{GM[n]}.weight"), arena) for n in GM}\n                if self.place[i] == "hyb":\n                    self.HW[i] = pk\n                else:\n                    self.W.append(pk)\n',
        ),
        (
            "    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True):\n",
            "    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True, cpu_frac=None):\n",
        ),
        (
            '        self.cpu_frac = float(env("RAMVI_CPU_FRAC", "0.58"))  # доля слоёв вне видеокарты на процессоре\n',
            '        if cpu_frac is None:  # 3.18: то же правило, что у плана (несжатые слои — процессору)\n            from ramvi_engine.planning.automatic import offload_cpu_fraction, unquantized_layers\n            cpu_frac = offload_cpu_fraction(unquantized_layers(g), env("RAMVI_CPU_FRAC"))\n        self.cpu_frac = float(cpu_frac)  # доля слоёв вне видеокарты на процессоре\n',
        ),
    ),
    "gcpu": (
        (
            "        else:  # F16 / F32 — в float32 (такие матрицы небольшие)\n",
            '        elif t == "F16" and K % 256 == 0 and _dll() is not None:\n            self.W = None  # 3.18: ядро C++ считает F16 прямо из файла — копия в float32 не нужна\n        else:  # F16 / F32 — в float32 (такие матрицы небольшие)\n',
        ),
    ),
}


def mapped_weights_318(name: str, source: str) -> str:
    for old, new in MAPPED_318.get(name, ()):
        assert source.count(old) == 1, (name, old[:60])
        source = source.replace(old, new)
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
