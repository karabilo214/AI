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
        if node.name in ("_planning_config", "translator", "with_defaults"):
            return None  # CLI configuration boundary / B.1 перевод справки — не вычисления
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
        if ast.unparse(node.value) == "with_defaults(ap, t)":
            return None  # B.1: значения по умолчанию в справке
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

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if ast.unparse(node.targets[0]) == "I18N_DIR":
            return None  # B.1: где лежат каталоги переводов справки
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
    if name in {"engine", "gengine"}:
        # B.15: текст ошибки переполнения контекста — машинная метка с размером запроса
        # (сервис показывает «нужно N, есть M» из каталога); вычисления не меняются.
        source = source.replace(
            'raise ValueError(f"текст длиннее {self.maxt} слов")',
            'raise ValueError(f"context overflow: needed={pos0 + len(tokens)} available={self.maxt}")',
        ).replace(
            'raise ValueError(f"запрос длиннее контекста {self.maxt}")',
            'raise ValueError(f"context overflow: needed={P} available={self.maxt}")',
        )
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
    return cli_help_b1(
        name,
        driver_jit_87(
            name,
            release_pages_86(
                name, tc_cfg_b6(name, multigpu_410(name, mapped_weights_318(name, source)))
            ),
        ),
    )


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


# 4.10 ENG-MULTIGPU: слои GGUF-модели, которой мало одной карты, раскладываются блоками по
# нескольким картам; на одной карте путь прежний (те же буферы и ядра, CUDA-граф). Арифметика
# слоя не меняется: добавлены выбор карты, буферы каждой карты и передача состояния между ними.
MULTIGPU_410: tuple[tuple[str, str], ...] = (
    (
        "    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True, cpu_frac=None):\n",
        "    multi = False  # 4.10: слои на нескольких картах (по умолчанию — одна)\n    def __init__(self, name, maxt=2048, resident=None, log=print, vram_reserve=None, extra_vram=0, cpu=True, cpu_frac=None):\n",
    ),
    (
        '        self.maxt, self.log, dev = maxt, log, "cuda"\n',
        '        # 4.10: какие карты использовать (RAMVI_DEVICES); первая — основная: на ней словарь,\n        # подкачка по шине и слои вне видеопамяти. set_device — «cuda» дальше значит основную карту.\n        from ramvi_engine.planning.multigpu import SECONDARY_OVERHEAD, split_layers, visible_devices\n        self.devices = [torch.device("cuda", i) for i in visible_devices(torch.cuda.device_count())]\n        torch.cuda.set_device(self.devices[0])\n        self.maxt, self.log, dev = maxt, log, "cuda"\n',
    ),
    (
        '        self.small = [{k: torch.from_numpy(g.full(f"blk.{i}.{n}.weight")).to(dev)\n                       for k, n in (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))}\n                      for i in range(self.L)]\n        # план памяти: байты слоёв\n        lb = [sum(g.raw(f"blk.{i}.{GM[n]}.weight")[0].size for n in GM) for i in range(self.L)]\n        kvb = 2 * self.KV * maxt * self.HD * 2\n        free, _ = torch.cuda.mem_get_info()\n        budget = free - vram_reserve - extra_vram - 16 * self.V * 4  # логиты\n        if resident is None:\n            R = 0\n            while R < self.L and sum(lb[:R + 1]) + self.L * kvb + (2 * max(lb) if R + 1 < self.L else 0) <= budget:\n                R += 1\n        else:\n            R = resident\n',
        '        # план памяти: байты слоёв\n        lb = [sum(g.raw(f"blk.{i}.{GM[n]}.weight")[0].size for n in GM) for i in range(self.L)]\n        kvb = 2 * self.KV * maxt * self.HD * 2\n        free, _ = torch.cuda.mem_get_info()\n        budget = free - vram_reserve - extra_vram - 16 * self.V * 4  # логиты\n        fits = 0  # сколько слоёв помещается в основную карту\n        while fits < self.L and sum(lb[:fits + 1]) + self.L * kvb + (2 * max(lb) if fits + 1 < self.L else 0) <= budget:\n            fits += 1\n        R = fits if resident is None else resident\n        # 4.10: модель, которой мало основной карты, раскладывается подряд блоками по всем картам.\n        # Помещается в одну — остаётся на одной (меньше передач между картами), путь прежний.\n        self.layer_dev = [0] * self.L\n        wanted = self.L if resident is None else resident\n        if len(self.devices) > 1 and fits < wanted:\n            budgets = [budget - 2 * max(lb) - self.L * kvb] + [\n                torch.cuda.mem_get_info(d)[0] - vram_reserve - SECONDARY_OVERHEAD for d in self.devices[1:]]\n            split = split_layers([b + kvb for b in lb], budgets, resident)\n            if len(split) > fits:\n                R = len(split)\n                self.layer_dev = split + [0] * (self.L - R)\n        self.multi = len(set(self.layer_dev)) > 1\n        self.small = [{k: torch.from_numpy(g.full(f"blk.{i}.{n}.weight")).to(self._dev(i))\n                       for k, n in (("ln1", "attn_norm"), ("ln2", "ffn_norm"), ("qn", "attn_q_norm"), ("kn", "attn_k_norm"))}\n                      for i in range(self.L)]\n',
    ),
    (
        '                self.W.append({n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight")) for n in GM})\n            else:\n                pk = {n: host_weights(g.raw(f"blk.{i}.{GM[n]}.weight"), arena) for n in GM}\n                if self.place[i] == "hyb":\n                    self.HW[i] = pk\n                else:\n                    self.W.append(pk)\n            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=dev, dtype=torch.float16)\n',
        '                self.W.append({n: repack(*g.raw(f"blk.{i}.{GM[n]}.weight"), dev=self._dev(i)) for n in GM})\n            else:\n                pk = {n: host_weights(g.raw(f"blk.{i}.{GM[n]}.weight"), arena) for n in GM}\n                if self.place[i] == "hyb":\n                    self.HW[i] = pk\n                else:\n                    self.W.append(pk)\n            self.Kc[i] = torch.zeros(self.KV, maxt, self.HD, device=self._dev(i), dtype=torch.float16)\n',
    ),
    (
        "        self.graphs, self.ctx = {}, []\n",
        "        self._buffers()\n        self.graphs, self.ctx = {}, []\n",
    ),
    (
        '            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")\n',
        '            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")\n        if self.multi:\n            per = ", ".join(f"{d}: {self.layer_dev[:R].count(i)} слоёв, {torch.cuda.memory_allocated(d) / GiB:.2f} ГиБ"\n                            for i, d in enumerate(self.devices))\n            log(f"несколько видеокарт (4.10): {per}")\n\n    # ---- 4.10: несколько видеокарт ----\n    def _dev(self, l):\n        """Устройство слоя: слой в видеопамяти — на своей карте, остальные — через основную."""\n        return self.devices[self.layer_dev[l]] if self.layer_dev[l] else self.devices[0]\n\n    def _buffers(self):\n        """Рабочие буферы каждой карты: на основной — прежние, на остальных — свои такие же."""\n        main = types.SimpleNamespace(device=self.devices[0], xb=self.xb, hb=self.hb, qkvb=self.qkvb, qb=self.qb,\n                                     attb=self.attb, gub=self.gub, mb=self.mb, pos=self.pos, inv=self.inv)\n        self.bufs = {0: main}\n        for d in sorted(set(self.layer_dev) - {0}):\n            device = self.devices[d]\n            f = lambda n: torch.zeros(16, n, device=device)\n            self.bufs[d] = types.SimpleNamespace(\n                device=device, xb=f(self.D), hb=f(self.D), qkvb=f(self.QKV), qb=f(self.H * self.HD),\n                attb=f(self.H * self.HD), gub=f(2 * self.FF), mb=f(self.FF),\n                pos=torch.zeros(1, dtype=torch.long, device=device), inv=self.inv.to(device))\n\n    def _fill_pos(self, value):\n        for b in getattr(self, "bufs", {0: self}).values():\n            b.pos.fill_(value)\n\n    def _to(self, x, d, T):\n        """Состояние слоя (T×D) — в буфер карты d; это единственная передача между картами."""\n        dst = self.bufs[d].xb[:T]\n        dst.copy_(x)\n        return dst\n',
    ),
    (
        '        for l in range(self.L):\n            pl = self.place[l]\n            routes = getattr(self, "_active_routes", None)\n            mode = routes[l] if routes is not None else ("stream" if pl == "hyb" and T > 1 else "cpu" if pl == "hyb" else pl)\n',
        '        cur, xd = 0, x  # 4.10: на какой карте сейчас состояние слоя\n        for l in range(self.L):\n            pl = self.place[l]\n            routes = getattr(self, "_active_routes", None)\n            mode = routes[l] if routes is not None else ("stream" if pl == "hyb" and T > 1 else "cpu" if pl == "hyb" else pl)\n            if self.multi:\n                d = self.layer_dev[l] if mode == "gpu" else 0\n                if d != cur:\n                    xd = x if d == 0 else self._to(xd, d, T)\n                    if d == 0:\n                        x.copy_(self.bufs[cur].xb[:T])\n                    cur = d\n                if d:\n                    self._gpu_layer(l, self.W[l], xd, T)\n                    continue\n',
    ),
    (
        "        rmsg_k[(T,)](x, self._norm, self.hb[:T], self.eps, self.D, DP=triton.next_power_of_2(self.D))\n",
        "        if self.multi and cur:  # последний блок был на другой карте — состояние на основную\n            x.copy_(self.bufs[cur].xb[:T])\n        rmsg_k[(T,)](x, self._norm, self.hb[:T], self.eps, self.D, DP=triton.next_power_of_2(self.D))\n",
    ),
    (
        "        H, KV, HD, D, s = self.H, self.KV, self.HD, self.D, self.small[l]\n        h, qkv, q, att, gu, mb = self.hb[:T], self.qkvb[:T], self.qb[:T], self.attb[:T], self.gub[:T], self.mb[:T]\n",
        '        if self.multi and self.place[l] == "gpu" and self.layer_dev[l]:\n            b = self.bufs[self.layer_dev[l]]\n            with torch.cuda.device(b.device):  # ядра Triton запускаются на текущей карте\n                return self._batch_layer_on(b, l, x, T, W, K, V)\n        return self._batch_layer_on(self._main_buffers(), l, x, T, W, K, V)\n\n    def _main_buffers(self):\n        bufs = getattr(self, "bufs", None)\n        if bufs is not None:\n            return bufs[0]\n        return types.SimpleNamespace(device=None, xb=self.xb, hb=self.hb, qkvb=self.qkvb, qb=self.qb, attb=self.attb,\n                                     gub=self.gub, mb=self.mb, pos=self.pos, inv=self.inv)\n\n    def _batch_layer_on(self, buf, l, x, T, W, K, V):\n        H, KV, HD, D, s = self.H, self.KV, self.HD, self.D, self.small[l]\n        h, qkv, q, att, gu, mb = buf.hb[:T], buf.qkvb[:T], buf.qb[:T], buf.attb[:T], buf.gub[:T], buf.mb[:T]\n',
    ),
    (
        '            rope_attn_k[(H,)](qkv, s["qn"], s["kn"], self.inv, self.pos, K, V, att, self.maxt, self.eps, HD ** -0.5,\n                              NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)\n        else:\n            qk_rope_cache_k[(H + 2 * KV, T)](qkv, s["qn"], s["kn"], self.inv, self.pos, q, K, V, self.maxt, self.eps,\n                                              NH=H, NKV=KV, HDIM=HD, half=HD // 2)\n            attn_k[(H, T)](q, K, V, self.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)\n',
        '            rope_attn_k[(H,)](qkv, s["qn"], s["kn"], buf.inv, buf.pos, K, V, att, self.maxt, self.eps, HD ** -0.5,\n                              NH=H, NKV=KV, HDIM=HD, half=HD // 2, BJ=64)\n        else:\n            qk_rope_cache_k[(H + 2 * KV, T)](qkv, s["qn"], s["kn"], buf.inv, buf.pos, q, K, V, self.maxt, self.eps,\n                                              NH=H, NKV=KV, HDIM=HD, half=HD // 2)\n            attn_k[(H, T)](q, K, V, buf.pos, att, self.maxt, HD ** -0.5, G=H // KV, HDIM=HD, BJ=64, NHQ=H)\n',
    ),
    (
        "        if self.all_gpu:\n            B = T  # граф обрабатывает только реальные строки, без записи KV за конец контекста\n",
        "        if self.all_gpu and not self.multi:  # 4.10: на нескольких картах — без CUDA-графа\n            B = T  # граф обрабатывает только реальные строки, без записи KV за конец контекста\n",
    ),
    (
        "        self.pos.fill_(pos0)\n        self._layers(self.xb[:T], T, pos0)\n",
        "        self._fill_pos(pos0)\n        self._layers(self.xb[:T], T, pos0)\n",
    ),
    (
        '        blocked = prefill_enabled(self, len(tokens), enabled)\n        batch = batch_limit(self)\n        self._active_routes = None\n        n = len(tokens)\n        self._set_phase("dN")\n        X = self._emb(tokens).cuda()\n        for l in range(self.L):\n',
        '        blocked = prefill_enabled(self, len(tokens), enabled) and not self.multi\n        batch = batch_limit(self)\n        self._active_routes = None\n        n = len(tokens)\n        self._set_phase("dN")\n        X = self._emb(tokens).cuda()\n        cur, Xd = 0, X  # 4.10: запрос переезжает на карту блока слоёв один раз\n        for l in range(self.L):\n            d = self.layer_dev[l] if self.multi and self.place[l] == "gpu" else 0\n            if d != cur:\n                if cur:\n                    X.copy_(Xd)\n                Xd = X.to(self.devices[d]) if d else X\n                cur = d\n',
    ),
    (
        '                    self.pos.fill_(start + c)\n                    self._batch_layer(l, X[c:c + T], T, W, self.Kc[l], self.Vc[l], None)\n            if self.place[l] != "gpu":\n                self._release()\n',
        '                    self._fill_pos(start + c)\n                    self._batch_layer(l, Xd[c:c + T], T, W, self.Kc[l], self.Vc[l], None)\n            if self.place[l] != "gpu":\n                self._release()\n        if cur:\n            X.copy_(Xd)\n',
    ),
    (
        '        self.place[l] = "stream"\n',
        '        if self.multi and self.layer_dev[l]:  # 4.10: слой по шине — на основной карте, и его кэш там же\n            self.Kc[l], self.Vc[l] = self.Kc[l].to(self.devices[0]), self.Vc[l].to(self.devices[0])\n            self.layer_dev[l] = 0\n            self.multi = len(set(self.layer_dev)) > 1\n        self.place[l] = "stream"\n',
    ),
)


def multigpu_410(name: str, source: str) -> str:
    if name != "gengine":
        return source
    for old, new in MULTIGPU_410:
        assert source.count(old) == 1, (name, old[:60])
        source = source.replace(old, new)
    return source


# B.6: подобранный блок (BN, warps) ядер тензорных блоков вместо постоянного (32, 4). BN делит
# только строки матрицы: каждое число считается теми же операциями (побайтно тот же результат).
TC_CFG_B6: dict[str, tuple[tuple[str, str], ...]] = {
    "gkernels": (
        (
            "import triton.language as tl\n",
            "from ramvi_engine.config import env\nimport triton.language as tl\n",
        ),
        (
            'def ggemv_tc(x, w, y, mode, T, cfg=None, ys=None):\n    t, N, K = w["t"], w["N"], w["K"]\n    ys = ys or N\n    if t == "Q4_K":\n        X3, SX = xprep(x, T, K)\n        BN, nw = cfg or CFGTC.get((t, N, K), (32, 4))\n        if N % BN:\n            raise ValueError("число строк не кратно блоку GPU-ядра")\n        q4k_tc_k[(N // BN,)](X3, SX, w["Q"], w["SC"], w["MN"], w["D"], y, T, N, K, ys, mode, BN=BN, num_warps=nw)\n    elif t == "Q6_K":\n        X3, SX = xprep(x, T, K)\n        BN, nw = cfg or CFGTC.get((t, N, K), (32, 4))\n',
            'def _cfg_tc(t: str, N: int, K: int) -> tuple[int, int]:\n    """B.6: блок (BN строк, warps) для пачки 2..16 слов. BN делит только строки матрицы — каждое\n    число считается теми же операциями, результат побайтно тот же. Замер RTX 5060 Laptop\n    (qwen3 8B/14B, 16 слов): Q4_K от 5120x5120 — (128, 8), остальные — (64, 4); быстрее\n    прежних (32, 4) в 1,25–2,3 раза. RAMVI_TC_CFG=fixed — прежние (32, 4) для сравнения."""\n    if env("RAMVI_TC_CFG", "auto") == "fixed":\n        return 32, 4\n    if t == "Q4_K" and N * K >= 5120 * 5120 and N % 128 == 0:\n        return 128, 8\n    return (64, 4) if N % 64 == 0 else (32, 4)\n\n\ndef ggemv_tc(x, w, y, mode, T, cfg=None, ys=None):\n    t, N, K = w["t"], w["N"], w["K"]\n    ys = ys or N\n    if t == "Q4_K":\n        X3, SX = xprep(x, T, K)\n        BN, nw = cfg or CFGTC.get((t, N, K)) or _cfg_tc(t, N, K)\n        if N % BN:\n            raise ValueError("число строк не кратно блоку GPU-ядра")\n        q4k_tc_k[(N // BN,)](X3, SX, w["Q"], w["SC"], w["MN"], w["D"], y, T, N, K, ys, mode, BN=BN, num_warps=nw)\n    elif t == "Q6_K":\n        X3, SX = xprep(x, T, K)\n        BN, nw = cfg or CFGTC.get((t, N, K)) or _cfg_tc(t, N, K)\n',
        ),
    ),
}


def tc_cfg_b6(name: str, source: str) -> str:
    for old, new in TC_CFG_B6.get(name, ()):
        assert source.count(old) == 1, (name, old[:60])
        source = source.replace(old, new)
    return source


# 8.6: страницы файла модели, прочитанные при загрузке весов, выводятся из рабочего набора
# процесса (VirtualUnlock / madvise). Вычисления не меняются: только вызов после загрузки,
# дополненная строка журнала и сама функция освобождения в gguf.
RELEASE_86: dict[str, tuple[tuple[str, str], ...]] = {
    "gengine": (
        (
            '        self.load_s = time.time() - t0\n        log(f"загружено за {self.load_s:.0f} с: видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ, "\n            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f})")\n        if self.multi:\n',
            '        self.load_s = time.time() - t0\n        # 8.6: прочитанные при загрузке страницы файла не держать в рабочем наборе процесса.\n        released = g.release_pages()\n        log(f"загружено за {self.load_s:.0f} с: видеокарта {torch.cuda.memory_allocated() / GiB:.2f} ГиБ, "\n            f"закреплено RAM {arena.pinned / GiB:.2f} ГиБ (обычной {arena.plain / GiB:.2f}); "\n            f"страницы файла из рабочего набора убраны: {\'да\' if released else \'нет\'}")\n        if self.multi:\n',
        ),
    ),
    "gguf": (
        (
            '    def close(self):\n        """После close все ранее выданные представления memmap использовать нельзя."""\n',
            '    def release_pages(self):\n        """Убрать прочитанные страницы файла из рабочего набора процесса (см. release_file_pages)."""\n        return self.mm is not None and release_file_pages(self.mm)\n\n    def close(self):\n        """После close все ранее выданные представления memmap использовать нельзя."""\n',
        ),
    ),
}


def release_pages_86(name: str, source: str) -> str:
    for old, new in RELEASE_86.get(name, ()):
        assert source.count(old) == 1, (name, old[:60])
        source = source.replace(old, new)
    if name == "gguf":
        source = (
            source.rstrip("\n")
            + '\n\ndef release_file_pages(mm):\n    """8.6: страницы файла модели, прочитанные при загрузке весов в видеокарту, остаются в рабочем\n    наборе процесса. Это не копия весов (страницы общие, файловые, их можно вернуть), но Windows\n    не считает их «доступной» памятью: с qwen3:8b целиком в видеокарте движок показывал 5,7 ГБ,\n    из них 4,8 ГБ — страницы файла, и голосу «не хватало» RAM. Здесь страницы выводятся из\n    рабочего набора; отображение остаётся, нужная позже страница подчитается из кэша системы.\n    Windows — VirtualUnlock на незакреплённой области (документированный способ убрать страницы\n    из рабочего набора, ответ ERROR_NOT_LOCKED ожидаем), POSIX — madvise(MADV_DONTNEED)."""\n    size = int(mm.size) * mm.itemsize\n    if size == 0:\n        return False\n    if sys.platform == "win32":\n        import ctypes\n        kernel = ctypes.WinDLL("kernel32", use_last_error=True)\n        kernel.VirtualUnlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)\n        kernel.VirtualUnlock.restype = ctypes.c_int\n        ok = kernel.VirtualUnlock(ctypes.c_void_p(mm.ctypes.data), ctypes.c_size_t(size))\n        return bool(ok) or ctypes.get_last_error() == 158  # ERROR_NOT_LOCKED\n    import mmap\n    raw = getattr(mm, "_mmap", None)\n    if raw is None or not hasattr(raw, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):\n        return False\n    raw.madvise(mmap.MADV_DONTNEED)\n    return True\n'
        )
    return source


# 8.7: без ptxas (его нельзя распространять) ядра Triton собирает драйвер NVIDIA — в cli
# только включение этого режима после set_grad_enabled; вычисления движка прежние.
DRIVER_JIT_87 = (
    "    torch.set_grad_enabled(False)\n",
    '    torch.set_grad_enabled(False)\n    if torch.cuda.is_available():  # 8.7: без ptxas ядра Triton собирает драйвер\n        from ramvi_engine.kernels.driver_jit import enable\n\n        if enable():\n            print("ядра Triton собирает драйвер NVIDIA (ptxas нет)", file=sys.stderr, flush=True)\n',
)


def driver_jit_87(name: str, source: str) -> str:
    if name != "__main__":
        return source
    old, new = DRIVER_JIT_87
    assert source.count(old) == 1, (name, old)
    return source.replace(old, new)


# B.1 CLI-HELP-01: справка и ошибки разбора параметров — ключи каталогов переводов.
CLI_HELP_B1 = (
    (
        '    ap = argparse.ArgumentParser(prog="ramvi-engine")',
        "    t = translator()\n"
        '    ap = argparse.ArgumentParser(prog="ramvi-engine", description=t("cli.engine.about"))',
    ),
    (
        'ap.error("--model обязателен для serve, chat и plan")',
        'ap.error(t("cli.engine.need_model"))',
    ),
    ('ap.error("maxt > 0, k от 0 до 15, resident >= 0")', 'ap.error(t("cli.engine.bad_numbers"))'),
)


def cli_help_b1(name: str, source: str) -> str:
    if name != "__main__":
        return source
    for old, new in CLI_HELP_B1:
        assert source.count(old) == 1, (name, old)
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
    old = tc_cfg_b6(name, source_before_move(f"{name}.py").decode("utf-8")).encode("utf-8")
    assert old == (PACKAGE_ROOT / "kernels" / f"{name}.py").read_bytes()


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
