"""Сверка HTTP-обработчиков с принятым сервером до разделения модулей."""

import ast
import copy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / "ramvi/engine/ramvi_engine/server"
BASELINE = "0759203"


def old_tree(relative: str) -> ast.Module:
    """Получить принятый исходник без его исполнения или checkout."""
    result = subprocess.run(
        ["git", "show", f"{BASELINE}:ramvi/engine/ramvi_engine/server/{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=15,
    )
    return ast.parse(result.stdout.decode("utf-8"))


def named_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    """Найти единственную функцию с заданным именем внутри выбранного владельца."""
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def named_class(tree: ast.AST, name: str) -> ast.ClassDef:
    """Найти класс приложения или локальный класс обработчика."""
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


class Computation(ast.NodeTransformer):
    """Убрать только аннотации, документацию и изменившиеся пути импорта."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        while node.args.args and node.args.args[-1].arg in {"_metrics_stats", "_request_id"}:
            node.args.args.pop()
            node.args.defaults.pop()
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *([node.args.vararg] if node.args.vararg else []),
            *([node.args.kwarg] if node.args.kwarg else []),
        ]:
            argument.annotation = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign | list[ast.Assign] | None:
        if node.value is None:
            return None
        # Callable-аннотация не меняет прежнюю lambda.
        return self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.Assign] | None:
        self.generic_visit(node)
        target = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
        if target == "self.events" and ast.unparse(node.value) == "EventLog()":
            return None
        if target == "self.metrics" and ast.unparse(node.value) == "MetricsStore()":
            return None
        if target == "self.lock" and ast.unparse(node.value) == "threading.RLock()":
            # ENG-04 удерживает тот же lock вокруг измерения и прежнего _run.
            node.value = ast.parse("threading.Lock()", mode="eval").body
        if (
            target == "st"
            and ast.unparse(node.value) == "_metrics_stats if _metrics_stats is not None else {}"
        ):
            node.value = ast.Dict(keys=[], values=[])
        # st, gen, seen = {}, [], 0 и отдельные объявления тех же локальных
        # эквивалентны: RHS не читает присваиваемые имена.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple)
            and len(node.targets[0].elts) == len(node.value.elts)
            and all(isinstance(target, ast.Name) for target in node.targets[0].elts)
            and not any(isinstance(value, ast.Name) for value in ast.walk(node.value))
        ):
            return [
                ast.Assign(targets=[target], value=value)
                for target, value in zip(node.targets[0].elts, node.value.elts, strict=True)
            ]
        return node

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt] | None:
        expression = ast.unparse(node.test)
        if expression == "getattr(self.eng, 'thinking_required', False)" and len(node.body) == 1:
            if (
                ast.unparse(node.body[0])
                == "raise ValueError('Thinking-only модель не поддерживает JSON-ограничение без reasoning adapter')"
            ):
                return None
        if expression in {
            "p == '/capabilities'",
            "p == '/engine/metrics'",
            "p == '/engine/versions'",
            "p == '/engine/events'",
            "not self._api_contract()",
            "p == '/engine/status' and isinstance(app.eng, EngineSession)",
            "p in PATHS",
            "isinstance(app.eng, EngineSession) and app.eng.status()['state'] not in ('ready', 'generating')",
        }:
            return None
        if expression == "isinstance(app.eng, EngineSession)":
            # Только новый cleanup управляемой сессии; прежний native cleanup сохранён.
            return [self.visit(statement) for statement in node.orelse]
        self.generic_visit(node)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler | None:
        if isinstance(node.type, ast.Name) and node.type.id in {
            "ModelUnavailable",
            "GenerationCancelled",
        }:
            return None
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id in {
            "with_metrics",
            "with_error_metrics",
        }:
            assert len(node.args) == 2 and not node.keywords
            result = self.visit(node.args[0])
            assert isinstance(result, ast.AST)
            return result
        node.keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg != "_event_context"
            and not (keyword.arg is None and "_generation_id" in ast.unparse(keyword.value))
        ]
        self.generic_visit(node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_version_headers"
        ):
            return None
        self.generic_visit(node)
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.Dict:
        # Only metadata's draft-loaded correction is allowed; generation remains frozen.
        for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True)):
            if (
                isinstance(key, ast.Constant)
                and key.value == "draft"
                and ast.unparse(value) == "self.effective_draft_loaded()"
            ):
                node.values[index] = ast.parse("self.draft is not None", mode="eval").body
        extra = {
            "thinking_required": "bool(getattr(engine, 'thinking_required', False))",
            "chat_template_source": "getattr(engine, 'chat_template_source', 'tokenizer')",
        }
        for index in range(len(node.keys) - 1, -1, -1):
            key = node.keys[index]
            if (
                isinstance(key, ast.Constant)
                and key.value in extra
                and ast.unparse(node.values[index]) == extra[key.value]
            ):
                node.keys.pop(index)
                node.values.pop(index)
        # Удаляется только namespaced extension; стандартные поля сверяются целиком.
        pairs = [
            (k, v)
            for k, v in zip(node.keys, node.values, strict=True)
            if not (
                isinstance(k, ast.Constant)
                and k.value in {"ramvi_metrics", "versions", "instance_id"}
            )
        ]
        node.keys = [k for k, _ in pairs]
        node.values = [v for _, v in pairs]
        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None


def normalized(function: ast.FunctionDef) -> str:
    """Сохранить все вычислительные выражения и порядок вызовов функции."""
    return ast.dump(Computation().visit(copy.deepcopy(function)), include_attributes=False)


GLOBALS = {
    "streaming.py": ["_split", "_held", "_first_stop"],
    "validation.py": [
        "_text",
        "_maybe_json",
        "_tool_result",
        "openai_messages",
        "anthropic_messages",
        "anthropic_tools",
        "_schema_from_openai",
        "_schema_from_ollama",
    ],
    "app.py": ["serve"],
}
ROUTES = {
    "routes_openai.py": ["openai_chat", "openai_completion"],
    "routes_anthropic.py": ["anthropic"],
    "routes_ollama.py": ["ollama"],
}
TRANSPORT = ["_cors", "_json", "_start_stream", "_stream_error", "_sse"]


@pytest.mark.parametrize("filename,names", GLOBALS.items())
def test_global_helpers_preserve_behavior(filename: str, names: list[str]) -> None:
    """Перемещение вспомогательных функций не меняет их выражения."""
    before, after = old_tree("app.py"), ast.parse((SERVER / filename).read_text(encoding="utf-8"))
    for name in names:
        assert normalized(named_function(before, name)) == normalized(
            named_function(after, name)
        ), name


@pytest.mark.parametrize("filename,names", ROUTES.items())
def test_routes_preserve_behavior(filename: str, names: list[str]) -> None:
    """Прежние тела обработчиков остаются замыканиями на то же приложение."""
    before = named_class(old_tree("app.py"), "H")
    after = ast.parse((SERVER / filename).read_text(encoding="utf-8"))
    for name in names:
        assert normalized(named_function(before, name)) == normalized(
            named_function(after, name)
        ), name


def test_application_and_transport_preserve_behavior() -> None:
    """Очередь, отмена, обработка HTTP и ограничения соединений перенесены целиком."""
    before = old_tree("app.py")
    after = {p.name: ast.parse(p.read_text(encoding="utf-8")) for p in SERVER.glob("*.py")}
    for old_class, filename, new_class, names in [
        ("App", "app.py", "App", ["__init__", "metadata", "build_ids", "chat_ids", "constraint"]),
        ("App", "streaming.py", "GenerationStream", ["run", "_run"]),
        ("H", "streaming.py", "StreamingResponse", TRANSPORT),
        (
            "H",
            "app.py",
            "H",
            # _auth намеренно изменён в 7.2 (ключи клиентов); проверка — test_protocol.py.
            # _post намеренно изменён в B.15: потолок тела запроса — MAX_REQUEST_BYTES вместо
            # 2 МБ; проверка — test_post_differs_only_by_request_ceiling ниже.
            ["setup", "log_message", "do_OPTIONS", "do_GET", "do_POST"],
        ),
        (
            "LimitedHTTPServer",
            "app.py",
            "LimitedHTTPServer",
            ["__init__", "process_request", "process_request_thread"],
        ),
    ]:
        first = named_class(before, old_class)
        second = named_class(after[filename], new_class)
        for name in names:
            assert normalized(named_function(first, name)) == normalized(
                named_function(second, name)
            ), name


def test_request_validation_preserves_behavior() -> None:
    """Прежняя проверка чисел и возможностей не меняется при переносе преобразований."""
    before, after = (
        old_tree("validation.py"),
        ast.parse((SERVER / "validation.py").read_text(encoding="utf-8")),
    )
    for name in ("validate_request", "validate_capabilities"):
        assert normalized(named_function(before, name)) == normalized(
            named_function(after, name)
        ), name


def test_comparison_detects_protocol_change() -> None:
    """Контроль: аннотации допустимы, изменение кода ответа обнаруживается."""
    before = named_function(ast.parse("def route(x):\n return {'status': 200}\n"), "route")
    typed = named_function(
        ast.parse("def route(x: object) -> object:\n return {'status': 200}\n"), "route"
    )
    broken = named_function(ast.parse("def route(x):\n return {'status': 400}\n"), "route")
    assert normalized(before) == normalized(typed)
    assert normalized(before) != normalized(broken)


def test_draft_metadata_allowlist_does_not_hide_other_changes() -> None:
    def tree(expression: str) -> ast.FunctionDef:
        return named_function(
            ast.parse(f"def metadata(self):\n    return {expression}\n"), "metadata"
        )

    baseline = tree("{'draft': self.draft is not None}")
    assert normalized(baseline) == normalized(tree("{'draft': self.effective_draft_loaded()}"))
    assert normalized(baseline) != normalized(tree("{'draft': self.effective_draft_loaded(1)}"))
    assert normalized(baseline) != normalized(tree("{'other': self.effective_draft_loaded()}"))
    assert normalized(baseline) != normalized(tree("{'draft': True}"))


def test_post_differs_only_by_request_ceiling() -> None:
    """B.15: в _post изменён только потолок тела запроса (2 МБ → MAX_REQUEST_BYTES)."""
    from ramvi_engine.server.app import MAX_REQUEST_BYTES

    before = named_function(named_class(old_tree("app.py"), "H"), "_post")
    after_tree = ast.parse((SERVER / "app.py").read_text(encoding="utf-8"))
    after = named_function(named_class(after_tree, "H"), "_post")
    old_text = normalized(before).replace(
        "BinOp(left=Constant(value=2), op=Mult(), right=BinOp(left=Constant(value=2), op=Pow(), right=Constant(value=20)))",
        "Name(id='MAX_REQUEST_BYTES', ctx=Load())",
    )
    assert old_text == normalized(after)
    assert MAX_REQUEST_BYTES > 2 * 2**20
