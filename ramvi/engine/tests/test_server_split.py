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

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.Assign]:
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
    before, after = old_tree("app.py"), ast.parse((SERVER / filename).read_text())
    for name in names:
        assert normalized(named_function(before, name)) == normalized(
            named_function(after, name)
        ), name


@pytest.mark.parametrize("filename,names", ROUTES.items())
def test_routes_preserve_behavior(filename: str, names: list[str]) -> None:
    """Прежние тела обработчиков остаются замыканиями на то же приложение."""
    before = named_class(old_tree("app.py"), "H")
    after = ast.parse((SERVER / filename).read_text())
    for name in names:
        assert normalized(named_function(before, name)) == normalized(
            named_function(after, name)
        ), name


def test_application_and_transport_preserve_behavior() -> None:
    """Очередь, отмена, обработка HTTP и ограничения соединений перенесены целиком."""
    before = old_tree("app.py")
    after = {p.name: ast.parse(p.read_text()) for p in SERVER.glob("*.py")}
    for old_class, filename, new_class, names in [
        ("App", "app.py", "App", ["__init__", "metadata", "build_ids", "chat_ids", "constraint"]),
        ("App", "streaming.py", "GenerationStream", ["run", "_run"]),
        ("H", "streaming.py", "StreamingResponse", TRANSPORT),
        (
            "H",
            "app.py",
            "H",
            ["setup", "log_message", "_auth", "do_OPTIONS", "do_GET", "do_POST", "_post"],
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
    before, after = old_tree("validation.py"), ast.parse((SERVER / "validation.py").read_text())
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
