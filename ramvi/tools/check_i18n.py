"""3.14 (CORE-I18N-01): проверка переводов для CI — `python tools/check_i18n.py`.

Сборка падает (код 1), если:
1. каталоги ядра `i18n/<язык>.json` или модулей `modules/<id>/i18n/` нарушают правила
   (`ramvi_service.i18n.check_catalogs`): нет языка, разные ключи, пустой перевод,
   разные параметры; ключ модуля вне его пространства или совпадает с ключом ядра;
2. ключ, который код использует, отсутствует в каталоге: строки-ключи в Python
   сервиса и хранилища, `L["…"]`, `L.Format("…")` и прочие ключи в C# оболочки,
   `{l:T …}` в разметке;
3. в коде есть пользовательская строка вне каталога: русский текст в строковом
   литерале Python (кроме docstring), C# или в разметке (кроме комментариев).
   Сообщения для разработчика пишутся по-английски, тексты для людей — ключами.

Движок (`engine/`) — отдельный backend: его сообщения сервис переводит в коды
(`engines.classify_failure`), а справка `ramvi-engine` — пункт 3.15.
"""

import ast
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ramvi_service.i18n import KEY, LANGUAGES, check_catalogs, load_catalog  # noqa: E402

PYTHON = ("service/ramvi_service", "storage/ramvi_storage")
CSHARP = ("desktop/src",)
CYRILLIC = re.compile("[А-Яа-яЁё]")
# Похоже на ключ, но это имя файла или модуля.
NOT_KEY = re.compile(r".*\.(json|sql|py|gguf|zip|txt|md|part|lock|png|ico|exe|dll|axaml|cs)")
CS_STRING = re.compile(r'\$?@?"((?:[^"\\\n]|\\.)*)"')
XAML_KEY = re.compile(r"\{l:T\s+([^}\s]+)\s*\}")
# Похоже на ключ, но это имя метода Scoped Services (services.py), а не текст.
NOT_KEYS = frozenset({"settings.get", "settings.set"})


def core_catalogs(root: Path = ROOT) -> dict[str, dict[str, str]]:
    return {lang: load_catalog(lang, root / "i18n") for lang in LANGUAGES}


def catalog_errors(root: Path = ROOT) -> list[str]:
    core = core_catalogs(root)
    errors = check_catalogs(core, where="i18n")
    for folder in sorted((root / "modules").glob("*/i18n")):
        module_id = folder.parent.name
        catalogs = {p.stem: load_catalog(p.stem, folder) for p in sorted(folder.glob("*.json"))}
        allowed = re.compile(rf"section\.{module_id}|module\.{module_id}\.[a-z0-9_.]+")
        where = f"modules/{module_id}"
        errors += check_catalogs(catalogs, where=where, allowed=allowed)
        clash = set().union(*catalogs.values()) & core[LANGUAGES[0]].keys()
        errors += [
            f"{where}: key {key} already exists in the core catalog" for key in sorted(clash)
        ]
    return errors


def _files(root: Path, folders: tuple[str, ...], suffix: str) -> Iterator[Path]:
    for folder in folders:
        for path in sorted((root / folder).rglob(f"*{suffix}")):
            if not {"bin", "obj", "build", "__pycache__", "tests"} & set(path.parts):
                yield path


def _docstrings(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            found.add(id(body[0].value))
    return found


def python_strings(root: Path = ROOT) -> Iterator[tuple[Path, int, str, bool]]:
    """(файл, строка, текст, это docstring) для каждого строкового литерала."""
    for path in _files(root, PYTHON, ".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield path, node.lineno, node.value, id(node) in docs


def _strip_cs_comment(line: str) -> str:
    """Строка C# без комментария `//` (вне строковых литералов)."""
    quoted = False
    i = 0
    while i < len(line):
        char = line[i]
        if char == "\\" and quoted:
            i += 2
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted and line.startswith("//", i):
            return line[:i]
        i += 1
    return line


def csharp_strings(root: Path = ROOT) -> Iterator[tuple[Path, int, str]]:
    for path in _files(root, CSHARP, ".cs"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in CS_STRING.finditer(_strip_cs_comment(line)):
                yield path, number, match.group(1)


def xaml_lines(root: Path = ROOT) -> Iterator[tuple[Path, int, str]]:
    """Строки разметки без комментариев `<!-- … -->`."""
    for path in _files(root, CSHARP, ".axaml"):
        text = re.sub(
            r"<!--.*?-->",
            lambda m: "\n" * m.group(0).count("\n"),
            path.read_text(encoding="utf-8"),
            flags=re.S,
        )
        for number, line in enumerate(text.splitlines(), 1):
            yield path, number, line


def _is_key(text: str, namespaces: set[str]) -> bool:
    return (
        bool(KEY.fullmatch(text))
        and text.split(".")[0] in namespaces
        and not NOT_KEY.fullmatch(text)
        and text not in NOT_KEYS
    )


def used_keys(root: Path = ROOT) -> list[tuple[str, str]]:
    """(ключ, где) — ключи, которые код берёт из каталога."""
    namespaces = {key.split(".")[0] for key in core_catalogs(root)[LANGUAGES[0]]}
    found: list[tuple[str, str]] = []
    for path, line, text, doc in python_strings(root):
        if not doc and _is_key(text, namespaces):
            found.append((text, f"{path.relative_to(root).as_posix()}:{line}"))
    for path, line, text in csharp_strings(root):
        if _is_key(text, namespaces):
            found.append((text, f"{path.relative_to(root).as_posix()}:{line}"))
    for path, line, text in xaml_lines(root):
        for match in XAML_KEY.finditer(text):
            found.append((match.group(1), f"{path.relative_to(root).as_posix()}:{line}"))
    return found


def missing_keys(root: Path = ROOT) -> list[str]:
    known = core_catalogs(root)[LANGUAGES[0]]
    return [
        f"{where}: key {key} is not in i18n/" for key, where in used_keys(root) if key not in known
    ]


def hardcoded_texts(root: Path = ROOT) -> list[str]:
    errors = []
    for path, line, text, doc in python_strings(root):
        if not doc and CYRILLIC.search(text):
            errors.append(
                f"{path.relative_to(root).as_posix()}:{line}: user text outside the catalog: {text[:60]!r}"
            )
    for path, line, text in csharp_strings(root):
        if CYRILLIC.search(text):
            errors.append(
                f"{path.relative_to(root).as_posix()}:{line}: user text outside the catalog: {text[:60]!r}"
            )
    for path, line, text in xaml_lines(root):
        if CYRILLIC.search(text):
            errors.append(
                f"{path.relative_to(root).as_posix()}:{line}: user text outside the catalog: {text.strip()[:60]!r}"
            )
    return errors


def main() -> int:
    errors = catalog_errors() + missing_keys() + hardcoded_texts()
    for error in errors:
        print(error)
    keys = len(core_catalogs()[LANGUAGES[0]])
    print(json.dumps({"languages": list(LANGUAGES), "keys": keys, "errors": len(errors)}))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
