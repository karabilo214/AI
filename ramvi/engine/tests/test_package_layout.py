"""Контракты нового пакета и переходного слоя, не требующие видеокарты."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ENGINE_ROOT.parents[1]
PACKAGE_ROOT = ENGINE_ROOT / "ramvi_engine"


def test_package_and_cli_help_do_not_import_compute_libraries(tmp_path: Path) -> None:
    """Пакет и справка работают вне репозитория, без загрузки Torch/Triton."""
    code = """
import runpy, sys
import ramvi_engine
assert 'torch' not in sys.modules and 'triton' not in sys.modules
sys.argv = ['ramvi-engine', '--help']
try:
    runpy.run_module('ramvi_engine', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ENGINE_ROOT)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    assert "ramvi-engine" in result.stdout
    assert "{serve,chat,plan}" in result.stdout


def test_module_and_direct_cli_help_match() -> None:
    """Модульный вход и основная CLI-функция принимают одинаковые параметры."""
    outputs = []
    commands = (
        [sys.executable, "-m", "ramvi_engine", "--help"],
        [sys.executable, "-c", "from ramvi_engine.cli import main; main()", "--help"],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(ENGINE_ROOT)},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("label", "canonical"),
    [
        ("server", "server.app"),
        ("request_validation", "server.validation"),
        ("gguf", "formats.gguf"),
        ("gcpu", "cpu.gcpu"),
        ("constrain", "generation.constrain"),
        ("measured_plan", "planning.measured_plan"),
    ],
)
def test_canonical_module_imports_without_legacy(label: str, canonical: str) -> None:
    """Канонические модули загружаются из пакета без прежних переходных файлов."""
    if label not in ("server", "request_validation"):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        if label == "gcpu":
            pytest.importorskip("numba")
    module = importlib.import_module(f"ramvi_engine.{canonical}")
    assert module.__name__ == f"ramvi_engine.{canonical}"
    assert Path(module.__file__ or "").resolve().is_relative_to(PACKAGE_ROOT)
    assert "llm_app" not in sys.modules


def test_native_loader_uses_package_directory_and_verifies_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """После переноса DLL ищется рядом с native, а чужой исходник отклоняется."""
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("numba")
    module = importlib.import_module("ramvi_engine.cpu.gcpu")
    package = tmp_path / "ramvi_engine"
    cpu = package / "cpu"
    native = package / "native"
    cpu.mkdir(parents=True)
    native.mkdir()
    source = native / "qmm.cpp"
    library = native / "qmm.dll"
    source.write_bytes(b"source fixture")
    library.write_bytes(b"binary fixture")
    library.with_suffix(".dll.build.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        )
    )
    seen: list[str] = []

    def intercept(path: str) -> NoReturn:
        seen.append(path)
        raise RuntimeError("native load reached")

    monkeypatch.setattr(module, "__file__", str(cpu / "gcpu.py"))
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(module, "_DLL", None)
    monkeypatch.setattr(module, "_cpp_cpu_supported", lambda: True)
    monkeypatch.setattr(module.ctypes, "CDLL", intercept)
    monkeypatch.delenv("LLM_NO_CPP", raising=False)
    monkeypatch.delenv("RAMVI_NO_CPP", raising=False)
    with pytest.raises(RuntimeError, match="native load reached"):
        module._dll()
    assert seen == [str(library)]
    seen.clear()
    source.write_bytes(b"different source")
    with pytest.raises(RuntimeError, match="не соответствует исходнику"):
        module._dll()
    assert seen == []


def test_internal_imports_are_absolute_and_independent_of_legacy() -> None:
    """Пакет не зависит от переходного llm_app и относительных путей импорта."""
    bad: list[str] = []
    for file in PACKAGE_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (
                node.level or (node.module or "").startswith("llm_app")
            ):
                bad.append(f"{file.relative_to(PACKAGE_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("llm_app") for alias in node.names
            ):
                bad.append(f"{file.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert not bad, bad


def test_metadata_hash_covers_nested_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hash сервера замечает правки ядра и учитывает относительные пути файлов."""
    module = importlib.import_module("ramvi_engine.server.app")
    package = tmp_path / "ramvi_engine"
    server = package / "server"
    core = package / "core"
    server.mkdir(parents=True)
    core.mkdir()
    app_source = server / "app.py"
    engine_source = core / "engine.py"
    app_source.write_bytes(b"# server\n")
    engine_source.write_bytes(b"# core version 1\n")
    monkeypatch.setattr(module, "__file__", str(app_source))
    app = module.App(SimpleNamespace(maxt=2048, place=["gpu"]), None, "fixture")

    def digest() -> str:
        value = app.metadata()["code_sha256"]
        assert isinstance(value, str)
        return value

    original = digest()
    expected = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        expected.update(source.relative_to(package).as_posix().encode())
        expected.update(source.read_bytes())
    assert original == expected.hexdigest()
    engine_source.write_bytes(b"# core version 2\n")
    changed = digest()
    assert changed != original
    engine_source.rename(core / "renamed.py")
    assert digest() != changed
    (core / "ignored.log").write_bytes(b"runtime log")
    before_log_change = digest()
    (core / "ignored.log").write_bytes(b"different runtime log")
    assert digest() == before_log_change


def test_windows_output_preserves_russian_in_redirected_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Контроль CP1252 воспроизводит Windows pipe без зависимости от locale Mac."""
    import io

    module = importlib.import_module("ramvi_engine.console")
    output_bytes, error_bytes = io.BytesIO(), io.BytesIO()
    with (
        io.TextIOWrapper(output_bytes, encoding="cp1252", write_through=True) as output,
        io.TextIOWrapper(error_bytes, encoding="cp1252", write_through=True) as error,
    ):
        monkeypatch.setattr(
            module, "sys", SimpleNamespace(platform="win32", stdout=output, stderr=error)
        )
        module.configure_output()
        output.write("Справка RAMVI")
        error.write("Ошибка RAMVI")
        assert output_bytes.getvalue().decode("utf-8") == "Справка RAMVI"
        assert error_bytes.getvalue().decode("utf-8") == "Ошибка RAMVI"


def test_legacy_package_is_absent() -> None:
    """Завершённый этап 1 не оставляет старый пакет в рабочем дереве."""
    assert not (PROJECT_ROOT / "llm_app").exists()
