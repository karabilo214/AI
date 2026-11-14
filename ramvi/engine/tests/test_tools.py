"""Проверки диспетчера инструментов без моделей и видеокарты."""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi.engine.tools import __main__ as launcher


def test_help_does_not_import_compute_modules(tmp_path: Path) -> None:
    """Общий список команд доступен в лёгком окружении."""
    code = """
import runpy,sys
sys.argv = ['ramvi-tools', '--help']
try:
    runpy.run_module('ramvi.engine.tools', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=launcher.ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    assert "{check,bench,unit}" in result.stdout


def test_registry_points_to_existing_sources() -> None:
    """Опубликованные команды не ссылаются на забытые при переносе файлы."""
    checks = launcher.ROOT / "ramvi/engine/tools/checks"
    for name in launcher.CHECKS:
        filename = name if name in ("selftest", "api_test", "native_pool") else "check_" + name
        assert (checks / (filename + ".py")).is_file()


def test_launcher_forwards_args_and_stops_timed_out_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Внутренние параметры передаются без потери, таймаут останавливает дерево."""
    commands: list[list[str]] = []
    waits: list[float] = []
    killed: list[int] = []

    class Child:
        pid = 12345

        def wait(self, timeout: float) -> int:
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

    def spawn(command: list[str], **kwargs: Any) -> Child:
        commands.append(command)
        assert kwargs["cwd"] == launcher.ROOT
        return Child()

    monkeypatch.setattr(
        sys, "argv", ["tools", "--timeout", "3", "bench", "wide", "--", "--compare", "a", "b"]
    )
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher, "kill_tree", killed.append)
    assert launcher.main() == 124
    assert commands == [
        [sys.executable, "-m", "ramvi.engine.tools.bench.bench_wide", "--compare", "a", "b"]
    ]
    assert waits == [3, 15]
    assert killed == [12345]


def test_benchmark_data_and_validation_remain_available() -> None:
    """Набор WIDE30 и проверка результатов доступны без старого пакета."""
    module = importlib.import_module("ramvi.engine.tools.bench.bench_wide")
    assert len(module.WIDE) == 30
    validation = importlib.import_module("ramvi.engine.tools.bench.bench_validation")
    with pytest.raises(ValueError):
        validation.paired_rows([], [], "q")


def test_help_supports_redirected_windows_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Русская справка диспетчера не падает в Windows pipe с исходной CP1252."""
    import io

    buffer = io.BytesIO()
    with io.TextIOWrapper(buffer, encoding="cp1252", write_through=True) as stream:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "platform", "win32")
            patch.setattr(sys, "argv", ["tools", "--help"])
            patch.setattr(sys, "stdout", stream)
            with pytest.raises(SystemExit) as result:
                launcher.main()
            assert result.value.code == 0
        assert "Единый запуск" in buffer.getvalue().decode("utf-8")
