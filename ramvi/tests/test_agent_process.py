"""6.5 (AG-05): дочерний процесс с лимитами времени и вывода, убийство дерева."""

import sys
import threading
import time
from pathlib import Path

import pytest
from ramvi_service.agent.process import ProcessLimits, run_process

PY = sys.executable


def _alive(pid_file: Path) -> bool:
    pid = int(pid_file.read_text(encoding="utf-8"))
    if sys.platform == "win32":
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_exit_code_and_both_streams(tmp_path: Path) -> None:
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    result = run_process([PY, "-c", code], tmp_path)
    assert result.exit_code == 3
    assert result.stdout.strip() == "out" and result.stderr.strip() == "err"
    assert not (result.timed_out or result.cancelled or result.truncated)


def test_output_is_capped_but_drained(tmp_path: Path) -> None:
    code = "import sys; sys.stdout.write('x' * 2_000_000); print('end')"
    result = run_process([PY, "-c", code], tmp_path, ProcessLimits(max_output_bytes=1000))
    assert result.exit_code == 0 and result.truncated and len(result.stdout) == 1000


def _spawn_grandchild(pid_file: Path) -> str:
    return (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open(r'{pid_file}', 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )


def test_timeout_kills_the_whole_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    started = time.monotonic()
    result = run_process(
        [PY, "-c", _spawn_grandchild(pid_file)], tmp_path, ProcessLimits(timeout=3)
    )
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - started < 20
    deadline = time.monotonic() + 5
    while _alive(pid_file) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid_file)


def test_background_children_die_after_normal_exit(tmp_path: Path) -> None:
    pid_file = tmp_path / "leftover.pid"
    code = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open(r'{pid_file}', 'w').write(str(child.pid))\n"
    )
    result = run_process([PY, "-c", code], tmp_path, ProcessLimits(timeout=30))
    assert result.exit_code == 0
    deadline = time.monotonic() + 5
    while _alive(pid_file) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid_file)


def test_cancel_stops_the_process(tmp_path: Path) -> None:
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    result = run_process(
        [PY, "-c", "import time; time.sleep(60)"],
        tmp_path,
        ProcessLimits(timeout=30),
        cancel=cancel,
    )
    assert result.cancelled and result.exit_code is None and result.duration < 20


def test_missing_program(tmp_path: Path) -> None:
    missing = [str(tmp_path / "no-such-program")]
    if sys.platform == "win32":
        # Программу запускает шлюз Job Object; его ошибка — ненулевой код выхода.
        assert run_process(missing, tmp_path).exit_code not in (0, None)
    else:
        with pytest.raises(OSError):
            run_process(missing, tmp_path)
    with pytest.raises(ValueError):
        run_process([], tmp_path)
