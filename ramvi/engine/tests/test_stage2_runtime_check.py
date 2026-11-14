"""Safety of the hardware harness itself; no hardware session is started."""

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi.engine.tools.checks import check_stage2_runtime as check


def test_absolute_deadline_requires_timezone_and_future() -> None:
    for invalid in (None, "2000-01-01T00:00:00Z", "2100-01-01T00:00:00", "bad"):
        with pytest.raises(ValueError):
            check.absolute_deadline(invalid)
    value = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
    assert check.absolute_deadline(value.isoformat()) == value


def test_help_is_light_and_does_not_need_a_hardware_deadline(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[3]
    script = """
import runpy,sys
sys.argv = ['stage2-runtime', '--help']
try:
    runpy.run_module('ramvi.engine.tools.checks.check_stage2_runtime',run_name='__main__')
except SystemExit as e:
    assert e.code == 0
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project)},
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert "--require-service" in result.stdout


class FakeProcess:
    def __init__(self, config: Any, **kwargs: Any) -> None:
        self.config = config
        self.closed = False
        self.base_url = "http://127.0.0.1:1"
        self._key = "not-a-real-credential"
        self._supervisor = None

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {"state": "stopped"}

    def diagnostics(self) -> list[str]:
        return []


@pytest.mark.parametrize("failing", [False, True])
def test_harness_reaps_its_worker_and_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: bool
) -> None:
    monkeypatch.setattr(check, "EngineProcess", FakeProcess)
    deadline = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30)
    harness = check.RuntimeCheck("fixture", 4096, tmp_path / "result.json", deadline, False)
    monkeypatch.setattr(harness, "driver_memory", lambda: 0)

    def run() -> None:
        if failing:
            raise ValueError("fixture check failed")

    monkeypatch.setattr(harness, "verify_protocol_metrics", run)
    monkeypatch.setattr(harness, "service_classes", lambda: None)
    monkeypatch.setattr(harness, "verify_restart_and_worker_crash", lambda: None)
    if failing:
        with pytest.raises(ValueError, match="fixture check failed"):
            harness.run()
    else:
        assert harness.run()["complete"]
    assert harness.worker.closed
    assert not harness._watchdog.is_alive()
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["complete"] is not failing
    assert value["driver_before_mib"] == value["driver_after_mib"] == 0
    assert not (tmp_path / "result.json.tmp").exists()


def test_missing_deadline_prevents_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAMVI_TEST_DEADLINE", raising=False)
    monkeypatch.setattr(sys, "argv", ["check", "--out", str(tmp_path / "result.json")])
    calls = []
    monkeypatch.setattr(check, "RuntimeCheck", lambda *args: calls.append(args))
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 2
    assert not calls
