"""Единый запуск инструментов из корня репозитория, без загрузки CUDA в диспетчере."""

import argparse
import math
import subprocess
import sys
from pathlib import Path

from ramvi.engine.tools.processes import kill_tree

ROOT = Path(__file__).resolve().parents[3]
CHECKS = (
    "audit_api",
    "audit_gpu",
    "audit_metadata",
    "audit_moe_recovery",
    "audit_recovery",
    "audit_sampling",
    "gcpu",
    "gengine",
    "gguf",
    "gspec",
    "measured_plan",
    "pool",
    "prefill",
    "single_token",
    "switch",
    "selftest",
    "api_test",
    "native_pool",
)
BENCHES = ("gsm8k", "wide", "o02", "compare", "report_migration")


def main() -> int:
    """Выбрать проверку, передать её параметры и ограничить время дочернего процесса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=600, help="лимит времени в секундах")
    sub = parser.add_subparsers(dest="group", required=True)
    for group, choices in (("check", CHECKS), ("bench", BENCHES)):
        item = sub.add_parser(group)
        item.add_argument("name", choices=choices)
        item.add_argument("args", nargs=argparse.REMAINDER, help="параметры выбранного инструмента")
    unit = sub.add_parser("unit", help="все модульные тесты; параметры передаются pytest")
    unit.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout должен быть больше нуля")
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    if args.group == "unit":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "ramvi/pyproject.toml",
            "ramvi/tests",
            "ramvi/engine/tests",
            *forwarded,
        ]
    else:
        name = args.name
        if args.group == "check":
            module = name if name in ("selftest", "api_test", "native_pool") else "check_" + name
            package = "checks"
        else:
            module = {
                "gsm8k": "bench",
                "wide": "bench_wide",
                "o02": "bench_o02",
                "compare": "compare_bench",
            }.get(name, name)
            package = "bench"
        command = [sys.executable, "-m", f"ramvi.engine.tools.{package}.{module}", *forwarded]
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=sys.platform != "win32")
    try:
        return child.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        kill_tree(child.pid)
        child.wait(timeout=15)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
