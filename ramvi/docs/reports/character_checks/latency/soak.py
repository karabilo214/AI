"""9.23: утечки — 300 реплик персонажу с тестовой моделью (без сети и денег).

Модуль Character в процессе, Scoped Services из тестов (`test_character_dialogue.Services`:
структурированный ответ без задержки). Каждые 50 реплик: память процесса (RSS), потоки,
открытые дескрипторы. Утечка — рост, который не выходит на плато.

usage: soak.py <отчёт.json>
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / p) for p in ("tests", "service", "storage", "engine", "character")]

from test_character_dialogue import Services, active, finish  # noqa: E402

TURNS = 300


def rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout
    return round(int(out.strip()) / 1024, 1)


def main() -> None:
    services = Services()
    module = active(services)
    samples = []
    started = time.monotonic()
    for turn in range(1, TURNS + 1):
        module.dialogue("start", {"text": f"Реплика номер {turn}", "model": "one"})
        result = finish(module)
        assert result["job"]["state"] == "completed", result["job"]
        if turn == 1 or turn % 50 == 0:
            samples.append(
                {
                    "turn": turn,
                    "rss_mb": rss_mb(),
                    "threads": threading.active_count(),
                    "fds": len(os.listdir("/dev/fd")),
                    "history": len(module.dialogue("status")["history"]),
                }
            )
            print(samples[-1], flush=True)
    module.dispose()
    report = {
        "turns": TURNS,
        "seconds": round(time.monotonic() - started, 1),
        "samples": samples,
        "threads_after_dispose": threading.active_count(),
        "fds_after_dispose": len(os.listdir("/dev/fd")),
    }
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
