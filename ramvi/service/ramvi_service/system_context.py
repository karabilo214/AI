"""Системный адаптер выбранного окна. Windows: UI Automation/PrintWindow.

На неподдержанной ОС — явная недоступность. Нет фонового опроса, снимков всего
экрана, подстановки команд из модели или автоматической сетевой передачи.
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from ramvi_service.agent.process import ProcessLimits, run_process


class SystemContextAdapter:
    def capabilities(self) -> dict[str, Any]:
        supported = sys.platform == "win32"
        return {
            "ui": supported,
            "screen": supported,
            "control": supported,
            "backend": "windows_uia" if supported else None,
        }

    def _request(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if sys.platform != "win32":
            raise NotImplementedError("system context adapter is not available on this OS")
        if cancel.is_set():
            raise InterruptedError("context cancelled")
        script = Path(__file__).parent / "native" / "windows_context.ps1"
        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / (
            "System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        # JSON never goes into the command line or logs. File is private and removed
        # even on cancellation. The helper is fixed product code, not model code.
        with tempfile.TemporaryDirectory(prefix="ramvi-context-") as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            path.chmod(0o600)
            result = run_process(
                [
                    str(powershell),
                    "-NoProfile",
                    "-MTA",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-RequestFile",
                    str(path),
                ],
                Path(directory),
                ProcessLimits(timeout=10, max_output_bytes=2 * 2**20),
                cancel=cancel,
            )
        if result.cancelled or cancel.is_set():
            raise InterruptedError("context cancelled")
        if result.timed_out:
            raise TimeoutError("system context adapter timed out")
        if result.exit_code != 0 or result.truncated:
            raise NotImplementedError("system context is unavailable or exceeds budget")
        response = json.loads(result.stdout.lstrip("\ufeff"))
        if not isinstance(response, dict) or not isinstance(response.get("result"), dict):
            raise ValueError("invalid system context response")
        return dict(response["result"])

    def current_target(self, *, describe: bool = False) -> dict[str, Any]:
        """Только доверенный UI выбора цели. Текст/изображение не собираются."""
        return self._request(
            {"operation": "target_title" if describe else "target"}, threading.Event()
        )

    def capture(
        self, application: str, window: str, kind: str, cancel: threading.Event
    ) -> dict[str, Any]:
        if kind not in ("ui", "screen"):
            raise ValueError("unknown context kind")
        return self._request(
            {"operation": "capture", "application": application, "window_id": window, "kind": kind},
            cancel,
        )

    def act(
        self,
        application: str,
        window: str,
        element_id: str,
        action: str,
        value: str | None,
        fingerprint: str,
        cancel: threading.Event,
    ) -> dict[str, Any]:
        if action not in ("invoke", "set_value"):
            raise ValueError("unsupported UI action")
        result = self._request(
            {
                "operation": "act",
                "application": application,
                "window_id": window,
                "element_id": element_id,
                "action": action,
                "value": value,
                "fingerprint": fingerprint,
            },
            cancel,
        )
        if result.get("applied") is not True:
            raise ValueError("UI action outcome is unknown")
        return result
