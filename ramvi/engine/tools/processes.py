"""Проверка занятости Windows и остановка дерева дочерних процессов."""

import base64
import os
import signal
import subprocess
import sys


def powershell(script: str) -> str:
    """Выполнить короткий PowerShell-опрос с ограниченным ожиданием."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    result = subprocess.run(
        ["powershell", "-NoProfile", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=15,
        check=True,
    )
    return result.stdout.decode("utf-8-sig", errors="replace").strip()


def available() -> None:
    """Отклонить Windows-проверку при запущенной игре или другом сервере модели."""
    if os.name != "nt":
        return
    result = powershell(
        "Get-Process | Where-Object { $_.ProcessName -match "
        "'^(WorldOfTanks.*|ollama.*|llama-server)$' } | "
        "Select-Object ProcessName,Id | ConvertTo-Json -Compress"
    )
    if result:
        raise RuntimeError("device busy: " + result)


def kill_tree(pid: int) -> None:
    """Остановить дочернее дерево; POSIX-процесс должен иметь собственную сессию."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    else:
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
