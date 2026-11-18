"""Диагностика RAMVI внутри Windows Sandbox: свежий код сервиса из подключённой папки."""

import json
import sys

sys.path[:0] = [r"C:\in\service", r"C:\in\storage", r"C:\RAMVI\ramvi\engine"]

from ramvi_service.hardware import collect  # noqa: E402
from ramvi_service.onboarding import checks, probe_backends  # noqa: E402

backends = probe_backends()
print(json.dumps(backends, ensure_ascii=False))
for item in checks(collect(r"C:\RAMVI"), backends):
    print(json.dumps(item, ensure_ascii=False))
