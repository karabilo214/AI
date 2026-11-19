"""8.8: собрать пакет модуля `ramvi.module/1` из папки модуля.

    python packaging/make_module.py <папка модуля> <пакет.zip>

В папке: manifest.json и при необходимости code/, migrations/, i18n/, resources/.
В package.json пакета записываются версия Module API и SHA-256 каждого файла.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
from ramvi_service.modules.packages import build_package  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    meta = build_package(Path(sys.argv[1]), Path(sys.argv[2]))
    print(
        json.dumps({k: v for k, v in meta.items() if k != "files"} | {"files": len(meta["files"])})
    )


if __name__ == "__main__":
    main()
