"""5.4 VOI-03: ускоритель распознавания на видеокарте — отдельная загрузка (решение владельца).

Официальная сборка whisper.cpp с CUDA 12.4 (Windows x64, NVIDIA): `whisper-server` и его
библиотеки. Библиотеки CUDA из torch комплекта не подходят (там CUDA 13). Архив скачивается
с проверкой SHA-256 (`downloads.Download`), из него берутся только нужные файлы; папка
появляется под своим именем только целиком (временная папка + переименование).
"""

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ramvi_service.voice.models import FOLDER


@dataclass(frozen=True)
class AccelPack:
    id: str
    url: str
    size: int
    sha256: str
    folder: str
    files: tuple[str, ...]

    def path(self, models_dir: Path) -> Path:
        return Path(models_dir) / FOLDER / self.folder

    def archive(self, models_dir: Path) -> str:
        """Имя архива относительно папки моделей (для `Download`)."""
        return f"{FOLDER}/{self.folder}.zip"

    def installed(self, models_dir: Path) -> bool:
        folder = self.path(models_dir)
        return all((folder / name).is_file() for name in self.files)


CUDA = AccelPack(
    id="cuda-12.4",
    url=(
        "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/"
        "whisper-bin-win-cuda-12.4.0-x64.zip"
    ),
    size=684_913_404,
    sha256="afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a",
    folder="whisper-cuda-12.4-b5454",
    files=(
        "whisper-server.exe",
        "whisper.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cuda.dll",
        "ggml-cpu-x64.dll",
        "ggml-cpu-sse42.dll",
        "ggml-cpu-sandybridge.dll",
        "ggml-cpu-haswell.dll",
        "ggml-cpu-skylakex.dll",
        "ggml-cpu-cannonlake.dll",
        "ggml-cpu-cascadelake.dll",
        "ggml-cpu-icelake.dll",
        "ggml-cpu-alderlake.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
    ),
)
# Видеопамять whisper-server с моделью (замер RTX 5060, пик): turbo q5_0 — 1056 МиБ,
# small q5_1 — 726 МиБ; плюс запас, чтобы не теснить активную LLM.
VRAM_NEED_MB = {"large-v3-turbo-q5_0": 1056 + 256, "small-q5_1": 726 + 256}


def offered(platform: str, has_nvidia: bool) -> bool:
    """Ускоритель предлагается только на Windows с NVIDIA (сборка — только под Windows x64)."""
    return platform == "win32" and has_nvidia


def extract(pack: AccelPack, archive: Path, models_dir: Path) -> Path:
    """Распаковать нужные файлы архива; при ошибке прежнее состояние не меняется."""
    target = pack.path(models_dir)
    temp = target.with_name(target.name + ".tmp")
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = {PurePosixPath(m.filename).name: m for m in zf.infolist() if not m.is_dir()}
            missing = [name for name in pack.files if name not in members]
            if missing:
                raise OSError(f"archive has no {', '.join(missing)}")
            for name in pack.files:
                with zf.open(members[name]) as source, (temp / name).open("wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
        shutil.rmtree(target, ignore_errors=True)
        temp.rename(target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    archive.unlink(missing_ok=True)
    return target
