"""3.9 (PORT-01/PORT-02): собрать переносимую папку RAMVI для Windows x64.

Запускать тем Python, чьё окружение проверено (зависимости берутся из него как есть):

    python packaging/build_portable.py --out D:/RAMVI --shell <публикация оболочки>
        --python-embed python-3.12.10-embed-amd64.zip --tokenizer <папка токенизатора Qwen3-8B>

Раскладка:

    RAMVI/
      RAMVI.exe …                    оболочка (dotnet publish -r win-x64 --self-contained)
      ramvi/service|engine|storage   код RAMVI без тестов
      runtime/python/                встроенный Python + Lib/site-packages (замыкание зависимостей)
      runtime/tokenizers/Qwen3-8B/   файлы токенизатора без весов
      BUILD.json                     состав комплекта

`settings.json` и `data/` не создаются: появятся при первом запуске (если папка доступна
на запись; иначе пользователь выберет папку данных, пункт 3.9).
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ("service/ramvi_service", "engine/ramvi_engine", "storage/ramvi_storage")
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "build", "*.egg-info")
# Прямые зависимости движка (engine/pyproject.toml), triton для ядер на GPU и голос (5.2).
ROOTS = (
    "torch",
    "numpy",
    "numba",
    "transformers",
    "safetensors",
    "xgrammar",
    "jsonschema",
    "triton",
    # 5.2 (ADR-0004): голосовой ввод — запись звука (PortAudio) и whisper.cpp, только CPU
    "sounddevice",
    "pywhispercpp",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "config.json",
)


def _requirement_tools() -> Any:
    try:
        from packaging.requirements import Requirement
    except ImportError:  # окружение без packaging: тот же код внутри pip
        from pip._vendor.packaging.requirements import Requirement  # type: ignore[assignment]
    return Requirement


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def find_distribution(name: str) -> md.Distribution | None:
    """Установленный пакет по имени; `triton` на Windows ставится как `triton-windows`."""
    for candidate in (name, f"{name}-windows"):
        try:
            return md.distribution(candidate)
        except md.PackageNotFoundError:
            continue
    return None


def closure(roots: tuple[str, ...]) -> dict[str, md.Distribution]:
    """Пакеты, нужные корням в этом окружении: маркеры платформы учитываются, extras — нет."""
    requirement = _requirement_tools()
    found: dict[str, md.Distribution] = {}
    queue = list(roots)
    while queue:
        name = queue.pop()
        dist = find_distribution(name)
        if dist is None:
            if name in roots:
                raise SystemExit(f"не установлен корневой пакет: {name}")
            continue
        key = _normal(dist.metadata["Name"])
        if key in found:
            continue
        found[key] = dist
        for text in dist.requires or []:
            req = requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            queue.append(req.name)
    return dict(sorted(found.items()))


def copy_distribution(dist: md.Distribution, site: Path) -> tuple[int, int]:
    """Файлы пакета внутри site-packages (скрипты из ../Scripts не нужны)."""
    files = count = size = 0
    for item in dist.files or []:
        files += 1
        if item.parts and item.parts[0] == "..":
            continue
        source = Path(str(dist.locate_file(item)))
        if not source.is_file():
            continue
        target = site / Path(*item.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
        size += source.stat().st_size
    return count, size


def install_python(embed: Path, target: Path) -> Path:
    """Распаковать встроенный Python и включить Lib/site-packages и модуль site."""
    with zipfile.ZipFile(embed) as archive:
        archive.extractall(target)
    pth = next(target.glob("python3*._pth"), None)
    if pth is None:
        raise SystemExit("в архиве нет python3XX._pth — это не встроенный Python для Windows")
    lines = [line for line in pth.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [line for line in lines if line.strip().lstrip("#").strip() != "import site"]
    # Встроенный Python с ._pth не читает PYTHONPATH: код RAMVI указывается здесь же,
    # относительными путями от папки python (переживают смену буквы диска).
    code = [f"..\\..\\ramvi\\{part.split('/')[0]}" for part in CODE]
    pth.write_text(
        "\n".join([*lines, "Lib\\site-packages", *code, "import site", ""]), encoding="utf-8"
    )
    site = target / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


DEV_DIRS = ("Include", "libs")  # заголовки и python3XX.lib: triton компилирует C-модуль при запуске


def copy_dev_files(base: Path, target: Path) -> list[str]:
    """Заголовки и библиотеки импорта базового Python (во встроенном их нет).

    Triton на Windows собирает `cuda_utils` встроенным tcc с `-I<python>/Include`
    и `-lpython312` из `<python>/libs`; без них движок падает при старте.
    """
    copied = []
    for name in DEV_DIRS:
        source = base / name
        if source.is_dir():
            shutil.copytree(source, target / name, dirs_exist_ok=True)
            copied.append(name)
    return copied


# 8.1 (PORT-02): библиотеки Microsoft Visual C++ 2015–2022 (x64) рядом с python.exe.
# Без них на чистой Windows torch не загружает c10.dll (найдено в Windows Sandbox);
# Microsoft разрешает распространять эти файлы вместе с приложением.
VC_RUNTIME = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
    "concrt140.dll",
    "vcomp140.dll",
)


def copy_vc_runtime(system32: Path, target: Path) -> list[str]:
    """Положить библиотеки Visual C++ рядом с python.exe; нет хотя бы одной — сборка
    останавливается: неполный комплект не должен уйти пользователю."""
    missing = [name for name in VC_RUNTIME if not (system32 / name).is_file()]
    if missing:
        raise SystemExit(
            "нет библиотек Visual C++ для комплекта: "
            + ", ".join(missing)
            + " — установите Visual C++ Redistributable x64 на машине сборки"
        )
    for name in VC_RUNTIME:
        shutil.copy2(system32 / name, target / name)
    return list(VC_RUNTIME)


def code_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build(
    out: Path,
    shell: Path,
    embed: Path,
    tokenizer: Path,
    roots: tuple[str, ...] = ROOTS,
    source: Path = ROOT,
) -> dict[str, Any]:
    if out.exists():
        raise SystemExit(f"папка уже существует: {out} (комплект собирается в новую папку)")
    if not (shell / "RAMVI.exe").is_file():
        raise SystemExit(f"в {shell} нет RAMVI.exe — нужна публикация оболочки под win-x64")
    started = time.time()
    shutil.copytree(shell, out)
    for part in CODE:
        shutil.copytree(source / part, out / "ramvi" / part, ignore=SKIP)
    # 3.14: каталог переводов — сервису (справка CLI); оболочка держит его в себе.
    shutil.copytree(source / "i18n", out / "ramvi" / "i18n")
    if (source / "modules").is_dir():
        shutil.copytree(source / "modules", out / "modules", ignore=SKIP)
    site = install_python(embed, out / "runtime" / "python")
    dev = (
        copy_dev_files(Path(sys.base_prefix), out / "runtime" / "python")
        if sys.platform == "win32"
        else []
    )
    if sys.platform == "win32":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        copy_vc_runtime(system32, out / "runtime" / "python")
    packages: dict[str, str] = {}
    copied = size = 0
    for key, dist in closure(roots).items():
        count, bytes_ = copy_distribution(dist, site)
        packages[key] = dist.version
        copied, size = copied + count, size + bytes_
    tokens = out / "runtime" / "tokenizers" / "Qwen3-8B"
    tokens.mkdir(parents=True)
    taken = [name for name in TOKENIZER_FILES if (tokenizer / name).is_file()]
    if "tokenizer.json" not in taken and "vocab.json" not in taken:
        raise SystemExit(f"в {tokenizer} нет файлов токенизатора")
    for name in taken:
        shutil.copy2(tokenizer / name, tokens / name)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_embed": embed.name,
        "source_python": sys.version.split()[0],
        "packages": packages,
        "package_files": copied,
        "package_bytes": size,
        "tokenizer_files": taken,
        "dev_dirs": dev,
        "code_sha256": code_digest(out / "ramvi"),
        "total_bytes": total,
        "seconds": round(time.time() - started, 1),
    }
    (out / "BUILD.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="build_portable", description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="новая папка комплекта")
    parser.add_argument(
        "--shell", required=True, type=Path, help="публикация оболочки (с RAMVI.exe)"
    )
    parser.add_argument(
        "--python-embed", required=True, type=Path, help="архив python-3.12.x-embed-amd64.zip"
    )
    parser.add_argument(
        "--tokenizer", required=True, type=Path, help="папка с файлами токенизатора Qwen3-8B"
    )
    args = parser.parse_args()
    manifest = build(args.out, args.shell, args.python_embed, args.tokenizer)
    print(json.dumps({k: v for k, v in manifest.items() if k != "packages"}, ensure_ascii=False))
    print(f"пакетов: {len(manifest['packages'])}")


if __name__ == "__main__":
    main()
