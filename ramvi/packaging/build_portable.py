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
import csv
import hashlib
import importlib.metadata as md
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "service/ramvi_service",
    "engine/ramvi_engine",
    "storage/ramvi_storage",
    "character/ramvi_character",
)
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


def _record(dist: md.Distribution) -> list[md.PackagePath]:
    """Все пути из RECORD. `dist.files` в Python 3.12 молча выбрасывает отсутствующие на диске
    пути — так терялись DLL pywhispercpp, записанные как `<пакет>.data/platlib/x.dll` (8.7)."""
    text = dist.read_text("RECORD")
    if text is None:
        return list(dist.files or [])
    return [md.PackagePath(row[0]) for row in csv.reader(text.splitlines()) if row and row[0]]


def copy_distribution(dist: md.Distribution, site: Path) -> tuple[int, int]:
    """Файлы пакета внутри site-packages (скрипты из ../Scripts не нужны)."""
    files = count = size = 0
    for item in _record(dist):
        files += 1
        if item.parts and item.parts[0] == "..":
            continue
        parts = item.parts
        source = Path(str(dist.locate_file(item)))
        if (
            not source.is_file()
            and len(parts) > 2
            and parts[0].endswith(".data")
            and parts[1] in ("platlib", "purelib")
        ):
            # 8.7: RECORD с путями из колеса (`<пакет>.data/platlib/x.dll`), а файл установлен
            # в корень site-packages — так у pywhispercpp; без этого его DLL терялись.
            parts = parts[2:]
            source = Path(str(dist.locate_file(Path(*parts))))
        if not source.is_file():
            continue
        target = site / Path(*parts)
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


# 8.1 (PORT-02): библиотеки Microsoft Visual C++ рядом с python.exe (без них на чистой Windows
# torch не загружает c10.dll). 8.7: берутся только из каталога VC\Redist установленной Visual
# Studio / Build Tools, с проверкой подписи и без изменений — см. redist.py.
VC_RUNTIME = redist.VC_RUNTIME


# 8.7: что не нужно движку из пакета torch (CUDA 13, Windows). Найдено по таблицам импорта DLL
# (`import torch` статически тянет 16 библиотек из torch/lib) и по коду движка: свёрток, FFT,
# решателей и cuDNN-внимания нет — ядра свои (Triton) и cuBLAS. cudnn64_9.dll остаётся
# (её импортируют torch_cuda и torch_python), уходят только её подбиблиотеки, которые она
# подгружает при первом вызове cuDNN. Заголовки и .lib нужны только для сборки расширений
# torch; Triton берёт свои из triton/ и python/Include. Пока --trim не проверен на Windows с
# настоящей моделью, он выключен по умолчанию.
TRIM_FILES = (
    "torch/lib/cudnn_adv64_9.dll",
    "torch/lib/cudnn_cnn64_9.dll",
    "torch/lib/cudnn_engines_precompiled64_9.dll",
    "torch/lib/cudnn_engines_runtime_compiled64_9.dll",
    "torch/lib/cudnn_engines_tensor_ir64_9.dll",
    "torch/lib/cudnn_ext64_9.dll",
    "torch/lib/cudnn_graph64_9.dll",
    "torch/lib/cudnn_heuristic64_9.dll",
    "torch/lib/cudnn_ops64_9.dll",
    "torch/lib/cusolverMg64_12.dll",
    "torch/lib/cufftw64_12.dll",
    "torch/lib/nvperf_host.dll",
)
TRIM_DIRS = ("torch/include", "torch/share")  # torchgen не трогать: его импортирует import torch
TRIM_GLOBS = ("torch/lib/*.lib",)


def trim_site(site: Path) -> dict[str, Any]:
    """Удалить из site-packages то, что не нужно движку (TRIM_*); вернуть, что и сколько ушло."""
    removed: list[str] = []
    freed = 0
    paths = [site / name for name in TRIM_FILES]
    for pattern in TRIM_GLOBS:
        paths += sorted(site.glob(pattern))
    for path in paths:
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
            removed.append(path.relative_to(site).as_posix())
    for name in TRIM_DIRS:
        folder = site / name
        if folder.is_dir():
            freed += sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            shutil.rmtree(folder)
            removed.append(name + "/")
    return {"removed": removed, "freed_bytes": freed}


NOTICES = "THIRD_PARTY_NOTICES.txt"
# Части комплекта не из pip: их условия перечисляются вручную (8.7).
FIXED_NOTICES = (
    ("Python (встроенный, python.org)", "PSF-2.0", "runtime/python/LICENSE.txt"),
    (
        "Microsoft Visual C++ Runtime (DLL рядом с python.exe, из VC\\Redist Visual Studio)",
        "лицензия Visual Studio, Distributable Code — условия в licenses/THIRD_PARTY_TERMS.txt",
        "https://visualstudio.microsoft.com/license-terms/",
    ),
    (
        "NVIDIA CUDA 13.2 и cuDNN 9 (torch/lib, triton/backends/nvidia)",
        f"{redist.CUDA_EULA}; {redist.CUDNN_SLA} — условия в licenses/THIRD_PARTY_TERMS.txt",
        redist.CUDA_EULA_URL,
    ),
    (
        ".NET, Avalonia, SkiaSharp, HarfBuzzSharp (оболочка RAMVI.exe)",
        "MIT",
        "https://avaloniaui.net/",
    ),
    (
        "Токенизатор Qwen3-8B (runtime/tokenizers)",
        "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen3-8B",
    ),
)


def _license(dist: md.Distribution) -> str:
    meta = dist.metadata
    text = meta.get("License-Expression") or (meta.get("License") or "").strip().splitlines()[:1]
    if isinstance(text, list):
        text = text[0] if text else ""
    if not text or len(text) > 120:
        classes = [
            c.split("::")[-1].strip()
            for c in meta.get_all("Classifier") or []
            if c.startswith("License ::")
        ]
        text = ", ".join(classes) or (text[:120] if text else "см. файлы лицензии пакета")
    return text


def third_party_notices(dists: dict[str, md.Distribution]) -> str:
    """8.7: сводка сторонних компонентов комплекта — название, версия, лицензия, где текст."""
    lines = [
        "RAMVI — сторонние компоненты и их лицензии",
        "Third-party components of the RAMVI portable bundle and their licenses",
        "",
        "RAMVI: MIT (файл LICENSE рядом). Тексты лицензий — папка licenses/.",
        "",
        "Не из pip / not from pip:",
    ]
    for name, license_, where in FIXED_NOTICES:
        lines.append(f"- {name}: {license_} — {where}")
    lines += ["", "Пакеты Python (runtime/python/Lib/site-packages) / Python packages:"]
    for key in sorted(dists):
        dist = dists[key]
        files = [
            "/".join(f.parts)
            for f in dist.files or []
            if f.parts
            and f.parts[0] != ".."
            and any(k in f.name.lower() for k in ("licen", "notice", "copying"))
        ]
        where = (
            files[0] if len(files) == 1 else (f"{files[0]} (+{len(files) - 1})" if files else "—")
        )
        lines.append(f"- {dist.metadata['Name']} {dist.version}: {_license(dist)} — {where}")
    return "\n".join(lines) + "\n"


def make_zip(folder: Path, target: Path) -> int:
    """8.7: архив для раздачи (папка RAMVI внутри), deflate; вернуть размер архива."""
    with zipfile.ZipFile(
        target, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True, strict_timestamps=False
    ) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            archive.write(path, Path(folder.name) / path.relative_to(folder))
    return target.stat().st_size


def sizes(out: Path) -> dict[str, int]:
    """Размер частей комплекта, байты: для BUILD.json и отчёта о размере (8.7)."""
    parts = {
        "shell": [p for p in out.iterdir() if p.is_file()],
        "code": [out / "ramvi", out / "modules"],
        "python": [out / "runtime" / "python"],
        "tokenizer": [out / "runtime" / "tokenizers"],
    }
    site = out / "runtime" / "python" / "Lib" / "site-packages"

    def size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0

    result = {name: sum(size(p) for p in paths) for name, paths in parts.items()}
    for name in ("torch", "triton", "llvmlite", "transformers"):
        result["python." + name] = size(site / name)
    return result


def product_version() -> str:
    """Версия продукта из `ramvi_service.__version__` (8.3: с ней сравниваются обновления)."""
    text = (ROOT / "service" / "ramvi_service" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([0-9.]+)"', text)
    return match.group(1) if match else "0.0.0"


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
    trim: bool = False,
    vs_license: str | None = None,
    internal: bool = False,
    nuget: Path | None = None,
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
    vc_runtime: dict[str, Any] | None = None
    if sys.platform == "win32":
        vs = redist.find_visual_studio(vs_license or "")
        folder = redist.vc_redist_dir(Path(vs["path"]))
        files = redist.copy_vc_runtime(folder, out / "runtime" / "python")
        vc_runtime = {"visual_studio": vs, "redist": folder.name, "files": files}
    packages: dict[str, str] = {}
    copied = size = 0
    dists = closure(roots)
    for key, dist in dists.items():
        count, bytes_ = copy_distribution(dist, site)
        packages[key] = dist.version
        copied, size = copied + count, size + bytes_
    trimmed = trim_site(site) if trim else None
    nvidia = redist.check_nvidia(site, internal=internal)
    licenses = redist.write_licenses(out, dists, shell, nuget)
    # 8.7: лицензия RAMVI и сводка сторонних компонентов — в корне комплекта.
    if (source / "LICENSE").is_file():
        shutil.copy2(source / "LICENSE", out / "LICENSE")
    (out / NOTICES).write_text(third_party_notices(dists), encoding="utf-8")
    tokens = out / "runtime" / "tokenizers" / "Qwen3-8B"
    tokens.mkdir(parents=True)
    taken = [name for name in TOKENIZER_FILES if (tokenizer / name).is_file()]
    if "tokenizer.json" not in taken and "vocab.json" not in taken:
        raise SystemExit(f"в {tokenizer} нет файлов токенизатора")
    for name in taken:
        shutil.copy2(tokenizer / name, tokens / name)
    redist.assert_no_models(out)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": product_version(),
        "python_embed": embed.name,
        "source_python": sys.version.split()[0],
        "packages": packages,
        "package_files": copied,
        "package_bytes": size,
        "tokenizer_files": taken,
        "dev_dirs": dev,
        "code_sha256": code_digest(out / "ramvi"),
        "total_bytes": total,
        "sizes": sizes(out),
        "trimmed": trimmed,
        # 8.7: внутренняя сборка (с нераспространяемыми файлами) — только для своих проверок.
        "internal": internal,
        "vc_runtime": vc_runtime,
        "nvidia": nvidia,
        "licenses": licenses,
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
    parser.add_argument(
        "--trim", action="store_true", help="убрать ненужные движку библиотеки torch (8.7)"
    )
    parser.add_argument("--zip", type=Path, help="собрать также ZIP для раздачи (8.7)")
    parser.add_argument(
        "--vs-license",
        choices=sorted(redist.VS_LICENSES),
        help="основание лицензии Visual Studio / Build Tools для распространения VC++ (8.7)",
    )
    parser.add_argument(
        "--internal",
        action="store_true",
        help="внутренняя сборка для своих проверок: оставить нераспространяемые файлы (не для раздачи)",
    )
    parser.add_argument(
        "--nuget",
        type=Path,
        help="папка пакетов NuGet для текстов лицензий оболочки (по умолчанию ~/.nuget/packages)",
    )
    args = parser.parse_args()
    manifest = build(
        args.out,
        args.shell,
        args.python_embed,
        args.tokenizer,
        trim=args.trim,
        vs_license=args.vs_license,
        internal=args.internal,
        nuget=args.nuget,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "packages"}, ensure_ascii=False))
    print(f"пакетов: {len(manifest['packages'])}")
    if args.zip and manifest["internal"]:
        raise SystemExit("внутренняя сборка не упаковывается для раздачи")
    if args.zip:
        print(f"zip: {args.zip} {make_zip(args.out, args.zip)} байт")


if __name__ == "__main__":
    main()
