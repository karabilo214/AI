"""B.12 PORT-MAC-01: комплект RAMVI для macOS (Apple Silicon) — RAMVI.app.

Локальный движок RAMVI работает только на NVIDIA (CUDA/Triton), поэтому комплект для Mac —
облачные модели по API (7.7–7.11), голосовой ввод на CPU, агент и персонаж; torch и CUDA в
него не входят. Запускать на Mac обычным Python (нужны сеть для pip и `iconutil`, `sips`, `ditto`):

    python packaging/build_mac.py --out <новая папка> --shell <dotnet publish -r osx-arm64>
        --python cpython-3.12.*-aarch64-apple-darwin-install_only.tar.gz --tokenizer <Qwen3-8B>
        [--zip RAMVI-<версия>-macos-arm64.zip] [--sign "<Developer ID Application: …>"]

Раскладка: оболочка — в `MacOS`, остальное (как в папке комплекта Windows) — в `Resources`:
рядом с исполняемым `RAMVI` папка `ramvi` невозможна, файловая система Mac не различает регистр.
Оболочка находит `Resources` сама (`ServiceOptions.MacBundleResources`).

    RAMVI.app/Contents/
      Info.plist
      MacOS/RAMVI …                      оболочка (self-contained osx-arm64)
      Resources/ramvi.icns
      Resources/ramvi/service|engine|storage|character   код без тестов
      Resources/modules/                 модули (Character)
      Resources/runtime/python/          python-build-standalone + пакеты из PACKAGES
      Resources/runtime/tokenizers/Qwen3-8B/   токенизатор без весов
      Resources/licenses/, LICENSE, THIRD_PARTY_NOTICES.txt, BUILD.json
"""

import argparse
import importlib.metadata as md
import json
import plistlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_portable as portable  # noqa: E402
import redist  # noqa: E402

ROOT = portable.ROOT
# Те же версии, что в комплекте Windows 0.9.0, без движка (torch, triton, numba, xgrammar).
PACKAGES = (
    "transformers==5.22.1",  # токенизатор персонажа (счёт контекста), без torch
    "jsonschema==4.26.0",  # структурированные ответы персонажа
    "jinja2==3.1.6",  # шаблон чата токенизатора (в Windows приходит с torch)
    "numpy==2.5.4",
    "sounddevice==0.5.6",  # микрофон (PortAudio в колесе)
    "pywhispercpp==1.5.1",  # распознавание речи на CPU
    "onnxruntime==1.24.4",  # паузы в речи (VAD)
)
BUNDLE_ID = "app.ramvi.desktop"
MIN_MACOS = "12.0"


def info_plist(version: str) -> dict[str, Any]:
    return {
        "CFBundleName": "RAMVI",
        "CFBundleDisplayName": "RAMVI",
        "CFBundleExecutable": "RAMVI",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleIconFile": "ramvi.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": MIN_MACOS,
        "NSHighResolutionCapable": True,
        # Голосовой ввод: без объяснения macOS не даст доступ к микрофону.
        "NSMicrophoneUsageDescription": "RAMVI слушает микрофон только по кнопке или Ctrl+M, "
        "чтобы распознать голосовой ввод.",
    }


def make_icns(png: Path, target: Path) -> None:
    """Значок из PNG 256×256: наборы 16–256 точек для iconutil."""
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "ramvi.iconset"
        iconset.mkdir()
        for size in (16, 32, 64, 128, 256):
            for scale in (1, 2):
                pixels = size * scale
                if pixels > 256:
                    continue
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(
                    [
                        "sips",
                        "-z",
                        str(pixels),
                        str(pixels),
                        str(png),
                        "--out",
                        str(iconset / name),
                    ],
                    check=True,
                    capture_output=True,
                )
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True)


def install_python(archive: Path, target: Path) -> Path:
    """python-build-standalone (install_only) → runtime/python; возвращает site-packages."""
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="tar")
        shutil.copytree(Path(tmp) / "python", target, symlinks=True)
    sites = list((target / "lib").glob("python3.*/site-packages"))
    if len(sites) != 1:
        raise SystemExit(f"в {archive} не найден site-packages")
    return sites[0]


def install_packages(python: Path, packages: tuple[str, ...]) -> None:
    """Только готовые колёса (без сборки из исходников), без кеша и без чужого site."""
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--only-binary",
            ":all:",
            "--disable-pip-version-check",
            *packages,
        ],
        check=True,
    )


def trim_python(base: Path) -> list[str]:
    """Убрать то, что программе не нужно: тесты стандартной библиотеки, idle, tkinter, pip."""
    removed = []
    lib = next((base / "lib").glob("python3.*"))
    for name in ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3"):
        if (lib / name).exists():
            shutil.rmtree(lib / name)
            removed.append(name)
    for pattern in ("pip", "pip-*.dist-info"):
        for path in (lib / "site-packages").glob(pattern):
            shutil.rmtree(path)
            removed.append(path.name)
    for cache in list(base.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    return removed


def sign(app: Path, identity: str | None) -> dict[str, Any]:
    """Без идентичности — подпись ad hoc (Apple Silicon требует подписи кода); с Developer ID —
    подпись с Hardened Runtime для нотаризации."""
    command = ["codesign", "--force", "--deep", "--sign", identity or "-"]
    if identity:
        command += ["--options", "runtime", "--timestamp"]
    subprocess.run([*command, str(app)], check=True, capture_output=True)
    check = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True
    )
    return {
        "identity": identity or "ad-hoc",
        "verified": check.returncode == 0,
        "detail": check.stderr[-300:],
    }


def build(
    out: Path, shell: Path, archive: Path, tokenizer: Path, identity: str | None = None
) -> dict[str, Any]:
    if out.exists():
        raise SystemExit(f"папка уже существует: {out}")
    if not (shell / "RAMVI").is_file():
        raise SystemExit(f"в {shell} нет RAMVI — нужна публикация оболочки под osx-arm64")
    started = time.time()
    app = out / "RAMVI.app"
    contents = app / "Contents"
    macos = contents / "MacOS"
    shutil.copytree(shell, macos, symlinks=True)
    for pdb in macos.rglob("*.pdb"):
        pdb.unlink()
    res = contents / "Resources"
    res.mkdir()
    make_icns(ROOT / "desktop/src/Ramvi.Desktop/Assets/ramvi-emblem.png", res / "ramvi.icns")
    version = portable.product_version()
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(info_plist(version), handle)
    for part in portable.CODE:
        shutil.copytree(ROOT / part, res / "ramvi" / part, ignore=portable.SKIP)
    shutil.copytree(ROOT / "i18n", res / "ramvi" / "i18n")
    if (ROOT / "modules").is_dir():
        shutil.copytree(ROOT / "modules", res / "modules", ignore=portable.SKIP)
    base = res / "runtime" / "python"
    site = install_python(archive, base)
    install_packages(base / "bin" / "python3", PACKAGES)
    trimmed = trim_python(base)
    dists = {portable._normal(d.metadata["Name"]): d for d in md.distributions(path=[str(site)])}
    # Лицензия Python: у python-build-standalone — в lib/python3.x/LICENSE.txt.
    python_license = next((base / "lib").glob("python3.*/LICENSE.txt"), None)
    if python_license is not None:
        shutil.copy2(python_license, base / "LICENSE.txt")
    licenses = redist.write_licenses(res, dists, shell)
    # Условия NVIDIA/Microsoft относятся к комплекту Windows: их двоичных файлов здесь нет.
    (res / "licenses" / "THIRD_PARTY_TERMS.txt").unlink(missing_ok=True)
    shutil.copy2(ROOT / "LICENSE", res / "LICENSE")
    (res / portable.NOTICES).write_text(portable.third_party_notices(dists), encoding="utf-8")
    tokens = res / "runtime" / "tokenizers" / "Qwen3-8B"
    tokens.mkdir(parents=True)
    taken = [n for n in portable.TOKENIZER_FILES if (tokenizer / n).is_file()]
    if "tokenizer.json" not in taken:
        raise SystemExit(f"в {tokenizer} нет файлов токенизатора")
    for name in taken:
        shutil.copy2(tokenizer / name, tokens / name)
    redist.assert_no_models(res)
    heavy = sorted(
        n
        for n in dists
        if n in {"torch", "triton", "triton-windows", "numba", "xgrammar"} or n.startswith("nvidia")
    )
    if heavy:
        raise SystemExit(f"в комплект для Mac попали пакеты движка: {heavy}")
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": version,
        "platform": "macos-arm64",
        "python": archive.name,
        "packages": {k: d.version for k, d in sorted(dists.items())},
        "trimmed": trimmed,
        "tokenizer_files": taken,
        "code_sha256": portable.code_digest(res / "ramvi"),
        "licenses": licenses,
    }
    (res / "BUILD.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["signature"] = sign(app, identity)
    manifest["total_bytes"] = sum(
        p.stat().st_size for p in app.rglob("*") if p.is_file() and not p.is_symlink()
    )
    manifest["seconds"] = round(time.time() - started, 1)
    return manifest


def make_zip(out: Path, target: Path) -> int:
    """ditto сохраняет ссылки, права и подпись пакета (обычный zip их теряет)."""
    subprocess.run(
        ["ditto", "-c", "-k", "--keepParent", str(out / "RAMVI.app"), str(target)], check=True
    )
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", required=True, type=Path, help="новая папка, в ней будет RAMVI.app"
    )
    parser.add_argument(
        "--shell", required=True, type=Path, help="dotnet publish -r osx-arm64 --self-contained"
    )
    parser.add_argument(
        "--python", required=True, type=Path, help="архив python-build-standalone install_only"
    )
    parser.add_argument("--tokenizer", required=True, type=Path, help="папка токенизатора Qwen3-8B")
    parser.add_argument("--zip", type=Path, help="собрать ZIP для раздачи")
    parser.add_argument(
        "--sign", help="идентичность Developer ID Application (по умолчанию — ad hoc)"
    )
    args = parser.parse_args()
    manifest = build(args.out, args.shell, args.python, args.tokenizer, args.sign)
    if args.zip:
        manifest["zip_bytes"] = make_zip(args.out, args.zip)
    print(json.dumps({k: v for k, v in manifest.items() if k != "packages"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
