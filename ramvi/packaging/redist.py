"""8.7: чужие файлы в комплекте RAMVI — откуда берутся и по каким правилам распространяются.

- Microsoft Visual C++: только из официального каталога распространяемых файлов
  установленной Visual Studio / Build Tools (`VC\\Redist\\MSVC\\<версия>\\x64\\Microsoft.VC14x.CRT`
  и `...OpenMP`), не из System32. Файлы копируются без изменений (сверка SHA-256), подпись
  Microsoft проверяется, установка и основание лицензии записываются в BUILD.json.
- NVIDIA: каждый файл NVIDIA в пакетах сверяется со списком распространяемых файлов
  CUDA Toolkit EULA (приложение A, версия 13.2 — как CUDA 13.2.86 в torch) и cuDNN SLA
  (v. 22.02.2022: «runtime files .so and .dll»). Нераспространяемые и ненужные движку файлы
  удаляются. ptxas.exe (Triton) тоже не распространяется: ядра собирает драйвер NVIDIA
  (engine/ramvi_engine/kernels/driver_jit.py; проверено — ответ побайтно тот же, скорость та же).
  Механизм «блокирующих» файлов оставлен для будущих находок (`internal` их пропускает).
- LLM/STT-модели и предсказатель пользователь скачивает сам. Служебный Silero VAD
  входит как фиксированный ресурс с MIT-лицензией; исключение только по пути и SHA-256.
- Тексты лицензий собираются в папку `licenses/`.
"""

import fnmatch
import hashlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

# --- Microsoft Visual C++ -------------------------------------------------------------------

VC_CRT = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
    "concrt140.dll",
)
VC_OPENMP = ("vcomp140.dll",)
VC_RUNTIME = VC_CRT + VC_OPENMP
VC_REDIST_COMPONENT = "Microsoft.VisualStudio.Component.VC.Redist.14.Latest"
# Основание лицензии, которое подтверждает тот, кто собирает (программно его не проверить):
# Community — физические лица и open source; Professional/Enterprise — подписка;
# Build Tools — при наличии лицензии Visual Studio или для сборки open source.
VS_LICENSES = {
    "community": (
        "Microsoft.VisualStudio.Product.Community",
        "Microsoft.VisualStudio.Product.BuildTools",
    ),
    "professional": (
        "Microsoft.VisualStudio.Product.Professional",
        "Microsoft.VisualStudio.Product.BuildTools",
    ),
    "enterprise": (
        "Microsoft.VisualStudio.Product.Enterprise",
        "Microsoft.VisualStudio.Product.BuildTools",
    ),
    "open-source": (
        "Microsoft.VisualStudio.Product.BuildTools",
        "Microsoft.VisualStudio.Product.Community",
    ),
}

Runner = Callable[..., Any]


def find_visual_studio(
    license_basis: str, vswhere: Path | None = None, run: Runner = subprocess.run
) -> dict[str, Any]:
    """Установка Visual Studio / Build Tools с распространяемыми файлами VC++ (vswhere)."""
    if license_basis not in VS_LICENSES:
        raise SystemExit(f"укажите основание лицензии Visual Studio: {', '.join(VS_LICENSES)}")
    exe = vswhere or (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not exe.is_file():
        raise SystemExit(
            "нет Visual Studio / Build Tools (vswhere.exe): библиотеки Visual C++ берутся только "
            f"из их каталога VC\\Redist; установите компонент {VC_REDIST_COMPONENT}"
        )
    args = [
        str(exe),
        "-latest",
        "-products",
        "*",
        "-requires",
        VC_REDIST_COMPONENT,
        "-format",
        "json",
        "-utf8",
    ]
    found = json.loads(run(args, capture_output=True, check=True).stdout or "[]")
    if not found:
        raise SystemExit(f"в Visual Studio / Build Tools нет компонента {VC_REDIST_COMPONENT}")
    vs = found[0]
    if vs.get("isComplete") is False:
        raise SystemExit("установка Visual Studio / Build Tools не завершена (isComplete=false)")
    product = vs.get("productId", "")
    if product not in VS_LICENSES[license_basis]:
        raise SystemExit(
            f"основание лицензии «{license_basis}» не подходит к установленному продукту {product}"
        )
    return {
        "path": vs["installationPath"],
        "product": product,
        "name": vs.get("displayName"),
        "version": vs.get("installationVersion"),
        "license_basis": license_basis,
    }


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in name.split("."))


def vc_redist_dir(vs_path: Path) -> Path:
    """Последняя версия каталога VC\\Redist\\MSVC\\<a.b.c> (папки вроде v143 пропускаются)."""
    root = vs_path / "VC" / "Redist" / "MSVC"
    versions = [p for p in root.glob("*") if p.is_dir() and re.fullmatch(r"\d+(\.\d+)+", p.name)]
    if not versions:
        raise SystemExit(f"нет каталога распространяемых файлов VC++: {root}")
    return max(versions, key=lambda p: _version_key(p.name))


def authenticode(path: Path, run: Runner = subprocess.run) -> tuple[str, str]:
    """Состояние подписи и подписант (Get-AuthenticodeSignature)."""
    script = (
        f"$s = Get-AuthenticodeSignature -LiteralPath '{path}'; "
        "Write-Output $s.Status; Write-Output $s.SignerCertificate.Subject"
    )
    out = run(
        ["powershell", "-NoProfile", "-Command", script], capture_output=True, check=True
    ).stdout
    lines = (out.decode() if isinstance(out, bytes) else out).splitlines() + ["", ""]
    return lines[0].strip(), lines[1].strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_vc_runtime(
    redist: Path, target: Path, signature: Callable[[Path], tuple[str, str]] = authenticode
) -> list[dict[str, str]]:
    """Скопировать библиотеки VC++ из официального каталога без изменений.

    Берутся только папки `x64\\Microsoft.VC14x.CRT` и `x64\\Microsoft.VC14x.OpenMP` — они
    перечислены Microsoft как распространяемые; `debug_nonredist` и прочее не трогается.
    Подпись должна быть действительной подписью Microsoft, копия — побайтно как оригинал."""
    x64 = redist / "x64"
    crt = sorted(x64.glob("Microsoft.VC14*.CRT"))
    omp = sorted(x64.glob("Microsoft.VC14*.OpenMP"))
    if not crt or not omp:
        raise SystemExit(f"в {x64} нет папок Microsoft.VC14x.CRT / Microsoft.VC14x.OpenMP")
    sources = {name: crt[-1] / name for name in VC_CRT} | {
        name: omp[-1] / name for name in VC_OPENMP
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise SystemExit("в каталоге распространяемых файлов VC++ нет: " + ", ".join(missing))
    copied = []
    for name, source in sources.items():
        status, subject = signature(source)
        if status != "Valid" or "O=Microsoft Corporation" not in subject:
            raise SystemExit(f"{source}: подпись Microsoft недействительна ({status}, {subject})")
        shutil.copy2(source, target / name)
        digest = _sha256(source)
        if _sha256(target / name) != digest:
            raise SystemExit(f"{name}: копия отличается от оригинала")
        copied.append(
            {"file": name, "from": source.relative_to(redist).as_posix(), "sha256": digest}
        )
    return copied


# --- NVIDIA ----------------------------------------------------------------------------------

CUDA_EULA = "NVIDIA CUDA Toolkit EULA v13.2 (Last updated: January 26, 2026), Attachment A"
CUDA_EULA_URL = "https://docs.nvidia.com/cuda/archive/13.2.0/eula/index.html"
CUDNN_SLA = "NVIDIA SDK License Agreement + cuDNN Supplement (v. February 22, 2022)"
CUDNN_SLA_URL = "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html"
# Имена Windows с суффиксами версии (приложение A: «version-specific variants» тоже разрешены).
NVIDIA_ALLOWED = {
    "cudart64_*.dll": CUDA_EULA,
    "cublas64_*.dll": CUDA_EULA,
    "cublaslt64_*.dll": CUDA_EULA,
    "cufft64_*.dll": CUDA_EULA,
    "cufftw64_*.dll": CUDA_EULA,
    "cusparse64_*.dll": CUDA_EULA,
    "cusolver64_*.dll": CUDA_EULA,
    "curand64_*.dll": CUDA_EULA,
    "nvrtc64_*.dll": CUDA_EULA,
    "nvrtc-builtins64_*.dll": CUDA_EULA,
    "nvjitlink_*.dll": CUDA_EULA,
    "cupti64_*.dll": CUDA_EULA,
    "nvtoolsext64_*.dll": CUDA_EULA,
    "nvvm64_*.dll": CUDA_EULA,
    "libdevice.10.bc": CUDA_EULA,
    "cudnn64_*.dll": CUDNN_SLA,
    "cudnn_*64_*.dll": CUDNN_SLA,
}
# Нет в приложении A — удаляются: cusolverMg/nvperf движку не нужны (в import torch статически
# не входят); вместо ptxas ядра Triton собирает драйвер (driver_jit).
NVIDIA_REMOVE = ("cusolvermg64_*.dll", "nvperf_host*.dll", "nvperf_target*.dll", "ptxas*.exe")
# Нет в приложении A, но нужны — релиз с ними не собирается (сейчас таких нет).
NVIDIA_BLOCKING: tuple[str, ...] = ()
_NVIDIA_NAME = re.compile(r"^(cu|nv|ptxas|libdevice)", re.IGNORECASE)


def _match(name: str, patterns: Any) -> str | None:
    low = name.lower()
    return next((p for p in patterns if fnmatch.fnmatchcase(low, p)), None)


def check_nvidia(site: Path, internal: bool = False) -> dict[str, Any]:
    """Сверить файлы NVIDIA со списками лицензий; удалить нераспространяемые ненужные."""
    allowed: list[dict[str, str]] = []
    removed: list[str] = []
    blocked: list[str] = []
    unknown: list[str] = []
    for path in sorted(site.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".dll", ".exe", ".bc"):
            continue
        if not _NVIDIA_NAME.match(path.name):
            continue
        rel = path.relative_to(site).as_posix()
        if (rule := _match(path.name, NVIDIA_ALLOWED)) is not None:
            allowed.append({"file": rel, "license": NVIDIA_ALLOWED[rule]})
        elif _match(path.name, NVIDIA_REMOVE):
            path.unlink()
            removed.append(rel)
        elif _match(path.name, NVIDIA_BLOCKING):
            blocked.append(rel)
        else:
            unknown.append(rel)
    if unknown:
        raise SystemExit(
            "файлы, похожие на NVIDIA, которых нет в списках лицензий: " + ", ".join(unknown)
        )
    if blocked and not internal:
        raise SystemExit(
            "в пакетах есть файлы NVIDIA, которых нет в приложении A CUDA EULA (распространять нельзя): "
            + ", ".join(blocked)
            + " — релиз не собирается; для своих проверок — внутренняя сборка (--internal)"
        )
    return {"allowed": allowed, "removed": removed, "kept_internal_only": blocked}


# --- модели ------------------------------------------------------------------------------------

MODEL_PATTERNS = ("*.gguf", "*.safetensors", "ggml-*.bin", "*.pt", "*.pth", "*.onnx", "*.ckpt")
MODEL_MIN_BYTES = 2**20  # *.pth — это и служебные файлы путей Python (distutils-precedence.pth)


RUNTIME_MODELS = {
    "ramvi/service/ramvi_service/voice/assets/silero_v5.1.2.onnx": "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
}


def assert_no_models(out: Path) -> None:
    """Нет пользовательских весов; только фиксированный служебный VAD по пути и хешу."""
    found = [
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file()
        and _match(p.name, MODEL_PATTERNS)
        and p.stat().st_size >= MODEL_MIN_BYTES
        and not (
            p.relative_to(out).as_posix() in RUNTIME_MODELS
            and hashlib.sha256(p.read_bytes()).hexdigest()
            == RUNTIME_MODELS[p.relative_to(out).as_posix()]
        )
    ]
    if (out / "models").exists():
        found.append("models/")
    if found:
        raise SystemExit("в комплект попали файлы моделей: " + ", ".join(found[:10]))


# --- тексты лицензий -------------------------------------------------------------------------

LICENSE_NAMES = re.compile(r"(licen[cs]e|notice|copying|copyright|authors?)", re.IGNORECASE)


def _license_files(dist: md.Distribution) -> list[Path]:
    files = []
    for item in dist.files or []:
        if item.parts and item.parts[0] != ".." and LICENSE_NAMES.search(item.name):
            path = Path(str(dist.locate_file(item)))
            if path.is_file():
                files.append((item, path))
    return [p for _, p in files]


def _meta_license(dist: md.Distribution) -> str:
    meta = dist.metadata
    classes = " ".join(c for c in meta.get_all("Classifier") or [] if c.startswith("License ::"))
    return " ".join(x for x in (meta.get("License-Expression"), meta.get("License"), classes) if x)


def _apache_text(dists: dict[str, md.Distribution]) -> Path | None:
    """Полный текст Apache License 2.0 из файла лицензии любого пакета комплекта."""
    for dist in dists.values():
        for path in _license_files(dist):
            try:
                head = path.read_text(encoding="utf-8", errors="replace")[:600]
            except OSError:
                continue
            if "Apache License" in head and "Version 2.0" in head and path.stat().st_size > 9000:
                return path
    return None


def nuget_packages(shell: Path) -> list[tuple[str, str]]:
    """Пакеты NuGet оболочки (id, версия) из *.deps.json публикации."""
    deps = next(shell.glob("*.deps.json"), None)
    if deps is None:
        return []
    libraries = json.loads(deps.read_text(encoding="utf-8")).get("libraries", {})
    result = []
    for key, info in libraries.items():
        if info.get("type") in ("package", "runtimepack"):
            name, version = key.rsplit("/", 1)
            result.append((name.removeprefix("runtimepack."), version))
    return sorted(result)


def _nuspec_license(folder: Path) -> str:
    spec = next(folder.glob("*.nuspec"), None)
    if spec is None:
        return ""
    root = ET.parse(spec).getroot()
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "license" and (element.text or "").strip():
            return (element.text or "").strip()
        if tag == "licenseUrl" and (element.text or "").strip():
            return (element.text or "").strip()
    return ""


END_USER_TERMS = """\
RAMVI — условия для сторонних компонентов / terms for third-party components

Лицензия MIT (файл LICENSE) относится только к коду RAMVI. Сторонние компоненты комплекта
распространяются на своих условиях; их тексты — в этой папке.
The MIT license (LICENSE) covers RAMVI code only. Third-party components keep their own terms.

NVIDIA CUDA и cuDNN (runtime/python/Lib/site-packages/torch/lib, triton/backends/nvidia)
  Условия: {cuda} — {cuda_url}
           {cudnn} — {cudnn_url}
  Эти файлы принадлежат NVIDIA Corporation и её лицензиарам, поставляются в объектном коде,
  без изменений, только как часть RAMVI и только для работы RAMVI в системах с GPU NVIDIA.
  Запрещено: отдельно распространять, изменять, декомпилировать и разбирать их, обходить
  защиту, удалять уведомления об авторских правах.
  These files are owned by NVIDIA Corporation and its licensors, provided in object code,
  unmodified, only as part of RAMVI and only for use by RAMVI on systems with NVIDIA GPUs.
  You may not redistribute them separately, modify, decompile or reverse engineer them,
  circumvent their security, or remove copyright notices.

Microsoft Visual C++ Runtime (runtime/python/*.dll из каталога VC\\Redist Visual Studio)
  Условия: лицензия Visual Studio, раздел «Distributable Code» —
           https://visualstudio.microsoft.com/license-terms/
  Файлы принадлежат Microsoft Corporation, поставляются без изменений и только как часть
  RAMVI. Запрещено: отдельно распространять, изменять, декомпилировать и разбирать их.
  Owned by Microsoft Corporation; provided unmodified and only as part of RAMVI. You may not
  redistribute them separately, modify, decompile or reverse engineer them.
"""


def write_licenses(
    out: Path,
    dists: dict[str, md.Distribution],
    shell: Path,
    nuget_root: Path | None = None,
) -> dict[str, Any]:
    """Папка licenses/: условия для сторонних компонентов и тексты лицензий всех частей."""
    folder = out / "licenses"
    folder.mkdir(exist_ok=True)
    (folder / "THIRD_PARTY_TERMS.txt").write_text(
        END_USER_TERMS.format(
            cuda=CUDA_EULA, cuda_url=CUDA_EULA_URL, cudnn=CUDNN_SLA, cudnn_url=CUDNN_SLA_URL
        ),
        encoding="utf-8",
    )
    python_license = out / "runtime" / "python" / "LICENSE.txt"
    if python_license.is_file():
        (folder / "python").mkdir(exist_ok=True)
        shutil.copy2(python_license, folder / "python" / "LICENSE.txt")
    missing: list[str] = []
    apache = _apache_text(dists)
    for key in sorted(dists):
        dist = dists[key]
        files = _license_files(dist)
        if not files and apache is not None and "apache" in _meta_license(dist).lower():
            # Пакет без файла лицензии (tokenizers): тот же текст Apache-2.0 из другого пакета.
            target = folder / "python-packages" / f"{dist.metadata['Name']}-{dist.version}"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(apache, target / "LICENSE-Apache-2.0.txt")
            continue
        if not files:
            missing.append(f"{dist.metadata['Name']} {dist.version}")
            continue
        target = folder / "python-packages" / f"{dist.metadata['Name']}-{dist.version}"
        for path in files:
            name = (
                path.name
                if not (target / path.name).exists()
                else f"{path.parent.name}-{path.name}"
            )
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / name)
    root = nuget_root or Path(os.environ.get("NUGET_PACKAGES", Path.home() / ".nuget" / "packages"))
    dotnet: list[dict[str, str]] = []
    for name, version in nuget_packages(shell):
        package = root / name.lower() / version.lower()
        texts = (
            [p for p in package.glob("*") if p.is_file() and LICENSE_NAMES.search(p.name)]
            if package.is_dir()
            else []
        )
        if texts:
            target = folder / "dotnet" / f"{name}-{version}"
            target.mkdir(parents=True, exist_ok=True)
            for path in texts:
                shutil.copy2(path, target / path.name)
        license_ = _nuspec_license(package) if package.is_dir() else ""
        dotnet.append(
            {"package": f"{name} {version}", "license": license_, "texts": str(len(texts))}
        )
        if not texts and not license_:
            missing.append(f"{name} {version} (NuGet)")
    return {"python_without_texts": missing, "dotnet": dotnet}
