"""3.9: сборщик переносимой папки на подменных входных данных (настоящая сборка — на Windows)."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_portable", ROOT / "packaging" / "build_portable.py"
)
assert spec is not None and spec.loader is not None
bp = importlib.util.module_from_spec(spec)
sys.modules["build_portable"] = bp
spec.loader.exec_module(bp)
redist_mod = bp.redist  # packaging/redist.py, загружен сборщиком

needs_jsonschema = pytest.mark.skipif(
    bp.find_distribution("jsonschema") is None,
    reason="в лёгком окружении нет jsonschema для замыкания",
)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    shell = tmp_path / "pub"
    shell.mkdir()
    (shell / "RAMVI.exe").write_bytes(b"MZ")
    (shell / "Avalonia.Base.dll").write_bytes(b"dll")
    embed = tmp_path / "python-3.12.10-embed-amd64.zip"
    with zipfile.ZipFile(embed, "w") as z:
        z.writestr("python.exe", b"MZ")
        z.writestr("python312.zip", b"PK")
        z.writestr(
            "python312._pth",
            "python312.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n",
        )
    tok = tmp_path / "Qwen3-8B"
    tok.mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ):
        (tok / name).write_text("{}")
    (tok / "model-00001-of-00005.safetensors").write_bytes(b"x" * 1000)  # веса не должны попасть
    return shell, embed, tok


@needs_jsonschema
def test_layout_closure_and_manifest(tmp_path: Path, inputs: tuple[Path, Path, Path]) -> None:
    shell, embed, tok = inputs
    out = tmp_path / "RAMVI"

    manifest = bp.build(out, shell, embed, tok, roots=("jsonschema",))

    assert (out / "RAMVI.exe").is_file() and (out / "Avalonia.Base.dll").is_file()
    for part in bp.CODE:
        assert (out / "ramvi" / part / "__init__.py").is_file()
    assert (out / "ramvi" / "service" / "ramvi_service" / "app.py").is_file()
    assert (out / "modules/agent/manifest.json").is_file()
    assert (out / "modules/character/manifest.json").is_file()
    assert (out / "ramvi/character/ramvi_character/assets/pack.json").is_file()
    pth_text = next((out / "runtime/python").glob("python*._pth")).read_text()
    assert "ramvi\\character" in pth_text
    assert (out / "ramvi/service/ramvi_service/system_context.py").is_file()
    assert (out / "ramvi/service/ramvi_service/native/windows_context.ps1").is_file()
    assert not list((out / "ramvi").rglob("__pycache__")) and not list(
        (out / "ramvi").rglob("tests")
    )
    pth = (out / "runtime" / "python" / "python312._pth").read_text().splitlines()
    assert pth == [
        "python312.zip",
        ".",
        "# Uncomment to run site.main() automatically",
        "Lib\\site-packages",
        "..\\..\\ramvi\\service",
        "..\\..\\ramvi\\engine",
        "..\\..\\ramvi\\storage",
        "..\\..\\ramvi\\character",
        "import site",
    ]
    for line in pth[4:8]:  # пути из ._pth указывают на скопированный код
        assert (out / "runtime" / "python" / Path(*line.split("\\"))).resolve().is_dir()
    site = out / "runtime" / "python" / "Lib" / "site-packages"
    # Замыкание jsonschema: его зависимости тоже скопированы, с dist-info.
    names = set(manifest["packages"])
    assert {"jsonschema", "attrs", "referencing", "jsonschema-specifications", "rpds-py"} <= names
    assert (site / "jsonschema" / "__init__.py").is_file()
    assert list(site.glob("jsonschema-*.dist-info/METADATA"))
    tokens = out / "runtime" / "tokenizers" / "Qwen3-8B"
    assert sorted(p.name for p in tokens.iterdir()) == sorted(manifest["tokenizer_files"])
    assert not list(tokens.glob("*.safetensors"))
    saved = json.loads((out / "BUILD.json").read_text(encoding="utf-8"))
    assert saved["packages"]["jsonschema"] == manifest["packages"]["jsonschema"]
    assert saved["total_bytes"] > 0 and len(saved["code_sha256"]) == 64
    # 8.7: лицензия RAMVI, сводка сторонних лицензий, размеры частей; обрезка — только по флагу
    assert (out / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    notices = (out / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    assert "jsonschema" in notices and "MIT" in notices and "NVIDIA CUDA" in notices
    assert saved["sizes"]["code"] > 0 and saved["trimmed"] is None
    assert (out / "licenses" / "THIRD_PARTY_TERMS.txt").is_file() and saved["internal"] is False
    assert saved["nvidia"] == {"allowed": [], "removed": [], "kept_internal_only": []}


@needs_jsonschema
def test_built_folder_imports_with_only_its_site_packages(
    tmp_path: Path, inputs: tuple[Path, Path, Path]
) -> None:
    """Скопированное замыкание самодостаточно: пакет импортируется только из site-packages комплекта."""
    import subprocess

    shell, embed, tok = inputs
    out = tmp_path / "RAMVI"
    bp.build(out, shell, embed, tok, roots=("jsonschema",))
    site = out / "runtime" / "python" / "Lib" / "site-packages"
    code = "import jsonschema, sys; print(jsonschema.__file__)"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import sys; sys.path.insert(0, {str(site)!r}); {code}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith(str(site))


@needs_jsonschema
def test_refuses_existing_folder_and_bad_inputs(
    tmp_path: Path, inputs: tuple[Path, Path, Path]
) -> None:
    shell, embed, tok = inputs
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit, match="уже существует"):
        bp.build(tmp_path / "exists", shell, embed, tok, roots=("jsonschema",))
    with pytest.raises(SystemExit, match="RAMVI.exe"):
        bp.build(tmp_path / "a", tmp_path, embed, tok, roots=("jsonschema",))
    with pytest.raises(SystemExit, match="не установлен"):
        bp.build(tmp_path / "b", shell, embed, tok, roots=("no-such-package-xyz",))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="токенизатора"):
        bp.build(tmp_path / "c", shell, embed, empty, roots=("jsonschema",))


@needs_jsonschema
def test_markers_exclude_other_platforms() -> None:
    """Зависимости с маркером чужой платформы не попадают (например, nvidia-* для Linux у torch)."""
    found = bp.closure(("jsonschema",))
    assert all(dist.metadata["Name"] for dist in found.values())
    requirement = bp._requirement_tools()
    assert not requirement('x; sys_platform == "nonexistent-os"').marker.evaluate({"extra": ""})


def test_dev_files_for_triton_are_copied(tmp_path: Path) -> None:
    """Triton на Windows компилирует C-модуль: нужны Include/ и libs/ базового Python."""
    base = tmp_path / "python312"
    (base / "Include" / "cpython").mkdir(parents=True)
    (base / "Include" / "Python.h").write_text("/* h */")
    (base / "libs").mkdir()
    (base / "libs" / "python312.lib").write_bytes(b"lib")
    target = tmp_path / "runtime" / "python"
    target.mkdir(parents=True)

    assert bp.copy_dev_files(base, target) == ["Include", "libs"]
    assert (target / "Include" / "Python.h").is_file() and (
        target / "libs" / "python312.lib"
    ).is_file()
    assert bp.copy_dev_files(tmp_path / "missing", target) == []


def _vc_redist(root: Path) -> Path:
    redist = root / "VC" / "Redist" / "MSVC" / "14.44.35112"
    crt = redist / "x64" / "Microsoft.VC143.CRT"
    omp = redist / "x64" / "Microsoft.VC143.OpenMP"
    crt.mkdir(parents=True)
    omp.mkdir(parents=True)
    (root / "VC" / "Redist" / "MSVC" / "v143").mkdir()  # не версия — пропускается
    for name in redist_mod.VC_CRT:
        (crt / name).write_bytes(b"dll " + name.encode())
    for name in redist_mod.VC_OPENMP:
        (omp / name).write_bytes(b"dll " + name.encode())
    (redist / "debug_nonredist").mkdir()
    return redist


MS_SIGNED = ("Valid", "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, C=US")


def test_vc_runtime_comes_from_visual_studio_redist_unmodified(tmp_path: Path) -> None:
    """8.7: VC++ — из каталога VC\\Redist, подпись Microsoft, побайтно как оригинал."""
    vs = tmp_path / "VS"
    redist = _vc_redist(vs)
    assert redist_mod.vc_redist_dir(vs) == redist
    target = tmp_path / "python"
    target.mkdir()
    copied = redist_mod.copy_vc_runtime(redist, target, signature=lambda p: MS_SIGNED)
    assert [c["file"] for c in copied] == list(redist_mod.VC_RUNTIME)
    assert copied[0]["from"] == "x64/Microsoft.VC143.CRT/msvcp140.dll"
    assert (target / "vcomp140.dll").read_bytes() == b"dll vcomp140.dll"
    with pytest.raises(SystemExit, match="подпись Microsoft"):
        redist_mod.copy_vc_runtime(redist, target, signature=lambda p: ("NotSigned", ""))
    with pytest.raises(SystemExit, match="подпись Microsoft"):
        redist_mod.copy_vc_runtime(
            redist, target, signature=lambda p: ("Valid", "CN=Someone, O=Other")
        )
    (redist / "x64" / "Microsoft.VC143.CRT" / "msvcp140.dll").unlink()
    with pytest.raises(SystemExit, match="msvcp140.dll"):
        redist_mod.copy_vc_runtime(redist, target, signature=lambda p: MS_SIGNED)
    assert bp.VC_RUNTIME == redist_mod.VC_RUNTIME


def test_visual_studio_and_license_basis_are_required(tmp_path: Path) -> None:
    vswhere = tmp_path / "vswhere.exe"
    with pytest.raises(SystemExit, match="vswhere"):
        redist_mod.find_visual_studio("community", vswhere)
    vswhere.write_bytes(b"")
    found = [
        {
            "installationPath": "C:/BuildTools",
            "productId": "Microsoft.VisualStudio.Product.BuildTools",
            "displayName": "Visual Studio Build Tools 2022",
            "installationVersion": "17.14.1",
            "isComplete": True,
        }
    ]
    calls: list[list[str]] = []

    def run(args: list[str], **kw: Any) -> Any:
        calls.append(args)
        return type("R", (), {"stdout": json.dumps(found)})()

    vs = redist_mod.find_visual_studio("open-source", vswhere, run)
    assert vs["product"].endswith("BuildTools") and vs["license_basis"] == "open-source"
    assert redist_mod.VC_REDIST_COMPONENT in calls[0]
    with pytest.raises(SystemExit, match="основание лицензии"):
        redist_mod.find_visual_studio("", vswhere, run)
    found[0]["productId"] = "Microsoft.VisualStudio.Product.Community"
    with pytest.raises(SystemExit, match="не подходит"):
        redist_mod.find_visual_studio("enterprise", vswhere, run)
    found.clear()
    with pytest.raises(SystemExit, match="нет компонента"):
        redist_mod.find_visual_studio("community", vswhere, run)


def test_nvidia_files_are_checked_against_license_lists(tmp_path: Path) -> None:
    """8.7: CUDA EULA 13.2 прил. A и cuDNN SLA; cusolverMg, nvperf и ptxas удаляются."""
    site = tmp_path / "site-packages"
    names = [
        "torch/lib/cublasLt64_13.dll",
        "torch/lib/cudnn64_9.dll",
        "torch/lib/cudnn_graph64_9.dll",
        "torch/lib/nvJitLink_130_0.dll",
        "torch/lib/nvrtc-builtins64_132.dll",
        "torch/lib/cupti64_2026.1.2.dll",
        "torch/lib/nvToolsExt64_1.dll",
        "torch/lib/cusolverMg64_12.dll",
        "torch/lib/nvperf_host.dll",
        "torch/lib/caffe2_nvrtc.dll",  # PyTorch, не NVIDIA
        "triton/backends/nvidia/lib/libdevice.10.bc",
        "triton/backends/nvidia/bin/ptxas.exe",
    ]
    for name in names:
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_bytes(b"x")
    result = redist_mod.check_nvidia(site)
    assert sorted(result["removed"]) == [
        "torch/lib/cusolverMg64_12.dll",
        "torch/lib/nvperf_host.dll",
        "triton/backends/nvidia/bin/ptxas.exe",  # ядра собирает драйвер (driver_jit)
    ]
    assert not (site / "triton/backends/nvidia/bin/ptxas.exe").exists()
    assert result["kept_internal_only"] == []
    allowed = {a["file"] for a in result["allowed"]}
    assert (
        "torch/lib/cudnn_graph64_9.dll" in allowed
        and "triton/backends/nvidia/lib/libdevice.10.bc" in allowed
    )
    assert "torch/lib/caffe2_nvrtc.dll" not in allowed
    (site / "torch/lib/nvfoo64_1.dll").write_bytes(b"x")
    with pytest.raises(SystemExit, match="nvfoo64_1.dll"):
        redist_mod.check_nvidia(site, internal=True)


def test_release_contains_no_models(tmp_path: Path) -> None:
    out = tmp_path / "RAMVI"
    (out / "runtime" / "tokenizers" / "Qwen3-8B").mkdir(parents=True)
    (out / "runtime" / "tokenizers" / "Qwen3-8B" / "tokenizer.json").write_text("{}")
    site = out / "runtime" / "python" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "distutils-precedence.pth").write_text("import os")  # служебный файл Python
    redist_mod.assert_no_models(out)
    (out / "runtime" / "big.gguf").write_bytes(b"x" * 2**20)
    with pytest.raises(SystemExit, match="big.gguf"):
        redist_mod.assert_no_models(out)
    (out / "runtime" / "big.gguf").unlink()
    (out / "models" / "stt").mkdir(parents=True)
    (out / "models" / "stt" / "ggml-large-v3-turbo-q5_0.bin").write_bytes(b"x")
    with pytest.raises(SystemExit, match="models/"):
        redist_mod.assert_no_models(out)


def test_licenses_folder_has_terms_and_texts(tmp_path: Path) -> None:
    out, shell, nuget = tmp_path / "RAMVI", tmp_path / "pub", tmp_path / "nuget"
    (out / "runtime" / "python").mkdir(parents=True)
    (out / "runtime" / "python" / "LICENSE.txt").write_text("PSF")
    shell.mkdir()
    (shell / "RAMVI.deps.json").write_text(
        json.dumps(
            {
                "libraries": {
                    "Avalonia/12.1.6": {"type": "package"},
                    "runtimepack.Microsoft.NETCore.App.Runtime.win-x64/10.0.1": {
                        "type": "runtimepack"
                    },
                    "RAMVI/1.0.0": {"type": "project"},
                }
            }
        )
    )
    avalonia = nuget / "avalonia" / "12.1.6"
    avalonia.mkdir(parents=True)
    (avalonia / "avalonia.nuspec").write_text(
        '<package><metadata><license type="expression">MIT</license>'
        "<copyright>Copyright 2013-2026 © The AvaloniaUI Project</copyright></metadata></package>"
    )
    runtime = nuget / "microsoft.netcore.app.runtime.win-x64" / "10.0.1"
    runtime.mkdir(parents=True)
    (runtime / "LICENSE.TXT").write_text("MIT .NET")
    (runtime / "THIRD-PARTY-NOTICES.TXT").write_text("notices")
    dists = bp.closure(("jsonschema",)) if bp.find_distribution("jsonschema") else {}
    result = redist_mod.write_licenses(out, dists, shell, nuget)
    terms = (out / "licenses" / "THIRD_PARTY_TERMS.txt").read_text(encoding="utf-8")
    assert "v13.2" in terms and "February 22, 2022" in terms and "Distributable Code" in terms
    assert (out / "licenses" / "python" / "LICENSE.txt").read_text() == "PSF"
    assert (
        out
        / "licenses"
        / "dotnet"
        / "Microsoft.NETCore.App.Runtime.win-x64-10.0.1"
        / "THIRD-PARTY-NOTICES.TXT"
    ).is_file()
    assert {d["package"]: d["license"] for d in result["dotnet"]}["Avalonia 12.1.6"] == "MIT"
    mit = (out / "licenses" / "dotnet" / "Avalonia-12.1.6" / "LICENSE-MIT.txt").read_text(
        encoding="utf-8"
    )
    assert mit.startswith("MIT License\n\nCopyright 2013-2026 © The AvaloniaUI Project\n")
    assert "THE SOFTWARE IS PROVIDED" in mit
    if dists:
        assert list((out / "licenses" / "python-packages").glob("jsonschema-*/COPYING"))


def test_trim_removes_only_listed_torch_parts(tmp_path: Path) -> None:
    """8.7: уходят подбиблиотеки cuDNN, заголовки и .lib torch; нужное движку остаётся."""
    site = tmp_path / "site-packages"
    keep = [
        "torch/lib/cudnn64_9.dll",  # её импортируют torch_cuda и torch_python
        "torch/lib/torch_cuda.dll",
        "torch/lib/cublas64_13.dll",
        "triton/backends/nvidia/lib/x64/cuda.lib",  # Triton собирает свой модуль
        "triton/backends/nvidia/include/cuda.h",
        "torchgen/model.py",  # его импортирует torch/utils/_python_dispatch.py
    ]
    gone = [
        "torch/lib/cudnn_engines_precompiled64_9.dll",
        "torch/lib/cusolverMg64_12.dll",
        "torch/lib/torch_cpu.lib",
        "torch/include/ATen/ATen.h",
    ]
    for name in keep + gone:
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_bytes(b"x" * 10)
    result = bp.trim_site(site)
    assert all((site / name).exists() for name in keep)
    assert not any((site / name).exists() for name in gone)
    assert result["freed_bytes"] == 10 * len(gone)
    assert "torch/include/" in result["removed"] and "torch/lib/torch_cpu.lib" in result["removed"]
    assert bp.trim_site(site)["freed_bytes"] == 0  # повторно — нечего удалять


def test_zip_contains_folder_and_reports_size(tmp_path: Path) -> None:
    folder = tmp_path / "RAMVI"
    (folder / "runtime").mkdir(parents=True)
    (folder / "RAMVI.exe").write_bytes(b"exe" * 1000)
    (folder / "runtime" / "a.txt").write_text("привет", encoding="utf-8")
    size = bp.make_zip(folder, tmp_path / "RAMVI.zip")
    assert size == (tmp_path / "RAMVI.zip").stat().st_size and size < 3000
    with zipfile.ZipFile(tmp_path / "RAMVI.zip") as archive:
        assert sorted(archive.namelist()) == ["RAMVI/RAMVI.exe", "RAMVI/runtime/a.txt"]


def test_record_with_wheel_data_paths_is_copied(tmp_path: Path) -> None:
    """8.7: RECORD pywhispercpp указывает `.data/platlib/x.dll`, файл лежит в корне site."""
    source = tmp_path / "src-site"
    info = source / "pkg-1.0.dist-info"
    info.mkdir(parents=True)
    (source / "pkg").mkdir()
    (source / "pkg" / "__init__.py").write_text("")
    (source / "whisper-abc.dll").write_bytes(b"dll")
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0\n")
    (info / "RECORD").write_text(
        "pkg/__init__.py,,\npkg-1.0.data/platlib/whisper-abc.dll,,\npkg-1.0.dist-info/RECORD,,\n"
    )
    dist = bp.md.PathDistribution(info)
    target = tmp_path / "site"
    count, _ = bp.copy_distribution(dist, target)
    assert (target / "whisper-abc.dll").read_bytes() == b"dll"
    assert (target / "pkg" / "__init__.py").is_file() and count == 3


def test_only_pinned_runtime_vad_is_allowed_not_arbitrary_model(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "service/ramvi_service/voice/assets/silero_v5.1.2.onnx"
    )
    target = tmp_path / "ramvi/service/ramvi_service/voice/assets/silero_v5.1.2.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    redist_mod.assert_no_models(tmp_path)
    target.write_bytes(b"x" * source.stat().st_size)
    with pytest.raises(SystemExit, match="silero_v5.1.2"):
        redist_mod.assert_no_models(tmp_path)
    target.unlink()
    (target.parent / "another.onnx").write_bytes(source.read_bytes())
    with pytest.raises(SystemExit, match="another.onnx"):
        redist_mod.assert_no_models(tmp_path)
