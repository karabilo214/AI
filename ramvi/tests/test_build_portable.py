"""3.9: сборщик переносимой папки на подменных входных данных (настоящая сборка — на Windows)."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_portable", ROOT / "packaging" / "build_portable.py"
)
assert spec is not None and spec.loader is not None
bp = importlib.util.module_from_spec(spec)
sys.modules["build_portable"] = bp
spec.loader.exec_module(bp)


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
        "import site",
    ]
    for line in pth[4:7]:  # пути из ._pth указывают на скопированный код
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
