"""Windows drive/UNC не должны попадать в Ollama resolver."""

from pathlib import Path

import pytest
from ramvi_engine.model_location import is_gguf_model, is_ollama_name, resolve_gguf_name


@pytest.mark.parametrize(
    "name",
    [
        r"C:\models\snapshot",
        "C:/models/snapshot",
        r"C:relative",
        r"\\host\share\snapshot",
        "/tmp/checkpoint:revision",
    ],
)
def test_drive_and_absolute_paths_are_not_ollama(name: str) -> None:
    assert not is_ollama_name(name)
    called: list[str] = []

    def resolve(value: str) -> str:
        called.append(value)
        return "wrong"

    assert resolve_gguf_name(name, resolve) == name
    assert not called


def test_existing_hf_directory_wins_over_colon_or_suffix(tmp_path: Path) -> None:
    for basename in ["snapshot", "model.GGUF"]:
        directory = tmp_path / basename
        directory.mkdir()
        assert not is_gguf_model(str(directory))
        assert not is_ollama_name(str(directory))
    if ":" not in str(tmp_path.drive):  # POSIX permits colon in a local directory name.
        directory = tmp_path / "qwen3:local"
        directory.mkdir()
        assert not is_gguf_model(str(directory))
        assert not is_ollama_name(str(directory))


def test_magic_detects_local_blob_and_suffix_is_case_insensitive(tmp_path: Path) -> None:
    blob = tmp_path / "sha256-weights"
    blob.write_bytes(b"GGUF" + b"\0" * 12)
    assert is_gguf_model(str(blob))
    assert resolve_gguf_name(str(blob), lambda _: "wrong") == str(blob)
    other = tmp_path / "weights.safetensors"
    other.write_bytes(b"notGGUF")
    assert not is_gguf_model(str(other))
    assert is_gguf_model(str(tmp_path / "model.GGUF"))
    assert is_gguf_model(r"C:\models\model.GgUf")


def test_real_ollama_tag_is_resolved_once() -> None:
    called: list[str] = []

    def resolve(value: str) -> str:
        called.append(value)
        return "/local/blob"

    name = "qwen3:14b"
    assert is_ollama_name(name) and is_gguf_model(name)
    assert resolve_gguf_name(name, resolve) == "/local/blob"
    assert called == [name]
    assert is_ollama_name("organization/model:tag")
    assert not is_ollama_name("Qwen/Qwen3-0.6B")
    for invalid in ["", "bad\0name"]:
        with pytest.raises(ValueError):
            is_gguf_model(invalid)
