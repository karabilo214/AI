"""B.12: комплект для Mac — состав пакетов и описание программы для macOS."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_mac", ROOT / "packaging" / "build_mac.py")
assert spec is not None and spec.loader is not None
build_mac = importlib.util.module_from_spec(spec)
sys.modules["build_mac"] = build_mac
spec.loader.exec_module(build_mac)


def test_packages_are_the_cloud_set_without_the_engine() -> None:
    names = {p.split("==")[0] for p in build_mac.PACKAGES}
    assert not names & {"torch", "triton", "triton-windows", "numba", "xgrammar", "safetensors"}
    # jinja2 нужен шаблону чата токенизатора персонажа (в Windows приходит вместе с torch).
    assert {
        "transformers",
        "jsonschema",
        "jinja2",
        "sounddevice",
        "pywhispercpp",
        "onnxruntime",
    } <= names
    assert all("==" in p for p in build_mac.PACKAGES)  # версии закреплены, как в комплекте Windows


def test_info_plist_asks_for_the_microphone_and_carries_the_version() -> None:
    plist = build_mac.info_plist("0.9.0")
    assert plist["CFBundleExecutable"] == "RAMVI" and plist["CFBundleShortVersionString"] == "0.9.0"
    assert plist["NSMicrophoneUsageDescription"]  # без него macOS не даст микрофон
    assert plist["LSMinimumSystemVersion"] == build_mac.MIN_MACOS
