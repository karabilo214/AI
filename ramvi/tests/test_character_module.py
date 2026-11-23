"""Character is optional; its real worker owns only its text/presentation state."""

import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from ramvi_character.module import CharacterModule
from ramvi_character.pack import DEFAULT_PACK, load_pack, presentation
from ramvi_service.contracts.modules import Manifest
from ramvi_service.modules.host import ModuleHost
from ramvi_service.sections import SectionRegistry

MODULES = Path(__file__).resolve().parents[1] / "modules"


def test_lazy_text_activation_has_no_device_provider_or_renderer_dependency(tmp_path: Path) -> None:
    with_host = ModuleHost(tmp_path)
    try:
        registry = SectionRegistry(with_host)
        registry.discover(MODULES)
        assert not registry.problems
        assert with_host.status("character")["character"]["worker"] is None
        assert registry.open("character")["state"] == "active"
        status = with_host.invoke("character", "status")
        assert status["mode"] == "text"
        assert not status["devices_open"] and not status["provider_connected"]
        assert status["presentation"] is None
        assert with_host.status("agent")["agent"]["worker"] is None
    finally:
        with_host.close()


def test_checkpoint_restart_and_shutdown_leave_other_worker_untouched(tmp_path: Path) -> None:
    host = ModuleHost(tmp_path)
    try:
        registry = SectionRegistry(host)
        registry.discover(MODULES)
        # No Engine or audio adapters are available to either module.
        host.register(Manifest("other", "1.0.0", "ramvi.tests.module_fixture:create"))
        host.activate("other")
        other = host.status("other")["other"]["worker"]["pid"]
        registry.open("character")
        host.invoke("character", "configure", {"draft": "Незавершённая мысль", "model": "model-a"})
        result = host.invoke("character", "present", {"emotion": "cheerful", "animation": "nod"})
        assert result["presentation"]["speed"] == 2
        host.suspend("character")
        assert host.status("character")["character"]["worker"] is None
        host.resume("character")
        result = host.invoke("character", "status")
        assert result["draft"] == "Незавершённая мысль" and result["model"] == "model-a"
        assert result["emotion"] == "cheerful" and result["presentation"] is None
        host.deactivate("character")
        host.activate("character")
        assert host.invoke("character", "status")["draft"] == "Незавершённая мысль"
        host.dispose("character")
        assert host.status("other")["other"]["worker"]["pid"] == other
        assert host.invoke("other", "value")["value"] == 1
    finally:
        host.close()
    # A new Host (not only suspend/resume) reads the saved namespace.
    host = ModuleHost(tmp_path)
    try:
        registry = SectionRegistry(host)
        registry.discover(MODULES)
        registry.open("character")
        result = host.invoke("character", "status")
        assert (result["draft"], result["model"], result["emotion"]) == (
            "Незавершённая мысль",
            "model-a",
            "cheerful",
        )
        assert result["presentation"] is None
    finally:
        host.close()


def test_pack_is_self_contained_and_has_no_baked_idle_blink_or_speech_sfx() -> None:
    pack = load_pack()
    # 14 лиц: + «в ярости» (красные глаза, сведённые брови) для злости.
    assert len(pack["expressions"]) == 14 and len(pack["sounds"]) == 62
    assert pack["expressions"]["furious"]["eyes"] == "furious"
    # Живые жесты из частей набора — рисует оболочка, кадров нет.
    for gesture in ("jump", "turn_away", "shout", "laugh"):
        assert (
            pack["animations"][gesture]["kind"] == "gesture"
            and "frames" not in pack["animations"][gesture]
        )
    idle = pack["animations"]["idle"]
    assert idle["kind"] == "expression_bob" and "frames" not in idle
    assert pack["speech"] == "tts" and pack["auto_blink"] is False
    pngs = list(DEFAULT_PACK.parent.rglob("*.png"))
    assert len(pngs) == 74  # + глаза и лицо «в ярости»
    assert not (DEFAULT_PACK.parent / "animations" / "idle").exists()
    for path in pngs:
        header = path.read_bytes()[:26]
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", header[16:24]) == (768, 768)
        assert header[25] == 6  # RGBA, no browser/image library needed in CI.
    for name in pack["animations"]:
        output = presentation(pack, "neutral", name)
        assert output["speed"] == (2 if name in {"nod", "shake"} else 1)
        assert output["sound"] is None


def test_invalid_directives_and_state_do_not_mutate_character() -> None:
    before = set(sys.modules)
    module = CharacterModule()
    module.initialize({})
    module.activate()
    module.configure("draft", "model-a")
    module.present("sad", "talk")
    original = module.status()
    for emotion, motion in [("unknown", "talk"), ("sad", "../anything"), ("sad", "exec")]:
        with pytest.raises(ValueError):
            module.present(emotion, motion)
        assert module.status() == original
    # Длина черновика — по общему лимиту текста (оболочка) и бюджету токенов, а не байтам (B.15).
    for draft in [3, None]:
        with pytest.raises(ValueError):
            module.configure(draft)  # type: ignore[arg-type]
        assert module.status() == original
    module.stop_presentation()
    assert module.status()["presentation"] is None
    module.suspend()
    with pytest.raises(RuntimeError):
        module.present("neutral")
    assert not {"torch", "triton", "sounddevice", "pywhispercpp", "human_brain"} & (
        set(sys.modules) - before
    )


def test_pack_rejects_paths_outside_its_directory(tmp_path: Path) -> None:
    pack: dict[str, Any] = load_pack()
    pack["body"] = "../outside.png"
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError, match="resource"):
        load_pack(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "2.0.0"),
        ("version", "bad"),
        ("auto_blink", True),
        ("speech", "sound"),
    ],
)
def test_pack_rejects_incompatible_policy(tmp_path: Path, field: str, value: Any) -> None:
    pack = load_pack()
    pack[field] = value
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pack(path)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, 60001])
def test_pack_rejects_invalid_clock(tmp_path: Path, duration: float) -> None:
    pack = load_pack()
    pack["animations"]["idle"]["duration_ms"] = duration
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError, match="duration"):
        load_pack(path)


@pytest.mark.parametrize("minimum", [-1, 0, 1.1, float("nan"), True])
def test_pack_rejects_invalid_effect(tmp_path: Path, minimum: float) -> None:
    pack = load_pack()
    pack["effects"]["pulse"]["minimum"] = minimum
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError, match="effect"):
        load_pack(path)
