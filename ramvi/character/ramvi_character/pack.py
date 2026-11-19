"""Versioned presentation resources, independent of any renderer or LLM."""

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_PACK = Path(__file__).parent / "assets" / "pack.json"


def load_pack(path: Path = DEFAULT_PACK) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ramvi.character.pack/1":
        raise ValueError("unsupported character pack schema")
    if value.get("canvas") != [768, 768]:
        raise ValueError("unsupported character canvas")
    images = [value["body"], *value["eyes"].values(), *value["mouths"].values()]
    images += [x["file"] for x in value["expressions"].values()]
    images += list(value["turns"].values())
    if "neutral" not in value["expressions"] or "idle" not in value["animations"]:
        raise ValueError("pack must provide neutral and idle")
    for item in value["expressions"].values():
        if item["eyes"] not in value["eyes"] or item["mouth"] not in value["mouths"]:
            raise ValueError("expression references an unknown facial layer")
    for name, item in value["animations"].items():
        if item["speed"] != (2 if name in {"nod", "shake"} else 1):
            raise ValueError("animation speed does not match approved policy")
        if item["duration_ms"] <= 0:
            raise ValueError("invalid frame duration")
        if item["kind"] == "expression_bob":
            if name != "idle" or item["frame_count"] < 1 or "frames" in item:
                raise ValueError("idle must not contain baked blinking frames")
        elif item["kind"] == "frames" and item["frames"]:
            images += item["frames"]
        else:
            raise ValueError("unsupported animation renderer")
        if item["loop"] != (name in {"idle", "talk"}):
            raise ValueError("invalid gesture repeat policy")
    files = images + [x["file"] for x in value["sounds"]]
    ids = set()
    for sound in value["sounds"]:
        identifier = sound["id"]
        if (
            identifier in ids
            or identifier.startswith("character/talk_")
            or identifier in {"motion/shake_head", "character/beep_no"}
        ):
            raise ValueError("duplicate or excluded sound")
        ids.add(identifier)
    for relative in files:
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent.resolve()) or not target.is_file():
            raise ValueError("missing resource or resource outside the pack")
    return copy.deepcopy(value)


def presentation(pack: dict[str, Any], emotion: str, animation: str) -> dict[str, Any]:
    """Only named capabilities, never paths or commands supplied by a model."""
    if emotion not in pack["expressions"] or animation not in pack["animations"]:
        raise ValueError("unsupported emotion or animation")
    motion = pack["animations"][animation]
    return {
        "schema": "ramvi.character.presentation/1",
        "emotion": emotion,
        "animation": animation,
        "speed": motion["speed"],
        "loop": motion["loop"],
        "sound": None,  # Speech belongs to TTS, not an animation side effect.
    }
