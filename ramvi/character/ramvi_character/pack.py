"""Versioned presentation resources, independent of any renderer or LLM."""

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_PACK = Path(__file__).parent / "assets" / "pack.json"


def load_pack(path: Path = DEFAULT_PACK) -> dict[str, Any]:
    try:
        return _load_pack(path)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("invalid character pack structure") from error


def _load_pack(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ramvi.character.pack/1":
        raise ValueError("unsupported character pack schema")
    if not isinstance(value.get("version"), str) or not re.fullmatch(
        r"1\.\d+\.\d+", value["version"]
    ):
        raise ValueError("unsupported character pack version")
    if value.get("speech") != "tts" or value.get("auto_blink") is not False:
        raise ValueError("unsupported speech or blinking policy")
    if value.get("canvas") != [768, 768]:
        raise ValueError("unsupported character canvas")
    images = [value["body"], *value["eyes"].values(), *value["mouths"].values()]
    images += [x["file"] for x in value["expressions"].values()]
    images += list(value["turns"].values())
    if "neutral" not in value["expressions"] or "idle" not in value["animations"]:
        raise ValueError("pack must provide neutral and idle")
    if "front" not in value["turns"]:
        raise ValueError("pack must provide a front turn")
    if set(value["animations"]) != {"idle", "blink", "talk", "nod", "shake"}:
        raise ValueError("unsupported animation states")
    if (
        not {"closed", "half"} <= value["eyes"].keys()
        or not {"neutral", "talk_a", "talk_e", "open", "o"} <= value["mouths"].keys()
    ):
        raise ValueError("missing animated facial layers")
    for item in value["expressions"].values():
        if item["eyes"] not in value["eyes"] or item["mouth"] not in value["mouths"]:
            raise ValueError("expression references an unknown facial layer")
    for name, item in value["animations"].items():
        if type(item["speed"]) is not int or item["speed"] != (
            2 if name in {"nod", "shake"} else 1
        ):
            raise ValueError("animation speed does not match approved policy")
        duration = item["duration_ms"]
        if (
            type(duration) not in {int, float}
            or not math.isfinite(duration)
            or not 0 < duration <= 60_000
        ):
            raise ValueError("invalid frame duration")
        if item["kind"] == "expression_bob":
            if (
                name != "idle"
                or type(item["frame_count"]) is not int
                or not 1 <= item["frame_count"] <= 256
                or "frames" in item
            ):
                raise ValueError("idle must not contain baked blinking frames")
        elif name != "idle" and item["kind"] == "frames" and 2 <= len(item["frames"]) <= 256:
            if name in {"blink", "talk"} and len(item["frames"]) != {"blink": 6, "talk": 8}[name]:
                raise ValueError("invalid facial frame count")
            images += item["frames"]
        else:
            raise ValueError("unsupported animation renderer")
        if type(item["loop"]) is not bool or item["loop"] != (name in {"idle", "talk"}):
            raise ValueError("invalid gesture repeat policy")
    for name, effect in value.get("effects", {}).items():
        if name == "none" or effect["kind"] != "opacity":
            raise ValueError("unsupported effect")
        for field, lower, upper in [("minimum", 0.25, 1), ("duration_ms", 200, 60_000)]:
            number = effect[field]
            if (
                type(number) not in {int, float}
                or not math.isfinite(number)
                or not lower <= number <= upper
            ):
                raise ValueError("invalid effect parameter")
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
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
            raise ValueError("invalid resource path")
        unresolved = path.parent / relative
        current = unresolved
        while current != path.parent and current != current.parent:
            if current.is_symlink():
                raise ValueError("linked resource is not supported")
            current = current.parent
        target = unresolved.resolve()
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
