"""Validate source packs and build the offline preview catalog. Run with Pillow."""

import json
import wave
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
BRANDING = HERE.parent


def build():
    character = BRANDING / "RAMVI_Character_Pack"
    sound = BRANDING / "RAMVI_Character_Sound_Pack"
    pack = json.loads((character / "character.json").read_text())
    sounds = json.loads((sound / "manifest.json").read_text())
    images = set(pack["eye_parts"] + pack["mouth_parts"] + [pack["character_base"]])
    for item in pack["expressions"].values():
        images.add(item["file"])
    turns = pack["turns"]
    for item in turns.values():
        images.add(item["file"] if isinstance(item, dict) else item)
    for name, item in pack["animations"].items():
        assert item["frames"] and item["duration_ms"] > 0, name
        images.update(item["frames"])
    for relative in sorted(images):
        path = character / relative
        assert path.resolve().is_relative_to(character.resolve()), relative
        with Image.open(path) as im:
            assert im.size == tuple(pack["canvas"]), (relative, im.size)
            assert im.mode == "RGBA", (relative, im.mode)
    ids = set()
    for item in sounds["files"]:
        assert item["id"] not in ids, item["id"]
        ids.add(item["id"])
        path = sound / item["file"]
        assert path.resolve().is_relative_to(sound.resolve()), item["file"]
        with wave.open(str(path)) as wav:
            assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 48000, 2), (
                item["file"]
            )
            assert abs(wav.getnframes() / wav.getframerate() - item["duration_s"]) < 0.011, item[
                "file"
            ]
    assert len(ids) == sounds["count"]
    data = {"character": pack, "sounds": sounds, "checked_images": len(images)}
    (HERE / "catalog.js").write_text(
        "window.RAMVI_PACK = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"
    )
    print(
        json.dumps(
            {
                "images": len(images),
                "animations": len(pack["animations"]),
                "expressions": len(pack["expressions"]),
                "sounds": len(ids),
            }
        )
    )


if __name__ == "__main__":
    build()
