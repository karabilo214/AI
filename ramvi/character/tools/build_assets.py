"""Build the self-contained runtime pack from the original branding sources."""

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BRANDING = ROOT / "assets" / "branding"
TARGET = ROOT / "character" / "ramvi_character" / "assets"


def build() -> None:
    visual = BRANDING / "RAMVI_Character_Pack"
    source = json.loads((visual / "character.json").read_text(encoding="utf-8"))
    audio = BRANDING / "RAMVI_Character_Sound_Pack"
    sounds = json.loads((audio / "manifest.json").read_text(encoding="utf-8"))["files"]
    pack: dict[str, Any] = {
        "schema": "ramvi.character.pack/1",
        "version": "1.0.0",
        "canvas": source["canvas"],
        "body": source["character_base"],
        "eyes": {Path(x).stem: x for x in source["eye_parts"]},
        "mouths": {Path(x).stem: x for x in source["mouth_parts"]},
        "expressions": source["expressions"],
        "turns": source["turns"],
        "animations": {},
        "sounds": [],
        "speech": "tts",
        "auto_blink": False,
    }
    images = {pack["body"], *pack["eyes"].values(), *pack["mouths"].values()}
    images.update(x["file"] for x in pack["expressions"].values())
    images.update(pack["turns"].values())
    for name, item in source["animations"].items():
        motion = {
            "kind": "expression_bob" if name == "idle" else "frames",
            "duration_ms": item["duration_ms"],
            "speed": 2 if name in {"nod", "shake"} else 1,
            "loop": name in {"idle", "talk"},
        }
        if name == "idle":
            motion["frame_count"] = len(item["frames"])
        else:
            motion["frames"] = item["frames"]
            images.update(item["frames"])
        pack["animations"][name] = motion
    for relative in sorted(images):
        target = TARGET / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(visual / relative, target)
    for item in sounds:
        if item["id"].startswith("character/talk_") or item["id"] in {
            "motion/shake_head",
            "character/beep_no",
        }:
            continue
        copy = {**item, "file": "sounds/" + item["file"]}
        target = TARGET / copy["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio / item["file"], target)
        pack["sounds"].append(copy)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Character pack: {len(images)} PNG, {len(pack['sounds'])} WAV")


if __name__ == "__main__":
    build()
