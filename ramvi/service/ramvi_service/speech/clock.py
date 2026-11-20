"""Bounded presentation telemetry in the device clock's time domain; no audio imports."""

import math
from typing import Any


def cue_at(alignment: Any, position: float) -> dict[str, str] | None:
    if not isinstance(alignment, dict):
        return None
    kind = alignment.get("kind")
    if not isinstance(kind, str):
        return None
    key = {"characters": "characters", "phonemes": "phonemes", "visemes": "visemes"}.get(kind)
    if key is None:
        return None
    values, starts, ends = (alignment.get(k) for k in [key, "starts", "ends"])
    if not isinstance(values, list) or not isinstance(starts, list) or not isinstance(ends, list):
        return None
    if not len(values) == len(starts) == len(ends) <= 2048:
        return None
    for value, start, end in zip(values, starts, ends, strict=True):
        if (
            isinstance(value, str)
            and 0 < len(value) <= 16
            and type(start) in {float, int}
            and type(end) in {float, int}
            and math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= position < end <= 120
        ):
            return {"kind": kind, "value": value}
    return None


def playback_view(raw: dict[str, Any] | None, age: float, alignment: Any = None) -> dict[str, Any]:
    """Only a short look-ahead is published. Missing/expired telemetry means silence."""
    result: dict[str, Any] = {"clock": "unavailable", "windows": []}
    if raw is None or raw.get("clock") != "portaudio_dac" or not raw.get("valid", False):
        return result
    now = raw["device_time"] + max(0, age)
    result["clock"] = "portaudio_dac"
    result["stamp"] = raw.get("observed_monotonic", 0)
    for at, duration, offset, amplitude in raw["windows"]:
        start, end = (at - now) * 1000, (at + duration - now) * 1000
        if end <= 0 or start >= 300:
            continue
        result["windows"].append(
            {
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
                "position_s": round(offset, 6),
                "amplitude": amplitude,
                "cue": cue_at(alignment, offset + duration / 2),
            }
        )
        if len(result["windows"]) >= 16:
            break
    return result
