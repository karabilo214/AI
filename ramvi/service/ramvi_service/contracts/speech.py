"""Optional speech output contract: no Character semantics, device or vendor imports."""

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

SPEAKER = "audio_output"


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice: str = ""
    language: str = "auto"
    emotion: str = "neutral"  # opaque module-owned label, not an access grant
    intensity: float = 0
    style: str = ""

    def __post_init__(self) -> None:
        for value, limit in [
            (self.text, 2048),
            (self.voice, 255),
            (self.language, 16),
            (self.emotion, 64),
            (self.style, 512),
        ]:
            if not isinstance(value, str) or len(value.encode()) > limit:
                raise ValueError("invalid speech request")
        if (
            not self.text.strip()
            or isinstance(self.intensity, bool)
            or not isinstance(self.intensity, (int, float))
            or not math.isfinite(self.intensity)
            or not 0 <= self.intensity <= 1
        ):
            raise ValueError("invalid speech text/intensity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioPiece:
    pcm: bytes
    rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pcm, bytes)
            or len(self.pcm) > 65536
            or self.rate not in {16000, 22050, 24000, 44100, 48000}
            or self.channels != 1
        ):
            raise ValueError("unsupported speech audio format")


class SpeechProvider(Protocol):
    def voices(self) -> list[dict[str, Any]]: ...
    def synthesize(self, request: SpeechRequest, cancel: Any) -> Iterator[AudioPiece]: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
