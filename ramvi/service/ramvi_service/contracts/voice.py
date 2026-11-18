"""5.6 CORE-VOICE-01: публичный контракт захвата звука и распознавания речи.

Контракт лёгкий: без импорта аудиобиблиотек, whisper, TTS и Character. Его реализуют
адаптеры (`voice/audio.py` — захват, `voice/stt.py` и `voice/server.py` — распознавание),
а пользуются им чат (кнопка микрофона) и модули — через Scoped Services `voice.*`.
Character повторно использует захват и распознавание, оставляя свою оркестрацию и
эмоциональную политику внутри модуля.

Микрофоном одновременно владеет один: запись берёт аренду ресурса `audio_input`
у общего `ResourceBroker` и отдаёт её после остановки или отмены.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

MICROPHONE = "audio_input"  # единица ресурса в ResourceBroker: один микрофон — одна запись


@dataclass(frozen=True)
class SpeechCapabilities:
    """Что умеет распознавание сейчас."""

    languages: tuple[str, ...]  # коды языков; "auto" — определить самому
    auto_language: bool = True
    streaming: bool = False  # потоковая транскрипция — VOI-04 (следующая версия)
    cancel: bool = True  # отмена записи и идущего распознавания
    backends: tuple[str, ...] = ("cpu",)  # где можно считать: cpu, gpu
    max_seconds: float = 120.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "languages": list(self.languages),
            "auto_language": self.auto_language,
            "streaming": self.streaming,
            "cancel": self.cancel,
            "backends": list(self.backends),
            "max_seconds": self.max_seconds,
            **self.extra,
        }


@dataclass(frozen=True)
class Device:
    """Устройство ввода звука."""

    name: str
    default: bool = False


class AudioStream(Protocol):
    rate: int

    def close(self) -> None: ...


class AudioCapture(Protocol):
    """Захват звука: блоки int16 моно приходят в `on_block`, в памяти, без диска."""

    def devices(self) -> list[Device]: ...

    def open(self, device: str | None, on_block: Any) -> AudioStream: ...


class SpeechRecognizer(Protocol):
    """Распознавание записанного звука (int16 моно, любая частота) в текст."""

    def prepare(self) -> None: ...

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]: ...

    def close(self) -> None: ...

    @property
    def running(self) -> bool: ...
