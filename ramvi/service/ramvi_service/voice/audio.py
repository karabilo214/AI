"""Захват звука: список устройств ввода и запись в память (16-бит, моно).

Звук держится только в памяти процесса сервиса и на диск не пишется (VOI-02).
`SoundDeviceMicrophone` — PortAudio через `sounddevice`, импортируется лениво;
в тестах подставляется свой `Microphone`.
"""

import array
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from ramvi_service.chat import ChatError
from ramvi_service.contracts.voice import AudioCapture, AudioStream, Device

__all__ = ["Device", "Microphone", "Recorder", "SoundDeviceMicrophone", "Stream", "peak"]

RATE = 16_000  # whisper ждёт 16 кГц; другую частоту устройства пересчитывает процесс распознавания
BLOCK_SECONDS = 0.1


# Типы — из публичного контракта (5.6); прежние имена оставлены для кода и тестов.
Microphone = AudioCapture
Stream = AudioStream


def peak(block: bytes) -> int:
    """Наибольшая амплитуда блока int16 (0…32768)."""
    samples = array.array("h")
    samples.frombytes(block[: len(block) // 2 * 2])
    return max((abs(s) for s in samples), default=0)


class Recorder:
    """Одна запись: блоки копятся в памяти; не дольше `max_seconds`."""

    def __init__(
        self,
        microphone: Microphone,
        device: str | None,
        max_seconds: float = 120.0,
        on_limit: Callable[[], None] | None = None,
    ) -> None:
        self.max_seconds, self.on_limit = max_seconds, on_limit
        self._data = bytearray()
        self._lock = threading.Lock()
        self._peak = 0
        self._level = 0
        self._full = False
        self.started = time.monotonic()
        # Частота известна только после открытия, а PortAudio может прислать первые
        # блоки раньше: до того они копятся без предела, потом лишнее отрезается.
        self.rate = 0
        self._stream = microphone.open(device, self._on_block)
        with self._lock:
            self.rate = self._stream.rate
            notify = self._trim()
        self._notify(notify)

    def _trim(self) -> bool:
        """Под замком: отрезать сверх предела; True — предел достигнут впервые."""
        if self._full or not self.rate:
            return False
        limit = int(self.max_seconds * self.rate) * 2
        if len(self._data) < limit:
            return False
        del self._data[limit:]
        self._full = True
        return True

    def _notify(self, reached: bool) -> None:
        if reached and self.on_limit is not None:
            self.on_limit()  # запись дошла до предела — сервис останавливает её сам

    def _on_block(self, block: bytes) -> None:
        with self._lock:
            if self._full:
                return
            self._data += block
            level = peak(block)
            self._level = level
            self._peak = max(self._peak, level)
            reached = self._trim()
        self._notify(reached)

    @property
    def seconds(self) -> float:
        with self._lock:
            return len(self._data) / 2 / self.rate if self.rate else 0.0

    @property
    def level(self) -> float:
        """Громкость последнего блока 0…1 — для индикатора записи."""
        return min(1.0, self._level / 32768)

    def finish(self) -> tuple[bytes, int, int]:
        """Остановить поток; вернуть (звук, частота, наибольшая амплитуда) и забыть звук."""
        self._stream.close()
        with self._lock:
            data, self._data = bytes(self._data), bytearray()
            return data, self.rate, self._peak


def _com_init() -> None:
    """Windows: COM в текущем потоке. Сервис обрабатывает каждый HTTP-запрос в своём потоке,
    а WASAPI без COM в потоке не запускает запись (ошибка «Unanticipated host error»).
    Повторный вызов безвреден (S_FALSE)."""
    import ctypes

    ctypes.windll.ole32.CoInitializeEx(None, 0)  # type: ignore[attr-defined,unused-ignore]


class SoundDeviceMicrophone:
    """PortAudio: устройства одного интерфейса, без дублей.

    Windows — WASAPI, если он есть: полные имена (у MME обрезаны до 31 символа) и нет
    служебного «Microsoft Sound Mapper»; запись 16 кГц моно — через `auto_convert`
    (без него WASAPI открывает только частоту устройства). Иначе — интерфейс системного
    устройства по умолчанию.
    """

    def __init__(self, platform: str = sys.platform, com: Callable[[], None] | None = None) -> None:
        self.platform = platform
        self.com = com or (_com_init if platform == "win32" else (lambda: None))

    def _sd(self) -> Any:
        try:
            import sounddevice
        except (ImportError, OSError) as error:  # нет пакета или библиотеки PortAudio
            raise ChatError("voice.unavailable", detail=str(error)) from None
        return sounddevice

    def _inputs(self) -> tuple[Any, list[tuple[int, str]], int | None, bool]:
        """(модуль, [(индекс, имя)], индекс по умолчанию, это WASAPI)."""
        self.com()
        sd = self._sd()
        try:
            devices = sd.query_devices()
            default = sd.default.device[0]
            apis = sd.query_hostapis() if self.platform == "win32" else []
        except Exception as error:  # PortAudio не поднялся
            raise ChatError("voice.device_error", detail=str(error)) from None
        default = default if isinstance(default, int) and default >= 0 else None
        host = devices[default]["hostapi"] if default is not None else None
        wasapi = False
        for index, api in enumerate(apis):
            if "WASAPI" in str(api["name"]) and api["default_input_device"] >= 0:
                host, default, wasapi = index, int(api["default_input_device"]), True
                break
        found = [
            (i, str(d["name"]))
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0 and (host is None or d["hostapi"] == host)
        ]
        return sd, found, default, wasapi

    def devices(self) -> list[Device]:
        _, found, default, _ = self._inputs()
        return [Device(name, i == default) for i, name in found]

    def open(self, device: str | None, on_block: Callable[[bytes], None]) -> Stream:
        sd, found, default, wasapi = self._inputs()
        if not found:
            raise ChatError("voice.no_device")
        if device:
            index = next((i for i, name in found if name == device), None)
            if index is None:
                raise ChatError("voice.device_missing", device=device)
        else:
            index = default if default is not None else found[0][0]

        def callback(indata: Any, frames: int, when: Any, status: Any) -> None:
            on_block(bytes(indata))

        native = int(sd.query_devices(index)["default_samplerate"])
        tries: list[tuple[int, Any]] = []
        if wasapi:
            tries.append((RATE, sd.WasapiSettings(auto_convert=True)))
        tries += [(RATE, None), (native, None)]
        last: Exception | None = None
        for rate, extra in dict.fromkeys(tries):
            try:
                stream = sd.RawInputStream(
                    samplerate=rate,
                    blocksize=int(rate * BLOCK_SECONDS),
                    device=index,
                    channels=1,
                    dtype="int16",
                    callback=callback,
                    extra_settings=extra,
                )
                stream.start()
            except Exception as error:  # частота не поддерживается или устройство занято
                last = error
                continue
            return _SoundDeviceStream(stream, rate, self.com)
        raise ChatError("voice.device_error", detail=str(last))


class _SoundDeviceStream:
    def __init__(self, stream: Any, rate: int, com: Callable[[], None]) -> None:
        self._stream, self.rate, self.com = stream, rate, com

    def close(self) -> None:
        self.com()  # остановка приходит другим HTTP-запросом — из другого потока
        try:
            self._stream.stop()
        finally:
            self._stream.close()
