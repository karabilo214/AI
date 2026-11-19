"""PCM playback. No sounddevice import or audio device until first validated audio."""

import sys
from typing import Any


class SoundOutput:
    def __init__(self, rate: int) -> None:
        from ramvi_service.voice.audio import _com_init

        if sys.platform == "win32":
            _com_init()
        import sounddevice as sd

        self.stream: Any = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", blocksize=512
        )
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            raise

    def write(self, pcm: bytes) -> None:
        self.stream.write(pcm)

    def close(self) -> None:
        self._com()
        self.stream.stop()
        self.stream.close()

    @staticmethod
    def _com() -> None:
        if sys.platform == "win32":
            from ramvi_service.voice.audio import _com_init

            _com_init()

    def cancel(self) -> None:
        self._com()
        self.stream.abort()
