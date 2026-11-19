"""PCM playback. No sounddevice import or audio device until first validated audio."""

import sys
from typing import Any


class DeviceOutput:
    def __init__(self, rate: int) -> None:
        from ramvi_service.voice.audio import _com_init

        if sys.platform == "win32":
            _com_init()
        import sounddevice as sd

        self.stream: Any = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", blocksize=512
        )

    def start(self) -> None:
        self._com()
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            raise

    def write(self, pcm: bytes) -> None:
        self.stream.write(pcm)

    def close(self) -> None:
        self._com()
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    @staticmethod
    def _com() -> None:
        if sys.platform == "win32":
            from ramvi_service.voice.audio import _com_init

            _com_init()

    def cancel(self) -> None:
        self._com()
        self.stream.abort()


# This worker owns PortAudio. Killing its process closes a stuck/lost device.
def main() -> None:
    import base64
    import json
    import sys

    output = None
    try:
        while line := sys.stdin.buffer.readline(65537):
            if len(line) > 65536:
                raise ValueError("audio RPC exceeds limit")
            request = {}
            try:
                request = json.loads(line)
                method, payload = request["method"], request["payload"]
                if method == "start":
                    rate = payload["rate"]
                    if (
                        isinstance(rate, bool)
                        or rate not in {16000, 22050, 24000, 44100, 48000}
                        or output is not None
                    ):
                        raise ValueError("invalid audio start")
                    output = DeviceOutput(rate)
                    output.start()
                elif method == "write" and output is not None:
                    pcm = base64.b64decode(payload["pcm"], validate=True)
                    if not pcm or len(pcm) > 8192 or len(pcm) % 2:
                        raise ValueError("invalid audio block")
                    output.write(pcm)
                elif method == "close" and output is not None:
                    output.close()
                    output = None
                else:
                    raise ValueError("unknown audio operation")
                result = {"id": request["id"], "result": {"ok": True}}
            except Exception:
                result = {
                    "id": request.get("id"),
                    "error": {"type": "RuntimeError", "message": "audio device operation failed"},
                }
            print(json.dumps(result), flush=True)
    finally:
        if output is not None:
            try:
                output.cancel()
            finally:
                output.close()


if __name__ == "__main__":
    main()
