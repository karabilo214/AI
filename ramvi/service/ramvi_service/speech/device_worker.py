"""PCM playback. No sounddevice import or audio device until first validated audio."""

import sys
from typing import Any


class DeviceOutput:
    def __init__(self, rate: int) -> None:
        from collections import deque

        from ramvi_service.voice.audio import _com_init

        if sys.platform == "win32":
            _com_init()
        import sounddevice as sd

        self.rate = rate
        self.frames = rate // 50
        self.silence = bytes(self.frames * 2)
        self.pending: Any = deque()
        self.last_dac: float | None = None
        self.offset = 0
        self.windows: Any = deque(maxlen=128)
        self.valid = True
        self.stream: Any = sd.RawOutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=self.frames,
            callback=self._callback,
        )

    def _callback(self, outdata: Any, frames: int, timing: Any, status: Any) -> None:
        # No provider, RPC, file access, RMS computation or waiting in the callback.
        outdata[:] = self.silence
        self.last_dac = float(timing.outputBufferDacTime)
        self.valid = not bool(status)
        if status:
            self.windows.clear()
        if not self.pending:
            return
        pcm, amplitude = self.pending.popleft()
        outdata[: len(pcm)] = pcm
        duration = len(pcm) / 2 / self.rate
        if not status:
            self.windows.append(
                (timing.outputBufferDacTime, duration, self.offset / self.rate, amplitude)
            )
        self.offset += len(pcm) // 2

    def start(self) -> None:
        self._com()
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            raise

    def write(self, pcm: bytes) -> dict[str, Any]:
        import math
        import struct
        import time

        size = self.frames * 2
        for start in range(0, len(pcm), size):
            block = pcm[start : start + size]
            samples = struct.unpack("<" + "h" * (len(block) // 2), block)
            amplitude = round(math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768, 6)
            while len(self.pending) >= 4:
                if not self.stream.active:
                    raise RuntimeError("audio callback stopped")
                time.sleep(0.001)
            if not self.stream.active:
                raise RuntimeError("audio callback stopped")
            self.pending.append((block, amplitude))
        return self.position()

    def position(self) -> dict[str, Any]:
        import time

        device_time = float(self.stream.time)
        windows = list(self.windows)
        # Queued PCM is projected from the last hardware callback's sample edge,
        # never from LLM/synthesis/write start. Later callbacks replace predictions.
        if self.last_dac is not None and self.valid:
            at = self.last_dac + self.frames / self.rate
            offset = self.offset / self.rate
            for pcm, amplitude in self.pending:
                duration = len(pcm) / 2 / self.rate
                windows.append((at, duration, offset, amplitude))
                at += self.frames / self.rate
                offset += duration
        return {
            "clock": "portaudio_dac",
            "device_time": device_time,
            "observed_monotonic": time.monotonic(),
            "valid": self.valid,
            "windows": windows,
            "forecast_blocks": len(self.pending),
        }

    def drain(self) -> None:
        import time

        while self.pending:
            if not self.stream.active:
                raise RuntimeError("audio callback stopped")
            time.sleep(0.002)
        windows = list(self.windows)
        end = windows[-1][0] + windows[-1][1] if windows else 0
        while self.stream.time < end:
            if not self.stream.active:
                raise RuntimeError("audio callback stopped")
            time.sleep(0.002)

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
                telemetry = None
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
                    telemetry = output.write(pcm)
                elif method == "position" and output is not None:
                    telemetry = output.position()
                elif method == "drain" and output is not None:
                    output.drain()
                    telemetry = output.position()
                elif method == "close" and output is not None:
                    output.close()
                    output = None
                else:
                    raise ValueError("unknown audio operation")
                result = {"id": request["id"], "result": {"ok": True, "playback": telemetry}}
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
