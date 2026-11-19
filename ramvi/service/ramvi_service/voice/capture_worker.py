"""Microphone-only native worker. No STT/LLM imports or audio files."""

import base64
import json
import queue
import sys
import time
from typing import Any

from ramvi_service.voice.audio import SoundDeviceMicrophone


def main() -> None:
    blocks: queue.Queue[tuple[float, bytes]] = queue.Queue(32)
    overflow = False
    stream = None

    def capture(pcm: bytes) -> None:
        nonlocal overflow
        try:
            blocks.put_nowait((time.monotonic(), pcm))
        except queue.Full:
            overflow = True

    try:
        while line := sys.stdin.buffer.readline(65537):
            if len(line) > 65536:
                raise ValueError("capture RPC exceeds limit")
            request = {}
            try:
                request = json.loads(line)
                result: dict[str, Any]
                if request["method"] == "start" and stream is None:
                    stream = SoundDeviceMicrophone().open(request["payload"].get("device"), capture)
                    result = {"rate": stream.rate}
                elif request["method"] == "read" and stream is not None:
                    if overflow:
                        raise RuntimeError("microphone queue overflow")
                    batch = []
                    try:
                        batch.append(blocks.get(timeout=0.1))
                    except queue.Empty:
                        pass
                    # One block at 192 kHz / 100 ms still fits the 64 KiB RPC.
                    if batch and len(batch[0][1]) > 38400:
                        raise ValueError("microphone block exceeds limit")
                    result = {
                        "blocks": [base64.b64encode(pcm).decode() for _, pcm in batch],
                        "captured_at": [captured for captured, _ in batch],
                    }
                else:
                    raise ValueError("invalid capture operation")
                response = {"id": request["id"], "result": result}
            except Exception:
                response = {
                    "id": request.get("id"),
                    "error": {"type": "RuntimeError", "message": "microphone operation failed"},
                }
            print(json.dumps(response), flush=True)
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
