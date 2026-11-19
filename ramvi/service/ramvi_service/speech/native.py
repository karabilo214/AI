"""Local OS speech synthesis in an owned worker. Text is carried over RPC/stdin."""

import base64
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ramvi_service.contracts.speech import AudioPiece, SpeechRequest


class SystemSpeech:
    def __init__(self) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        service = Path(__file__).resolve().parents[2]
        paths = [str(service), str(service.parent / "engine"), str(service.parent / "storage")]
        paths.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
        self.worker = ProcessSupervisor(
            [
                sys.executable,
                "-P",
                "-c",
                "import sys,runpy; sys.path[:0]=sys.argv[1:]; "
                "sys.argv=['ramvi_service.speech.native_worker']; "
                "runpy.run_module('ramvi_service.speech.native_worker',run_name='__main__')",
                *paths,
            ],
            timeout=45,
        )

    def voices(self) -> list[dict[str, Any]]:
        self.worker.start()
        return list(self.worker.request("voices", {}, timeout=20)["voices"])

    def synthesize(self, request: SpeechRequest, cancel: Any) -> Iterator[AudioPiece]:
        if cancel.is_set():
            raise InterruptedError("speech cancelled")
        self.worker.start()
        # Core owns the temporary directory, so cancellation of the child cannot orphan audio.
        with tempfile.TemporaryDirectory(prefix="ramvi-speech-") as directory:
            result = self.worker.request(
                "synthesize",
                {**request.to_dict(), "path": str(Path(directory) / "speech.wav")},
                timeout=45,
            )
            size, rate = result["size"], result["rate"]
            if not isinstance(size, int) or not 0 < size <= rate * 2 * 120:
                raise ValueError("invalid native speech length")
            offset = 0
            while offset < size:
                if cancel.is_set():
                    raise InterruptedError("speech cancelled")
                result = self.worker.request("read", {"offset": offset}, timeout=5)
                pcm = base64.b64decode(result["pcm"], validate=True)
                if not pcm or len(pcm) > 16384:
                    raise ValueError("invalid native speech chunk")
                offset += len(pcm)
                yield AudioPiece(pcm, rate)

    def cancel(self) -> None:
        self.worker.stop(timeout=3)

    def close(self) -> None:
        self.cancel()
