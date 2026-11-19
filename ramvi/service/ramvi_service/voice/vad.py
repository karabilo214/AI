"""Silero VAD in a disposable worker; no model/device imports at service activation."""

import os
import sys
from pathlib import Path
from typing import Any

from ramvi_service.voice.stt import WhisperProcess


class VADProcess:
    def __init__(self, model: Path) -> None:
        service = Path(__file__).resolve().parents[2]
        paths = [str(service), str(service.parent / "engine"), str(service.parent / "storage")]
        paths.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
        command = [
            sys.executable,
            "-P",
            "-c",
            "import sys,runpy; n=int(sys.argv[1]); sys.path[:0]=sys.argv[2:2+n]; "
            "sys.argv=['ramvi_service.voice.vad_worker']+sys.argv[2+n:]; "
            "runpy.run_module('ramvi_service.voice.vad_worker',run_name='__main__')",
            str(len(paths)),
            *paths,
        ]
        self._process = WhisperProcess(model, threads=1, command=command, timeout=15)

    def prepare(self) -> None:
        self._process.prepare()
        self._process.process.request("ping", {}, timeout=15)

    def speech(self, pcm: bytes, rate: int) -> bool:
        return bool(self._process.transcribe(pcm, rate, "auto")["speech"])

    def close(self) -> None:
        self._process.close()


class Silero:
    """Pinned Silero ONNX on CPU; does not depend on the broken pywhispercpp VAD wrapper."""

    def __init__(self, model: str) -> None:
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-not-found]

        self.np: Any = np
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model, sess_options=options, providers=["CPUExecutionProvider"]
        )

    def speech(self, pcm: bytes, rate: int) -> bool:
        from ramvi_service.voice.worker import to_16k

        np = self.np
        audio = to_16k(pcm, rate)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        probabilities = []
        for offset in range(0, len(audio), 512):
            frame = audio[offset : offset + 512]
            if len(frame) < 512:
                frame = np.pad(frame, (0, 512 - len(frame)))
            data = np.concatenate([context, frame.reshape(1, -1)], axis=1)
            probability, state = self.session.run(
                None, {"input": data, "state": state, "sr": np.array(16000, dtype=np.int64)}
            )
            probabilities.append(float(probability[0][0]))
            context = data[:, -64:]
        return bool(probabilities and max(probabilities[-3:]) >= 0.5)

    def close(self) -> None:
        self.session = None
