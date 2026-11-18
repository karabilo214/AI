"""5.4 VOI-03: распознавание на видеокарте — `whisper-server` из официальной сборки CUDA.

Процесс живёт одну запись: запускается, пока человек говорит (модель грузится 3 с),
распознаёт и сразу останавливается — видеопамять не держится и не мешает LLM.
Сервер слушает только 127.0.0.1 на свободном порту; звук передаётся телом запроса
(WAV в памяти), временных файлов сервер не пишет.
"""

import array
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

from ramvi_service.chat import ChatError
from ramvi_service.voice.models import VAD, join_segments

RATE = 16_000
SPEECH_PAD_MS = 1000  # как у процесса на CPU (worker.SPEECH_PAD_MS)
NAMES = {"russian": "ru", "english": "en", "ukrainian": "uk"}


def to_16k_pcm(pcm: bytes, rate: int) -> bytes:
    """int16 любой частоты → int16 16 кГц без numpy (сервис работает и без него)."""
    if rate == RATE:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if rate % RATE == 0:  # кратная частота — среднее по блокам
        step = rate // RATE
        out = array.array(
            "h",
            (sum(samples[i : i + step]) // step for i in range(0, len(samples) - step + 1, step)),
        )
        return out.tobytes()
    count = int(len(samples) * RATE / rate)
    ratio = rate / RATE
    out = array.array("h")
    for i in range(count):
        x = i * ratio
        j = int(x)
        nxt = samples[j + 1] if j + 1 < len(samples) else samples[j]
        out.append(int(samples[j] + (nxt - samples[j]) * (x - j)))
    return out.tobytes()


def wav_bytes(pcm16k: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16k)
    return buffer.getvalue()


def language_code(reply: dict[str, Any], asked: str) -> str:
    if asked != "auto":
        return asked
    probabilities = reply.get("language_probabilities")
    if isinstance(probabilities, dict) and probabilities:
        return str(max(probabilities, key=lambda k: probabilities[k]))
    name = str(reply.get("detected_language") or reply.get("language") or "")
    return NAMES.get(name, name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class WhisperServer:
    """Transcriber для видеокарты: один процесс на одну запись."""

    single_use = True  # сервис останавливает процесс сразу после распознавания

    def __init__(
        self,
        model: Path,
        folder: Path,
        threads: int = 4,
        command: list[str] | None = None,
        start_timeout: float = 120.0,  # первый запуск на новой видеокарте собирает ядра ~40 с
        timeout: float = 120.0,
    ) -> None:
        from ramvi_engine.processes import ProcessSupervisor

        self.model, self.port = Path(model), free_port()
        vad = self.model.parent / VAD.file
        argv = command or [str(Path(folder) / "whisper-server.exe")]
        args = [
            *argv,
            "-m",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-t",
            str(threads),
            "-l",
            "auto",
        ]
        if vad.is_file():
            args += ["--vad", "-vm", str(vad), "-vp", str(SPEECH_PAD_MS)]
        self.process = ProcessSupervisor(args, cwd=Path(folder), timeout=30)
        self.start_timeout, self.timeout = start_timeout, timeout
        self._lock = threading.Lock()
        self._ready = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return bool(self.process.status()["state"] == "running")

    def prepare(self) -> None:
        """Запустить сервер и дождаться загрузки модели."""
        with self._lock:
            if self._ready and self.running:
                return
            self._ready = False
            self.process.start()
            deadline = time.monotonic() + self.start_timeout
            while time.monotonic() < deadline:
                if not self.running:
                    break
                try:
                    with urllib.request.urlopen(self.url + "/", timeout=1):
                        self._ready = True
                        return
                except (urllib.error.URLError, OSError):
                    time.sleep(0.05)
            log = " | ".join(self.process.diagnostics()[-3:])
            self.process.stop()
            raise ChatError("voice.stt_failed", detail=f"whisper-server did not start {log}"[:500])

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
        self.prepare()
        boundary = uuid.uuid4().hex
        fields = (("response_format", "verbose_json"), ("language", language))
        body = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields
        )
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body += wav_bytes(to_16k_pcm(pcm, rate)) + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.url + "/inference",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            log = " | ".join(self.process.diagnostics()[-3:])
            raise ChatError("voice.stt_failed", detail=f"{error} {log}".strip()[:500]) from None
        finally:
            del body
        if not isinstance(reply, dict) or "error" in reply:
            raise ChatError("voice.stt_failed", detail=str(reply)[:500])
        segments = reply.get("segments")
        texts = (
            [str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)]
            if isinstance(segments, list) and segments
            else [str(reply.get("text", ""))]
        )
        return {"text": join_segments(texts), "language": language_code(reply, language)}

    def close(self) -> None:
        with self._lock:
            self._ready = False
            self.process.stop()
