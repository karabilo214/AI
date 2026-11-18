"""Процесс распознавания: `python -m ramvi_service.voice.worker --model <файл> --threads N`.

Отдельный процесс держит модель whisper.cpp: падение или нехватка памяти не роняют
сервис, а остановка процесса сразу освобождает память модели.
Протокол — JSONL RPC `ProcessSupervisor` (как у module worker). Звук приходит через
общую память (`multiprocessing.shared_memory`), не через диск и не через stdout:
строка RPC ограничена 64 КБ.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from multiprocessing import shared_memory
from typing import Any

from ramvi_service.voice.models import join_segments

MAX_LINE = 65536
RATE = 16_000
SPEECH_PAD_MS = 1000  # ADR-0004: при 30 и 300 мс точность ниже (FLEURS RU/UK/EN)


def to_16k(pcm: bytes, rate: int) -> Any:
    """int16 → float32 16 кГц. Кратная частота — среднее по блокам, иначе интерполяция."""
    import numpy as np

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if rate == RATE or audio.size == 0:
        return audio
    if rate % RATE == 0:
        step = rate // RATE
        usable = audio[: audio.size // step * step]
        return usable.reshape(-1, step).mean(axis=1)
    count = int(audio.size * RATE / rate)
    return np.interp(np.arange(count) * (rate / RATE), np.arange(audio.size), audio).astype(
        np.float32
    )


class Whisper:
    def __init__(self, model: str, threads: int, vad: str | None = None) -> None:
        import _pywhispercpp as pw
        from pywhispercpp.model import Model

        self._pw = pw
        # Детектор речи: тишина и шум не доходят до whisper (иначе он выдумывает текст).
        self._vad = {"vad": True, "vad_model_path": vad} if vad else {}
        self.model = Model(
            model,
            n_threads=threads,
            context_params={"use_gpu": False},  # VOI-03: видеокарта остаётся основной модели
            print_progress=False,
            print_realtime=False,
            redirect_whispercpp_logs_to=None,
        )
        if vad:
            # Запас вокруг речи: при 30 мс (по умолчанию) обрезаются первые звуки слов.
            params = self.model._params.vad_params
            params.speech_pad_ms = SPEECH_PAD_MS
            self.model._params.vad_params = params

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
        audio = to_16k(pcm, rate)
        started = time.perf_counter()
        segments = self.model.transcribe(audio, language=language, **self._vad)
        lang_id = self._pw.whisper_full_lang_id(self.model._ctx)
        return {
            "text": join_segments([s.text for s in segments]),
            "language": self._pw.whisper_lang_str(lang_id) if language == "auto" else language,
            "audio_s": round(audio.size / RATE, 2),
            "proc_s": round(time.perf_counter() - started, 2),
        }


def read_shared(name: str, size: int) -> bytes:
    block = shared_memory.SharedMemory(name=name)
    if sys.platform != "win32":  # память создал и удалит сервис, а не этот процесс (Python 3.12)
        from multiprocessing import resource_tracker

        resource_tracker.unregister(block._name, "shared_memory")  # type: ignore[attr-defined]
    try:
        buf = block.buf
        assert buf is not None
        return bytes(buf[:size])
    finally:
        block.close()


def lower_priority() -> None:
    """5.4 VOI-03: распознавание на CPU уступает интерфейсу и LLM (ниже обычного)."""
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
        else:
            os.nice(5)
    except (OSError, AttributeError):
        pass  # без прав — обычный приоритет


def run(model: str, threads: int, vad: str | None = None) -> None:
    lower_priority()
    with contextlib.redirect_stdout(sys.stderr):  # stdout — только ответы RPC
        whisper = Whisper(model, threads, vad)
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            return  # сервис закрыл stdin — выходим
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            method, payload = request["method"], request["payload"]
            if method == "ping":
                result: dict[str, Any] = {}
            elif method == "transcribe":
                pcm = read_shared(str(payload["shm"]), int(payload["size"]))
                with contextlib.redirect_stdout(sys.stderr):
                    result = whisper.transcribe(pcm, int(payload["rate"]), str(payload["language"]))
                del pcm
            else:
                raise ValueError("unknown method")
            encoded = json.dumps({"id": request["id"], "result": result}, ensure_ascii=False)
        except Exception as error:
            encoded = json.dumps(
                {
                    "id": request.get("id"),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        print(encoded, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ramvi-stt-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vad")
    args = parser.parse_args(argv)
    run(args.model, max(1, args.threads), args.vad)


if __name__ == "__main__":
    main()
