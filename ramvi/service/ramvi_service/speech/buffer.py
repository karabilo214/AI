"""Bounded PCM handoff. Producer backpressure is cancellable, never drops audio."""

import queue
import threading
import time
from typing import Any

from ramvi_service.contracts.speech import AudioPiece

PCM_BLOCK_MS = 20
PCM_BLOCKS = 4


class PCMBuffer:
    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.queue: queue.Queue[AudioPiece] = queue.Queue(PCM_BLOCKS)
        self.done = threading.Event()
        self.error: Exception | None = None
        self.high_water = 0

    def put(self, piece: AudioPiece) -> None:
        while not self.cancel.is_set():
            try:
                self.queue.put(piece, timeout=0.02)
                self.high_water = max(self.high_water, self.queue.qsize())
                return
            except queue.Full:
                continue
        raise InterruptedError("speech cancelled")

    def get(
        self, deadline: float | None = None, code: str = "speech.audio_timeout"
    ) -> AudioPiece | None:
        while not self.cancel.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(code)
            try:
                return self.queue.get(timeout=0.02)
            except queue.Empty:
                if self.done.is_set():
                    if self.error:
                        raise self.error from None
                    return None
        raise InterruptedError("speech cancelled")

    def produce(self, provider: Any, request: Any) -> None:
        carry = b""
        rate: int | None = None
        total = 0
        try:
            for piece in provider.synthesize(request, self.cancel):
                if self.cancel.is_set():
                    raise InterruptedError("speech cancelled")
                if rate is not None and rate != piece.rate:
                    raise ValueError("speech format changed midstream")
                rate = piece.rate
                pcm = carry + piece.pcm
                total += len(piece.pcm)
                if total > rate * 2 * 120:
                    raise ValueError("speech duration exceeds limit")
                size = rate * 2 * PCM_BLOCK_MS // 1000
                complete = len(pcm) // size * size
                carry = pcm[complete:]
                for offset in range(0, complete, size):
                    self.put(AudioPiece(pcm[offset : offset + size], rate))
            if len(carry) % 2 or not total:
                raise ValueError("truncated or empty speech audio")
            if carry:
                assert rate is not None
                self.put(AudioPiece(carry, rate))
        except Exception as error:
            self.error = error
        finally:
            self.done.set()
