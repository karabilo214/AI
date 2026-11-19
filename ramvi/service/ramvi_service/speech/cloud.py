"""Independent cloud TTS adapters: fixed origins, no redirects, no raw-mic input."""

import base64
import http.client
import json
import math
import socket
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote, urlsplit

from ramvi_service.contracts.speech import AudioPiece, SpeechRequest
from ramvi_service.transport import loopback_url

ORIGINS = {
    "openai": "https://api.openai.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "elevenlabs": "https://api.elevenlabs.io",
}
MAX_RESPONSE = 12 * 1024 * 1024


class CloudSpeech:
    def __init__(
        self, provider: str, model: str, key: str, *, test_origin: str | None = None
    ) -> None:
        self.provider, self.model, self.key = provider, model, key
        self.origin = loopback_url(test_origin) if test_origin else ORIGINS[provider]
        self.lock = threading.Lock()
        self.connection: http.client.HTTPConnection | None = None
        self.socket: socket.socket | None = None
        self.timings: dict[str, Any] | None = None

    def voices(self) -> list[dict[str, Any]]:
        return []  # built-in catalog; no unsolicited network call to enumerate voices

    def _stream(self, path: str, body: dict[str, Any], cancel: Any) -> Iterator[bytes]:
        url = urlsplit(self.origin)
        if not url.hostname:
            raise ValueError("missing speech host")
        connection = (
            http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        )(url.hostname, url.port, timeout=15)
        with self.lock:
            self.connection = connection
        header = {
            "openai": "Authorization",
            "gemini": "x-goog-api-key",
            "elevenlabs": "xi-api-key",
        }[self.provider]
        value = "Bearer " + self.key if self.provider == "openai" else self.key
        try:
            connection.connect()
            with self.lock:
                self.socket = connection.sock
            if cancel.is_set():
                raise InterruptedError("speech cancelled")
            connection.request(
                "POST",
                path,
                json.dumps(body, ensure_ascii=False).encode(),
                {header: value, "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("speech.api_http_" + str(response.status))
            size = 0
            while chunk := response.read(8192):
                if cancel.is_set():
                    raise InterruptedError("speech cancelled")
                size += len(chunk)
                if size > MAX_RESPONSE:
                    raise ValueError("speech audio response exceeds limit")
                yield chunk
        finally:
            connection.close()
            with self.lock:
                if self.connection is connection:
                    self.connection = None
                    self.socket = None

    def synthesize(self, request: SpeechRequest, cancel: Any) -> Iterator[AudioPiece]:
        if self.provider == "openai":
            body: dict[str, Any] = {
                "model": self.model,
                "voice": request.voice,
                "input": request.text,
                "response_format": "pcm",
            }
            if request.style:
                body["instructions"] = request.style
            for chunk in self._stream("/v1/audio/speech", body, cancel):
                yield AudioPiece(chunk, 24000)
            return
        if self.provider == "gemini":
            content: dict[str, Any] = {"type": "text", "text": request.text}
            if request.style:
                content["annotations"] = [{"type": "speech_metadata", "style": request.style}]
            body = {
                "model": self.model,
                "input": [{"type": "user_input", "content": [content]}],
                "response_format": {
                    "type": "audio",
                    "mime_type": "audio/l16",
                    "sample_rate": 24000,
                },
                "generation_config": {"speech_config": [{"voice": request.voice}]},
            }
            result = json.loads(b"".join(self._stream("/v1beta/interactions", body, cancel)))
            parts = [
                content
                for step in result.get("steps", [])
                for content in step.get("content", [])
                if content.get("type") == "audio"
            ]
            if not parts:
                raise ValueError("missing Gemini audio")
            audio = b"".join(base64.b64decode(part["data"], validate=True) for part in parts)
        else:
            body = {
                "model_id": self.model,
                "text": request.text,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0},
            }
            path = (
                "/v1/text-to-speech/"
                + quote(request.voice, safe="")
                + "/with-timestamps?output_format=pcm_24000"
            )
            result = json.loads(b"".join(self._stream(path, body, cancel)))
            audio = base64.b64decode(result["audio_base64"], validate=True)
            alignment = result.get("alignment")
            if alignment is not None:
                self.timings = validate_alignment(alignment)
        if not audio or len(audio) > 24000 * 2 * 120:
            raise ValueError("invalid cloud audio size")
        for offset in range(0, len(audio), 8192):
            if cancel.is_set():
                raise InterruptedError("speech cancelled")
            yield AudioPiece(audio[offset : offset + 8192], 24000)

    def cancel(self) -> None:
        with self.lock:
            connection, active_socket = self.connection, self.socket
        # HTTPConnection may clear .sock after a Connection: close response while
        # HTTPResponse is still reading it; retain the connected socket for cancellation.
        if active_socket:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection:
            connection.close()

    def close(self) -> None:
        self.cancel()
        self.key = ""


def validate_alignment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("invalid TTS alignment")
    chars, starts, ends = (
        value.get(k)
        for k in ["characters", "character_start_times_seconds", "character_end_times_seconds"]
    )
    if (
        not isinstance(chars, list)
        or not isinstance(starts, list)
        or not isinstance(ends, list)
        or not len(chars) <= 2048
        or len(chars) != len(starts)
        or len(chars) != len(ends)
    ):
        raise ValueError("invalid TTS alignment lengths")
    previous = 0.0
    for char, start, end in zip(chars, starts, ends, strict=True):
        if (
            not isinstance(char, str)
            or len(char) != 1
            or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in [start, end]
            )
            or not previous <= start <= end <= 120
        ):
            raise ValueError("invalid TTS alignment times")
        previous = start
    result = {
        "kind": "characters",
        "source": "provider",
        "characters": chars,
        "starts": [round(v, 4) for v in starts],
        "ends": [round(v, 4) for v in ends],
    }
    if len(json.dumps(result).encode()) > 48000:
        return {"kind": "none", "source": "provider", "reason": "alignment_size_limit"}
    return result
