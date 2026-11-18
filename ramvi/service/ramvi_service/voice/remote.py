"""5.7 VOI-API-01: распознавание через сторонний API (OpenAI-совместимый стандарт).

`POST {адрес}/audio/transcriptions`, multipart: `file` (WAV 16 кГц моно), `model`,
`language` (если задан), `response_format=json`. Так работают OpenAI, Groq и многие
другие сервисы, а также локальные серверы в сети. Адрес — до `/audio`, например
`https://api.openai.com/v1`. Только HTTPS (HTTP — лишь к 127.0.0.1/localhost).

Ключ берётся из хранилища секретов при каждом запросе и в журналы не попадает;
переадресации не выполняются (иначе заголовок с ключом ушёл бы на другой адрес).
Звук уходит на сервер — поэтому способ выключен по умолчанию и у микрофона видна метка.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from ramvi_service import web
from ramvi_service.chat import ChatError
from ramvi_service.voice.server import to_16k_pcm, wav_bytes


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Переадресация не выполняется: иначе ключ ушёл бы на другой адрес."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


OPENER = urllib.request.build_opener(_NoRedirects())


def host_of(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


class RemoteTranscriber:
    """SpeechRecognizer для стороннего API; процессов не держит."""

    def __init__(
        self, url: str, model: str, key: Callable[[], str | None], timeout: float = 60.0
    ) -> None:
        self.url, self.model, self.key, self.timeout = url.rstrip("/"), model, key, timeout

    @property
    def running(self) -> bool:
        return False

    def prepare(self) -> None:
        pass

    def close(self) -> None:
        pass

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
        endpoint = self.url + "/audio/transcriptions"
        host = host_of(endpoint)
        if not web.allowed(endpoint):
            raise ChatError("voice.api_only_https", host=host)
        key = self.key()
        boundary = uuid.uuid4().hex
        fields = [("model", self.model), ("response_format", "json")]
        if language != "auto":
            fields.append(("language", language))
        body = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields
        )
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body += wav_bytes(to_16k_pcm(pcm, rate)) + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = web.request(endpoint, headers)
        request.data = body
        request.method = "POST"
        try:
            with OPENER.open(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise ChatError("voice.api_key_rejected", host=host) from None
            detail = error.read()[:300].decode("utf-8", "replace")
            raise ChatError(
                "voice.api_error", host=host, status=error.code, detail=detail
            ) from None
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ChatError("voice.api_unreachable", host=host, detail=str(reason)[:200]) from None
        except ValueError:
            raise ChatError("voice.api_error", host=host, status=200, detail="not JSON") from None
        finally:
            del body
        if not isinstance(reply, dict) or not isinstance(reply.get("text", ""), str):
            raise ChatError("voice.api_error", host=host, status=200, detail=str(reply)[:200])
        detected = reply.get("language")
        return {
            "text": " ".join(str(reply.get("text", "")).split()),
            "language": language if language != "auto" else (detected or "auto"),
        }
