"""LLM Provider поверх публичного HTTP API; без импорта inference-кода."""

import json
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from typing import Any

from ramvi_service.contracts.modules import ProviderCapabilities
from ramvi_service.transport import local_opener, loopback_url

MAX_STREAM_BYTES = 4 * 2**20
MAX_STREAM_LINE = 2**20


def stream_chunks(event: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Перевести один OpenAI chunk движка в общие куски StreamingLLMProvider."""
    if "error" in event:
        error = event["error"]
        message = str(error.get("message", "")) if isinstance(error, dict) else str(error)
        if isinstance(error, dict) and error.get("type") == "cancelled":
            raise InterruptedError(message or "request cancelled")
        raise RuntimeError(message or "Engine error")
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            yield {"type": "reasoning", "text": delta["reasoning_content"]}
        if delta.get("content"):
            yield {"type": "text", "text": delta["content"]}
        if delta.get("tool_calls"):
            yield {"type": "tool_calls", "tool_calls": delta["tool_calls"]}
        if choice.get("finish_reason") is not None:
            yield {"type": "finish", "finish_reason": choice["finish_reason"]}
    if event.get("usage"):
        yield {"type": "usage", "usage": event["usage"]}


class LocalEngineProvider:
    capabilities = ProviderCapabilities(streaming=True)

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 30,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        base_url = loopback_url(base_url)
        if capabilities is not None:
            self.capabilities = capabilities
        self._opener = local_opener()
        self._lock = threading.RLock()
        self._operation = threading.Lock()
        self._active_request: str | None = None
        self.base_url, self.model, self.api_key, self.timeout = (
            base_url.rstrip("/"),
            model,
            api_key,
            timeout,
        )

    def _call(
        self, route: str, payload: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if request_id is not None:
            headers["X-Request-ID"] = request_id
        request = urllib.request.Request(
            self.base_url + route, json.dumps(payload).encode(), headers
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            data = response.read(4 * 2**20 + 1)
            if len(data) > 4 * 2**20:
                raise ValueError("Engine response exceeds the limit")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Engine response must be an object")
            return result

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if cancel.is_set():
            raise InterruptedError("request cancelled")
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("Provider is already running a request; use the shared queue")
        request_id = uuid.uuid4().hex
        try:
            with self._lock:
                self._active_request = request_id
            payload = dict(request, model=self.model, stream=False)
            result = self._call("/v1/chat/completions", payload, request_id)
            if cancel.is_set():
                raise InterruptedError("late response cancelled")
            return result
        finally:
            with self._lock:
                self._active_request = None
            self._operation.release()

    def stream(self, request: dict[str, Any], cancel: threading.Event) -> Iterator[dict[str, Any]]:
        """Читает SSE движка по мере генерации; отмена прерывает чтение сразу."""
        if cancel.is_set():
            raise InterruptedError("request cancelled")
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("Provider is already running a request; use the shared queue")
        request_id = uuid.uuid4().hex
        done = threading.Event()
        completed = False
        response: Any = None
        try:
            with self._lock:
                self._active_request = request_id
            payload = dict(
                request, model=self.model, stream=True, stream_options={"include_usage": True}
            )
            headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
            if self.api_key:
                headers["Authorization"] = "Bearer " + self.api_key
            response = self._opener.open(
                urllib.request.Request(
                    self.base_url + "/v1/chat/completions", json.dumps(payload).encode(), headers
                ),
                timeout=self.timeout,
            )
            opened = response

            def watch() -> None:
                # Отмена: остановить генерацию в движке и разбудить чтение.
                while not done.wait(0.01):
                    if cancel.is_set():
                        self.cancel()
                        try:
                            opened.close()
                        except OSError:
                            pass
                        return

            threading.Thread(target=watch, name="ramvi-stream-cancel", daemon=True).start()
            finish: str | None = None
            usage: dict[str, Any] | None = None
            total = 0
            while True:
                try:
                    line = response.readline(MAX_STREAM_LINE + 1)
                except (OSError, ValueError, AttributeError):
                    if cancel.is_set():
                        raise InterruptedError("request cancelled") from None
                    raise
                if cancel.is_set():
                    raise InterruptedError("request cancelled")
                if not line:
                    raise RuntimeError("Engine stream ended without [DONE]")
                total += len(line)
                if len(line) > MAX_STREAM_LINE or total > MAX_STREAM_BYTES:
                    raise ValueError("Engine stream exceeds the limit")
                text = line.decode().strip()
                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("Engine chunk must be an object")
                for chunk in stream_chunks(event):
                    if chunk["type"] == "finish":
                        finish = chunk["finish_reason"]
                    elif chunk["type"] == "usage":
                        usage = chunk["usage"]
                    else:
                        yield chunk
            completed = True
            yield {"type": "end", "finish_reason": finish, "usage": usage}
        finally:
            done.set()
            if not completed and response is not None:
                self.cancel()  # поток брошен или отменён: генерация в движке не должна идти дальше
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass
            with self._lock:
                self._active_request = None
            self._operation.release()

    def cancel(self) -> None:
        with self._lock:
            request_id = self._active_request
        if request_id is None:
            return
        try:
            self._call("/engine/cancel", {"request_id": request_id})
        except (OSError, urllib.error.URLError):
            pass

    def unload(self) -> None:
        self._call("/engine/unload", {})
