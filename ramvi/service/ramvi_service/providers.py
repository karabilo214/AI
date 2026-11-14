"""LLM Provider поверх публичного HTTP API; без импорта inference-кода."""

import json
import threading
import urllib.error
import urllib.request
import uuid
from typing import Any

from ramvi_service.contracts.modules import ProviderCapabilities
from ramvi_service.transport import local_opener, loopback_url


class LocalEngineProvider:
    capabilities = ProviderCapabilities()

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
                raise ValueError("ответ Engine превышает лимит")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("ответ Engine должен быть объектом")
            return result

    def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if cancel.is_set():
            raise InterruptedError("запрос отменён")
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("Provider уже выполняет запрос; используйте общую очередь")
        request_id = uuid.uuid4().hex
        try:
            with self._lock:
                self._active_request = request_id
            payload = dict(request, model=self.model, stream=False)
            result = self._call("/v1/chat/completions", payload, request_id)
            if cancel.is_set():
                raise InterruptedError("поздний ответ отменён")
            return result
        finally:
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
