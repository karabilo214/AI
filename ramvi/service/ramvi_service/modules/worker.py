"""Изолированный module worker: ограниченный JSONL RPC поверх stdio."""

import contextlib
import importlib
import json
import re
import sys
from typing import Any

from ramvi_service.services import ServiceClient

MAX_LINE = 65536


def run(entrypoint: str) -> None:
    package, name = entrypoint.split(":", 1)
    with contextlib.redirect_stdout(sys.stderr):
        module = getattr(importlib.import_module(package), name)()
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            return
        if len(line) > MAX_LINE:
            raise ValueError("module RPC line exceeds limit")
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"id", "method", "payload"}:
                raise ValueError("invalid RPC envelope")
            method, payload = request["method"], request["payload"]
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            with contextlib.redirect_stdout(sys.stderr):
                if method == "invoke":
                    extension = payload["method"]
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", extension) or extension in {
                        "initialize",
                        "activate",
                        "deactivate",
                        "suspend",
                        "resume",
                        "dispose",
                        "checkpoint",
                    }:
                        raise ValueError("invalid extension method")
                    result = getattr(module, extension)(**payload.get("arguments", {}))
                elif method in {
                    "initialize",
                    "activate",
                    "deactivate",
                    "suspend",
                    "resume",
                    "dispose",
                    "checkpoint",
                }:
                    if method == "initialize":
                        context = dict(payload["context"])
                        services = context.get("services")
                        if services:
                            context["services"] = ServiceClient(services["url"], services["token"])
                        result = module.initialize(context)
                    elif method == "checkpoint":
                        hook = getattr(module, "checkpoint", None)
                        result = hook() if hook else {}
                    else:
                        result = getattr(module, method)()
                else:
                    raise ValueError("unknown lifecycle method")
            if result is None:
                result = {}
            if not isinstance(result, dict):
                raise ValueError("module result must be a JSON object")
            response = {"id": request["id"], "result": result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode()) > MAX_LINE:
                raise ValueError("module result exceeds limit")
        except Exception as error:
            encoded = json.dumps(
                {
                    "id": request.get("id"),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        print(encoded, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: worker package.module:factory")
    run(sys.argv[1])
