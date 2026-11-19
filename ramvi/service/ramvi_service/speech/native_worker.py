"""System voice inventory and PCM synthesis. Native child is in the supervisor's Job/group."""

import base64
import json
import re
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

MAX_LINE = 65536


def windows(operation: str, **fields: Any) -> dict[str, Any]:
    script = Path(__file__).with_name("native_windows.ps1")
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        input=json.dumps({"operation": operation, **fields}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("system speech failed")  # no text/key/native stderr in user error
    return dict(json.loads(result.stdout)) if result.stdout.strip() else {}


def voices() -> list[dict[str, Any]]:
    if sys.platform == "win32":
        return list(windows("voices")["voices"])
    if sys.platform != "darwin":
        return []
    result = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, timeout=10, check=True
    )
    found = []
    for line in result.stdout.splitlines():
        match = re.match(r"(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if match:
            found.append(
                {"id": match[1].strip(), "name": match[1].strip(), "languages": [match[2][:2]]}
            )
    return found[:128]


def synthesize(payload: dict[str, Any]) -> tuple[bytes, int]:
    selected = payload.get("voice", "")
    if selected and selected not in {v["id"] for v in voices()}:
        raise ValueError("voice is not installed")
    path = Path(payload["path"])
    if sys.platform == "darwin":
        command = [
            "say",
            "-o",
            str(path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            "-f",
            "-",
        ]
        if selected:
            command += ["-v", selected]
        subprocess.run(
            command,
            input=payload["text"],
            text=True,
            capture_output=True,
            timeout=40,
            check=True,
        )
    elif sys.platform == "win32":
        windows("synthesize", voice=selected, text=payload["text"], path=str(path))
    else:
        raise NotImplementedError("system speech is unavailable")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getsampwidth() != 2
            or audio.getnchannels() != 1
            or audio.getnframes() > audio.getframerate() * 120
        ):
            raise ValueError("invalid system audio")
        return audio.readframes(audio.getnframes()), audio.getframerate()


def main() -> None:
    pcm, rate = b"", 16000
    while line := sys.stdin.buffer.readline(MAX_LINE + 1):
        if len(line) > MAX_LINE:
            raise ValueError("speech RPC exceeds limit")
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            method, payload = request["method"], request["payload"]
            result: dict[str, Any]
            if method == "voices":
                result = {"voices": voices()}
            elif method == "synthesize":
                pcm, rate = synthesize(payload)
                result = {"size": len(pcm), "rate": rate}
            elif method == "read":
                offset = payload["offset"]
                if not isinstance(offset, int) or not 0 <= offset <= len(pcm):
                    raise ValueError("invalid audio offset")
                result = {"pcm": base64.b64encode(pcm[offset : offset + 16384]).decode()}
            else:
                raise ValueError("unknown speech operation")
            response = {"id": request["id"], "result": result}
        except Exception as error:
            response = {
                "id": request.get("id"),
                "error": {
                    "type": type(error).__name__,
                    "message": "system speech operation failed",
                },
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
