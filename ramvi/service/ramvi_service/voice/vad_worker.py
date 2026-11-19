"""Small VAD-only worker using the shared-memory audio RPC contract."""

import argparse
import contextlib
import json
import sys

from ramvi_service.voice.vad import Silero
from ramvi_service.voice.worker import MAX_LINE, read_shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads")
    parser.add_argument("--vad")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        detector = Silero(args.model)
    try:
        while line := sys.stdin.buffer.readline(MAX_LINE + 1):
            if len(line) > MAX_LINE:
                raise ValueError("VAD RPC exceeds limit")
            request = json.loads(line)
            try:
                method, payload = request["method"], request["payload"]
                result = {}
                if method == "transcribe":
                    pcm = read_shared(payload["shm"], payload["size"])
                    with contextlib.redirect_stdout(sys.stderr):
                        result = {"speech": detector.speech(pcm, payload["rate"])}
                    del pcm
                elif method != "ping":
                    raise ValueError("unknown VAD method")
                response = {"id": request["id"], "result": result}
            except Exception as error:
                response = {
                    "id": request["id"],
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            print(json.dumps(response), flush=True)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
