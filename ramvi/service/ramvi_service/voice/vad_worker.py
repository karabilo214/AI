"""Small VAD-only worker using the shared-memory audio RPC contract."""

import argparse
import contextlib
import json
import sys
from collections.abc import Callable

from ramvi_service.i18n import translator, with_defaults
from ramvi_service.voice.vad import Silero
from ramvi_service.voice.worker import MAX_LINE, read_shared


def build_parser(t: Callable[..., str]) -> argparse.ArgumentParser:
    """B.1: внутренний процесс поиска речи (запускает сервис); --threads и --vad — для
    совместимости с командой распознавания, не используются."""
    parser = argparse.ArgumentParser(prog="ramvi-vad-worker", description=t("cli.vad.about"))
    parser.add_argument("--model", required=True, help=t("cli.vad.model"))
    parser.add_argument("--threads", help=t("cli.vad.unused"))
    parser.add_argument("--vad", help=t("cli.vad.unused"))
    return with_defaults(parser, t)


def main() -> None:
    parser = build_parser(translator())
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
