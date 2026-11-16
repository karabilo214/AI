"""3.4: процесс сервиса для оболочки: `python -m ramvi_service.app --data-dir <папка>`.

Ключ интерфейса приходит через окружение `RAMVI_UI_TOKEN` (не в argv). После
запуска в stdout печатается одна строка `{"event": "ready", "port": N}`.
Сервис следит за stdin, как движок (3.3): закрылась оболочка — сервис
останавливает движок и выходит, не оставляя процессов.
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from ramvi_engine.processes import ParentWatch
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core

from ramvi_service.chat import ChatService
from ramvi_service.contracts.modules import Authority
from ramvi_service.engines import (
    EngineManager,
    EngineTarget,
    SwitchableProvider,
    default_engine,
)
from ramvi_service.resources import SharedLLM
from ramvi_service.ui_api import UIServer

TOKEN_ENV = "RAMVI_UI_TOKEN"


def build(
    data_dir: Path, token: str, *, port: int = 0, engine_command: list[str] | None = None
) -> tuple[UIServer, EngineManager]:
    """Собрать сервис; `engine_command` — только для проверок без модели."""
    db = open_core(data_dir)
    make_engine = default_engine
    if engine_command is not None:
        command = list(engine_command)

        def make_engine(model: Model) -> EngineTarget:
            from ramvi_engine.engine_process import EngineProcess
            from ramvi_engine.model_loader import ModelConfig

            return EngineProcess(ModelConfig(model.path), command=command, startup_timeout=30)

    engines = EngineManager(Models(db), data_dir / "run", make_engine=make_engine)
    authority = Authority()
    llm = SharedLLM(authority, SwitchableProvider(engines))
    chat = ChatService(db, llm, default_model=lambda: engines.model_id)
    return UIServer(chat, token, port=port, engine_status=engines.status), engines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ramvi-service")
    parser.add_argument("--data-dir", required=True, help="папка данных приложения")
    parser.add_argument("--port", type=int, default=0, help="порт API интерфейса (0 — любой)")
    parser.add_argument(
        "--engine-command-json",
        help="для проверок: JSON-список команды поддельного движка вместо настоящего",
    )
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 16:
        parser.error(f"нужен {TOKEN_ENV} длиной от 16 символов")
    command = json.loads(args.engine_command_json) if args.engine_command_json else None
    server, engines = build(Path(args.data_dir), token, port=args.port, engine_command=command)
    stopped = threading.Event()
    watch = ParentWatch.from_env()
    if watch is not None:
        watch.set_shutdown(stopped.set)
    server.start()
    print(json.dumps({"event": "ready", "port": server.port}), flush=True)
    try:
        stopped.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        engines.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
