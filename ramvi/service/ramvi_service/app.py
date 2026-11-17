"""3.4: процесс сервиса для оболочки: `python -m ramvi_service.app --data-dir <папка>`.

Ключ интерфейса приходит через окружение `RAMVI_UI_TOKEN` (не в argv). После
запуска в stdout печатается одна строка `{"event": "ready", "port": N}`, а если
папка данных недоступна — `{"event": "error", "code": "storage.no_access", ...}`
и выход с кодом 3 (3.5).
Сервис следит за stdin, как движок (3.3): закрылась оболочка — сервис
останавливает движок и выходит, не оставляя процессов.
"""

import argparse
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ramvi_engine.processes import ParentWatch
from ramvi_storage.repositories import Model, Models, NotFound
from ramvi_storage.schema import open_core

from ramvi_service import settings as app_settings
from ramvi_service.app_config import AppConfig, check_folder
from ramvi_service.catalog import Catalog
from ramvi_service.chat import ChatService
from ramvi_service.contracts.modules import Authority
from ramvi_service.control_run import ControlRun
from ramvi_service.download_manager import DownloadManager, register_downloaded
from ramvi_service.engines import (
    EngineManager,
    EngineTarget,
    SwitchableProvider,
    default_engine,
)
from ramvi_service.guard import Guard
from ramvi_service.hardware import collect
from ramvi_service.i18n import load_catalog, render
from ramvi_service.model_control import ModelControl
from ramvi_service.model_fit import ModelFit
from ramvi_service.modules.host import ModuleHost
from ramvi_service.onboarding import Onboarding
from ramvi_service.resources import SharedLLM
from ramvi_service.sections import ModuleDataApi, SectionRegistry
from ramvi_service.supervisor import RestartPolicy
from ramvi_service.task_profile import TaskProfile
from ramvi_service.ui_api import UIServer

TOKEN_ENV = "RAMVI_UI_TOKEN"


def build(
    data_dir: Path,
    token: str,
    *,
    port: int = 0,
    engine_command: list[str] | None = None,
    restart_policy: Callable[[], RestartPolicy] = RestartPolicy,
    app_dir: Path | None = None,
    catalog: Catalog | None = None,
    download_options: dict[str, Any] | None = None,
    auto_control: bool | None = None,
) -> tuple[UIServer, EngineManager]:
    """Собрать сервис; `engine_command` — только для проверок без модели."""
    db = open_core(data_dir)
    make_engine = default_engine
    if engine_command is not None:
        command = list(engine_command)

        def make_engine(
            model: Model, draft: Model | None = None, maxt: int = 2048, resident: int | None = None
        ) -> EngineTarget:
            from ramvi_engine.engine_process import EngineProcess
            from ramvi_engine.model_loader import ModelConfig

            extra = ["--maxt", str(maxt)] + ([] if draft is None else ["--draft", draft.path])
            if resident is not None:
                extra += ["--resident", str(resident)]
            return EngineProcess(
                ModelConfig(model.path),
                command=[*command, "--model", model.path, *extra],
                startup_timeout=30,
            )

    root = None if app_dir is None else Path(app_dir).resolve()

    def task() -> TaskProfile:
        """4.5: профиль задачи из настроек; испорченный файл — профиль по умолчанию."""
        try:
            return app_settings.load(root or data_dir).task
        except (OSError, ValueError):
            return TaskProfile()

    engines = EngineManager(
        Models(db),
        data_dir / "run",
        make_engine=make_engine,
        policy=restart_policy,
        model_root=root,
        context=lambda: task().effective_context,
    )
    catalog = Catalog(data_dir) if catalog is None else catalog
    authority = Authority()
    llm = SharedLLM(authority, SwitchableProvider(engines))
    onboarding = Onboarding(data_dir, Models(db), app_dir=root)
    chat = ChatService(
        db, llm, default_model=lambda: engines.model_id or onboarding.default_model()
    )
    control = ModelControl(engines, llm, chat.stop_all)
    host = ModuleHost(data_dir, llm=llm)
    sections = SectionRegistry(host)
    if root is not None:
        sections.discover(root / "modules")  # процессы модулей не запускаются (3.11)
    server = UIServer(
        chat,
        token,
        port=port,
        model_control=control,
        onboarding=onboarding,
        config=None if app_dir is None else AppConfig(app_dir, data_dir),
        sections=sections,
        module_data=ModuleDataApi(sections, data_dir / "exports"),
        catalog=catalog,
    )

    def hardware() -> Any:  # диск — папка моделей, куда пойдут скачанные файлы
        folder = onboarding.models_folder()
        return collect(folder if folder.is_dir() else data_dir)

    fit = ModelFit(catalog.headers, hardware, profile=task)
    runs = ControlRun(
        engines, llm, Models(db), fit.hardware_profile, lambda: task().effective_context
    )

    def local_result(model_id: str, hw: Any, context: int) -> Any:
        try:
            return runs.latest(Models(db).get(model_id), hw, context)
        except NotFound:
            return None

    fit.local_result = local_result
    engines.resident_for = runs.resident_for  # безопасный план после отката (4.7)
    guard = Guard(fit, Models(db), root, task)
    engines.preflight = guard.preflight  # невыполнимый план — сразу понятная ошибка (4.8)

    def after_download(model_id: str) -> None:
        nvidia = any(g.vendor == "NVIDIA" for g in fit.hardware_profile().gpus)
        if auto_control if auto_control is not None else nvidia:
            runs.start(model_id)

    server.downloads = DownloadManager(
        db,
        catalog,
        onboarding.models_folder,
        register_downloaded(onboarding.import_model, Models(db), catalog),
        on_installed=after_download,
        **(download_options or {}),
    )
    server.fit, server.control, server.guard = fit, runs, guard
    server.app_dir = root
    # Закрывается в обратном порядке: сначала загрузки, потом база
    # (иначе файл базы открыт и после close — Windows не даст перенести папку).
    server.on_close.append(db.close)
    server.on_close.append(server.downloads.close)
    return server, engines


def check_data_dir(data_dir: Path) -> None:
    """Папка данных существует (создаётся) и доступна на запись; иначе OSError."""
    check_folder(data_dir)


def main(argv: list[str] | None = None) -> None:
    texts = load_catalog(os.environ.get("RAMVI_LANGUAGE", "ru"))  # 3.14: справка на языке

    def t(key: str, **params: Any) -> str:
        return render(key, texts, **params)

    parser = argparse.ArgumentParser(prog="ramvi-service", description=t("cli.service.about"))
    parser.add_argument("--data-dir", required=True, help=t("cli.service.data_dir"))
    parser.add_argument("--port", type=int, default=0, help=t("cli.service.port"))
    parser.add_argument("--app-dir", help=t("cli.service.app_dir"))
    parser.add_argument("--engine-command-json", help=t("cli.service.engine_command_json"))
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 16:
        parser.error(t("cli.service.need_token", name=TOKEN_ENV))
    command = json.loads(args.engine_command_json) if args.engine_command_json else None
    data_dir = Path(args.data_dir)
    try:
        check_data_dir(data_dir)
        server, engines = build(
            data_dir,
            token,
            port=args.port,
            engine_command=command,
            app_dir=Path(args.app_dir) if args.app_dir else None,
        )
    except (OSError, sqlite3.Error) as error:
        # 3.5 (UI-03): оболочка показывает «нет доступа к каталогу» с действиями.
        print(
            json.dumps(
                {
                    "event": "error",
                    "code": "storage.no_access",
                    "params": {"path": str(data_dir.resolve()), "detail": str(error)},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        sys.exit(3)
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
        if server.sections is not None:
            server.sections.host.close()  # остановить процессы открытых модулей
        engines.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
