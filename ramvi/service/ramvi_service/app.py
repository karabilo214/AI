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
import secrets
import sqlite3
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ramvi_engine.processes import ParentWatch
from ramvi_storage.repositories import AgentRuns, Model, Models, NotFound
from ramvi_storage.schema import open_core

from ramvi_service import driver_gate
from ramvi_service import settings as app_settings
from ramvi_service.agent.desktop import AgentDesktop
from ramvi_service.agent.gateway import AgentGateway
from ramvi_service.agent.readiness import facts_from_fit
from ramvi_service.api_server import ApiOptions, ApiServer
from ramvi_service.app_config import AppConfig, check_folder
from ramvi_service.catalog import Catalog
from ramvi_service.catalog.support import capabilities
from ramvi_service.chat import ChatService
from ramvi_service.client_keys import KeyStore
from ramvi_service.cloud import CloudConnections, CloudProvider, is_cloud
from ramvi_service.context import ContextProvider
from ramvi_service.contracts.modules import Authority
from ramvi_service.control_run import ControlRun
from ramvi_service.download_manager import DownloadManager, register_downloaded
from ramvi_service.engines import (
    EngineManager,
    EngineTarget,
    SwitchableProvider,
    default_engine,
    resolve_model,
)
from ramvi_service.guard import Guard
from ramvi_service.hardware import collect
from ramvi_service.i18n import load_catalog, render
from ramvi_service.model_control import ModelControl
from ramvi_service.model_fit import ModelFit
from ramvi_service.model_library import ModelLibrary
from ramvi_service.modules.host import ModuleHost
from ramvi_service.modules.packages import ModulePackages
from ramvi_service.onboarding import Onboarding
from ramvi_service.resources import SharedLLM
from ramvi_service.secrets_store import os_secrets
from ramvi_service.sections import ModuleDataApi, SectionRegistry
from ramvi_service.supervisor import RestartPolicy
from ramvi_service.system_context import SystemContextAdapter
from ramvi_service.task_profile import TaskProfile
from ramvi_service.ui_api import UIServer
from ramvi_service.updates import Updater
from ramvi_service.voice.service import VoiceService

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
    voice_options: dict[str, Any] | None = None,
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
    shared_secrets = (voice_options or {}).get("secrets") or os_secrets(data_dir)
    cloud = CloudConnections(data_dir, shared_secrets, Models(db))

    def cloud_for(model_id: str) -> CloudProvider | None:
        """7.7: облачная модель идёт в свой сервис, локальный движок не нужен."""
        try:
            model = Models(db).get(model_id)
        except NotFound:
            return None
        return cloud.provider(model) if is_cloud(model) else None

    llm = SharedLLM(authority, SwitchableProvider(engines, cloud_for))
    onboarding = Onboarding(data_dir, Models(db), app_dir=root)
    chat = ChatService(
        db, llm, default_model=lambda: engines.model_id or onboarding.default_model()
    )
    control = ModelControl(engines, llm, chat.stop_all)
    host = ModuleHost(data_dir, llm=llm, secrets=shared_secrets)
    host.agent = AgentGateway(host.authority, host.grants)
    host.context = ContextProvider(host.authority, host.grants, SystemContextAdapter())
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
    if engine_command is None:  # 8.7: настоящий движок — сначала драйвер NVIDIA с нужной CUDA
        engines.preflight = lambda model: driver_gate.check() or guard.preflight(model)

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
    server.library = ModelLibrary(
        Models(db),
        onboarding.remember,
        onboarding.models_folder,
        root,
        lambda: engines.model_id,
        control.unload,
        lambda model_id: runs.status(model_id)["state"] in ("queued", "running"),
    )
    server.app_dir = root
    server.cloud = cloud  # 7.7: подключения к облачным сервисам

    def current_settings() -> app_settings.Settings:
        try:
            return app_settings.load(root or data_dir)
        except (OSError, ValueError):
            return app_settings.Settings()

    # 5.6: микрофоном владеет держатель аренды MICROPHONE в общем ResourceBroker хоста модулей;
    # интерфейс RAMVI — свой владелец, модули — свои (Scoped Services voice.*).
    voice_owner = host.authority.issue("core", "local_user", "voice")
    # Модели распознавания — в папке приложения, а не в папке моделей LLM: та может указывать
    # в чужое хранилище (например, blobs Ollama), куда нельзя класть свои подпапки.
    stt_root = (root or data_dir) / "models"
    if root is not None:  # 8.8: установка, обновление, отключение и удаление пакетов модулей
        server.packages = ModulePackages(root / "modules", sections)
    server.voice = VoiceService(
        lambda: stt_root,
        current_settings,
        resources=host.resources,
        owner=voice_owner,
        # 5.7: ключ стороннего API — в хранилище ОС (в проверках подставляется своё)
        **{"secrets": shared_secrets, **(voice_options or {})},
    )
    host.voice = server.voice
    from ramvi_service.speech.service import SpeechService

    host.speech = SpeechService(host.resources, server.voice)
    server.agent_ui = AgentDesktop(
        data_dir,
        AgentRuns(db),
        host,
        sections,
        llm,
        facts=lambda m, context: facts_from_fit(
            fit.for_file(
                resolve_model(m, root).path, m.id, task=TaskProfile("agent", context=context)
            ),
            capabilities("gguf", m.architecture),
        ),
    )

    def stop_all_generations() -> int:
        count = chat.stop_all()
        assert server.agent_ui is not None
        server.agent_ui.close()
        return count

    control._stop_generations = stop_all_generations
    # 7.1: внешний API — общий ключевой реестр и общая очередь модели; порт из настроек.
    server.api = ApiServer(
        KeyStore(db),
        llm,
        engines,
        data_dir / "run",
        ApiOptions.from_settings(current_settings()),
    )
    server.api.host = host
    server.api.cloud, server.api.models = cloud, Models(db)  # 7.9: транслятор облачных моделей
    if root is not None:
        server.updater = Updater(root, data_dir)  # 8.3: обновления комплекта
    # Закрывается в обратном порядке: сначала загрузки, потом база
    # (иначе файл базы открыт и после close — Windows не даст перенести папку).
    server.on_close.append(db.close)
    server.on_close.append(server.api.close)
    server.on_close.append(cloud.budget.close)  # 7.10: после внешнего API — он пишет учёт
    server.on_close.append(server.downloads.close)
    server.on_close.append(host.agent.close)
    server.on_close.append(host.close)
    server.on_close.append(server.agent_ui.close)
    server.on_close.append(host.speech.close)
    server.on_close.append(server.voice.close)  # первым: запись и процесс распознавания
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
    parser.add_argument("--headless", action="store_true", help=t("cli.service.headless"))
    parser.add_argument("--model", help=t("cli.service.model"))
    parser.add_argument("--api-port", type=int, help=t("cli.service.api_port"))
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV, "")
    if args.headless and not token:
        token = secrets.token_urlsafe(32)  # без окна ключ интерфейса никому не нужен
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
    api = server.api
    if args.api_port is not None:
        api.options = replace(api.options, port=args.api_port)
    try:
        api.start()
        api_url: str | None = api.url
    except OSError as error:  # порт занят — интерфейс работает, API сообщает об ошибке
        api_url = None
        print(
            json.dumps(
                {
                    "event": "api_error",
                    "code": "api.port_busy",
                    "params": {"port": api.options.port, "detail": str(error)},
                }
            ),
            flush=True,
        )
    print(json.dumps({"event": "ready", "port": server.port, "api": api_url}), flush=True)
    if args.model:
        threading.Thread(
            target=lambda: engines.ensure(args.model), name="ramvi-load", daemon=True
        ).start()
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
