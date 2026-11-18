"""5.2 VOI-01: запись → локальное распознавание → текст для поля ввода.

Состояния: `idle` → `recording` (после `start`) → `transcribing` (после `stop`) → `idle`.
`stop` возвращает текст; в чат он не отправляется — человек правит его в поле ввода.
Звук живёт только в памяти до конца распознавания (VOI-02).

5.4 VOI-03: при каждой записи выбирается, где распознавать. Видеокарта — если скачан
ускоритель (`accel.py`) и свободной видеопамяти хватает с запасом; процесс на ней живёт
одну запись и сразу освобождает видеопамять. Иначе процессор — если хватает оперативной
памяти; не хватает — запасная модель small, не хватает и ей — понятная ошибка: голос
не отнимает память у активной LLM.
"""

import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ramvi_service import hardware
from ramvi_service.chat import ChatError
from ramvi_service.contracts.modules import Owner
from ramvi_service.contracts.voice import MICROPHONE, SpeechCapabilities
from ramvi_service.downloads import Download, DownloadError
from ramvi_service.voice import accel
from ramvi_service.voice.audio import Microphone, Recorder, SoundDeviceMicrophone
from ramvi_service.voice.models import (
    DEFAULT_MODEL,
    FOLDER,
    LANGUAGES,
    MODELS,
    VAD,
    SttModel,
    installed,
)
from ramvi_service.voice.server import WhisperServer
from ramvi_service.voice.stt import Transcriber, WhisperProcess, default_threads

MIN_SECONDS = 0.3
FALLBACK_MODEL = "small-q5_1"
# Оперативная память процесса распознавания на CPU (пик, замер Mac M1: turbo 907 МБ,
# small 541 МБ) плюс запас, чтобы не теснить активную LLM.
RAM_NEED_MB = {"large-v3-turbo-q5_0": 907 + 512, "small-q5_1": 541 + 512}
# whisper-server на видеокарте тоже держит оперативную память (пик 427 МБ, ASUS, turbo).
GPU_RAM_NEED_MB = 427 + 256


def _memory_now() -> tuple[int | None, int | None]:
    return hardware.ram_available_mb(), hardware.nvidia_free_mb()


def _gpu_transcriber(model: Path, folder: Path) -> Transcriber:
    return WhisperServer(model, folder, threads=default_threads())


class VoiceSettings(Protocol):
    @property
    def voice_device(self) -> str: ...

    @property
    def voice_language(self) -> str: ...

    @property
    def voice_model(self) -> str: ...


class VoiceService:
    def __init__(
        self,
        models_dir: Callable[[], Path],
        settings: Callable[[], VoiceSettings],
        *,
        microphone: Microphone | None = None,
        make_transcriber: Callable[[Path], Transcriber] = WhisperProcess,
        make_gpu_transcriber: Callable[[Path, Path], Transcriber] = _gpu_transcriber,
        memory: Callable[[], tuple[int | None, int | None]] = _memory_now,
        has_nvidia: Callable[[], bool] = lambda: shutil.which("nvidia-smi") is not None,
        platform: str = sys.platform,
        resources: Any = None,
        owner: Owner | None = None,
        max_seconds: float = 120.0,
        idle_seconds: float = 300.0,
    ) -> None:
        self.models_dir, self.settings = models_dir, settings
        self.microphone = microphone or SoundDeviceMicrophone()
        self.make_transcriber = make_transcriber
        self.make_gpu_transcriber, self.memory = make_gpu_transcriber, memory
        self._has_nvidia_probe = has_nvidia
        self._has_nvidia: bool | None = None
        self.platform = platform
        # 5.6: микрофоном владеет тот, у кого аренда MICROPHONE в общем ResourceBroker.
        self.resources, self.owner = resources, owner
        self._lease: tuple[Owner, str] | None = None
        self._recording_owner: Owner | None = None
        self._cancelled = False
        self._active: tuple[str, SttModel] | None = None  # где распознаётся текущая запись
        self._last: dict[str, Any] | None = None
        self.max_seconds, self.idle_seconds = max_seconds, idle_seconds
        self._lock = threading.Lock()
        self._state = "idle"
        self._recorder: Recorder | None = None
        self._limit = False
        self._transcriber: Transcriber | None = None
        self._transcriber_key: tuple[str, Path] | None = None
        self._idle_timer: threading.Timer | None = None
        self._download: dict[str, Any] = {"state": "idle"}
        self._downloader: Any = None
        self._download_cancelled = False

    # --- модель ------------------------------------------------------------------

    def model(self) -> SttModel:
        return MODELS.get(self.settings().voice_model) or MODELS[DEFAULT_MODEL]

    def _language(self) -> str:
        language = self.settings().voice_language
        return language if language in LANGUAGES else "auto"

    def _model_path(self) -> Path:
        return self.model().path(self.models_dir())

    def has_nvidia(self) -> bool:
        if self._has_nvidia is None:
            self._has_nvidia = bool(self._has_nvidia_probe())
        return self._has_nvidia

    def accel_offered(self) -> bool:
        return accel.offered(self.platform, self.has_nvidia())

    def _choose(self) -> tuple[str, SttModel]:
        """Где распознавать эту запись: («gpu» | «cpu», модель)."""
        model, models = self.model(), self.models_dir()
        if not installed(model, models):
            raise ChatError("voice.model_missing", model=model.id)
        ram, vram = self.memory()
        if (
            self.accel_offered()
            and accel.CUDA.installed(models)
            and vram is not None
            and vram >= accel.VRAM_NEED_MB[model.id]
            and (ram is None or ram >= GPU_RAM_NEED_MB)
        ):
            return "gpu", model
        need = RAM_NEED_MB[model.id]
        if ram is None or ram >= need:
            return "cpu", model
        small = MODELS[FALLBACK_MODEL]
        if model.id != small.id and installed(small, models) and ram >= RAM_NEED_MB[small.id]:
            return "cpu", small  # турбо не помещается — запасная модель
        raise ChatError("voice.no_memory", need_mb=need, free_mb=ram)

    def _get_transcriber(self, backend: str, model: SttModel) -> Transcriber:
        """Процесс распознавания для способа и модели; смена — новый процесс."""
        path = model.path(self.models_dir())
        key = (backend, path)
        if self._transcriber is not None and self._transcriber_key != key:
            self._transcriber.close()
            self._transcriber = None
        if self._transcriber is None:
            self._transcriber = (
                self.make_gpu_transcriber(path, accel.CUDA.path(self.models_dir()))
                if backend == "gpu"
                else self.make_transcriber(path)
            )
            self._transcriber_key = key
        return self._transcriber

    def _cancel_idle(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle(self) -> None:
        """Модель распознавания не держится в памяти без дела (по умолчанию 5 минут)."""
        self._cancel_idle()
        timer = threading.Timer(self.idle_seconds, self._unload_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _unload_idle(self) -> None:
        with self._lock:
            if self._state == "idle" and self._transcriber is not None:
                self._transcriber.close()

    # --- состояние ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        model = self.model()
        settings = self.settings()
        with self._lock:
            recorder = self._recorder
            status: dict[str, Any] = {
                "state": self._state,
                "seconds": round(recorder.seconds, 1) if recorder else 0.0,
                "level": round(recorder.level, 3) if recorder else 0.0,
                "limit_reached": self._limit,
                "max_seconds": self.max_seconds,
                "language": self._language(),
                "languages": list(LANGUAGES),
                "device": settings.voice_device,
                "model": {
                    "id": model.id,
                    "installed": installed(model, self.models_dir()),
                    "size": model.size,
                    "loaded": bool(self._transcriber and self._transcriber.running),
                },
                "models": [
                    {"id": m.id, "size": m.size, "installed": installed(m, self.models_dir())}
                    for m in MODELS.values()
                ],
                "download": dict(self._download),
                "backend": self._active[0] if self._active else None,
                "active_model": self._active[1].id if self._active else None,
                "last": dict(self._last) if self._last else None,
                "accel": {
                    "id": accel.CUDA.id,
                    "offered": self.accel_offered(),
                    "installed": accel.CUDA.installed(self.models_dir()),
                    "size": accel.CUDA.size,
                },
            }
        return status

    def devices(self) -> dict[str, Any]:
        return {
            "devices": [{"name": d.name, "default": d.default} for d in self.microphone.devices()]
        }

    def capabilities(self) -> SpeechCapabilities:
        """5.6: что умеет распознавание сейчас (для чата и модулей)."""
        gpu = self.accel_offered() and accel.CUDA.installed(self.models_dir())
        return SpeechCapabilities(
            languages=LANGUAGES,
            backends=("cpu", "gpu") if gpu else ("cpu",),
            max_seconds=self.max_seconds,
            extra={
                "model": self.model().id,
                "installed": installed(self.model(), self.models_dir()),
            },
        )

    def _acquire_microphone(self, owner: Owner | None) -> None:
        who = owner or self.owner
        if self.resources is None or who is None:
            return
        try:
            lease = self.resources.acquire(who, {MICROPHONE: 1}, timeout=0.1)
        except TimeoutError:
            raise ChatError("voice.mic_busy") from None  # микрофон у другой записи
        self._lease = (who, lease)

    def _release_microphone(self) -> None:
        if self._lease is not None and self.resources is not None:
            self.resources.release(*self._lease)
        self._lease = None

    def start(self, device: str | None = None, owner: Owner | None = None) -> dict[str, Any]:
        """Начать запись. `owner` — модуль (Scoped Services); None — интерфейс RAMVI."""
        with self._lock:
            if self._state != "idle":
                raise ChatError("voice.busy", state=self._state)
            backend, model = self._choose()
            name = device if device is not None else self.settings().voice_device
            self._acquire_microphone(owner)
            self._limit = False
            try:
                self._recorder = Recorder(
                    self.microphone, name or None, self.max_seconds, on_limit=self._on_limit
                )
            except BaseException:
                self._release_microphone()
                raise
            self._recording_owner = owner
            self._cancelled = False
            self._state = "recording"
            self._active = (backend, model)
            self._cancel_idle()
            transcriber = self._get_transcriber(backend, model)
        threading.Thread(target=self._prepare, args=(transcriber,), daemon=True).start()
        return self.status()

    def _prepare(self, transcriber: Transcriber) -> None:
        try:
            transcriber.prepare()  # модель грузится, пока человек говорит
        except Exception:
            pass  # ошибка повторится и будет показана при распознавании

    def _on_limit(self) -> None:
        with self._lock:
            self._limit = True

    def _take(self) -> tuple[bytes, int, int, float]:
        with self._lock:
            if self._state != "recording" or self._recorder is None:
                raise ChatError("voice.not_recording")
            recorder, self._recorder = self._recorder, None
            self._state = "transcribing"
        seconds = recorder.seconds
        pcm, rate, loudest = recorder.finish()
        return pcm, rate, loudest, seconds

    def _check_owner(self, owner: Owner | None) -> None:
        """Модуль останавливает только свою запись; интерфейс (None) — любую (общий Stop)."""
        if owner is not None and self._recording_owner != owner:
            raise ChatError("voice.not_owner")

    def stop(self, owner: Owner | None = None) -> dict[str, Any]:
        """Остановить запись и распознать; ответ — текст для поля ввода."""
        with self._lock:
            self._check_owner(owner)
        pcm, rate, loudest, seconds = self._take()
        try:
            if seconds < MIN_SECONDS:
                raise ChatError("voice.too_short")
            if loudest == 0:
                # Тишина из нулей: так macOS отдаёт звук без разрешения на микрофон.
                raise ChatError("voice.no_signal")
            with self._lock:
                backend, model = self._active or ("cpu", self.model())
                transcriber = self._get_transcriber(backend, model)
            started = time.monotonic()
            try:
                result = transcriber.transcribe(pcm, rate, self._language())
            except ChatError:
                if self._cancelled:
                    raise ChatError("voice.cancelled") from None
                raise
            finally:
                proc = round(time.monotonic() - started, 2)
                with self._lock:
                    self._last = {"backend": backend, "model": model.id, "proc_s": proc}
            if self._cancelled:
                raise ChatError("voice.cancelled")
            if not str(result.get("text", "")).strip():
                raise ChatError("voice.no_speech")
            return {
                "text": str(result["text"]).strip(),
                "language": result.get("language"),
                "audio_s": round(seconds, 2),
                "proc_s": proc,
                "backend": backend,
                "model": model.id,
            }
        finally:
            del pcm
            self._after_use()

    def _after_use(self) -> None:
        with self._lock:
            self._state = "idle"
            self._limit = False
            self._active = None
            self._recording_owner = None
            self._release_microphone()
            if self._transcriber is not None and getattr(self._transcriber, "single_use", False):
                self._transcriber.close()  # видеопамять — только на время записи
                self._transcriber = None
            self._schedule_idle()

    def cancel(self, owner: Owner | None = None) -> dict[str, Any]:
        """Прекратить запись без распознавания или прервать идущее распознавание."""
        recorder = transcriber = None
        cancelled = False
        with self._lock:
            if self._state != "idle":
                self._check_owner(owner)
                recorder, self._recorder = self._recorder, None
                cancelled = self._state == "recording"
                if self._state == "transcribing":
                    self._cancelled = True
                    transcriber = self._transcriber
        if recorder is not None:
            recorder.finish()
        if transcriber is not None:
            transcriber.close()  # процесс распознавания останавливается, stop вернёт «отменено»
            with self._lock:
                if self._transcriber is transcriber:
                    self._transcriber = None
        if cancelled:
            self._after_use()
        return self.status()

    # --- скачивание модели -------------------------------------------------------

    def download_model(self, model_id: str | None = None) -> dict[str, Any]:
        model = MODELS.get(model_id or self.model().id)
        if model is None:
            raise ChatError("voice.unknown_model", model=str(model_id))
        jobs = [
            Download(
                m.url,
                self.models_dir(),
                f"{FOLDER}/{m.file}",
                m.sha256,
                size=m.size,
                on_progress=progress,
            )
            for m, progress in (
                (VAD, None),  # детектор речи — первым, он маленький
                (model, self._progress),
            )
        ]
        return self._start_jobs(model.id, model.size, jobs)

    def download_accel(self) -> dict[str, Any]:
        """5.4: скачать и распаковать ускоритель распознавания на видеокарте."""
        if not self.accel_offered():
            raise ChatError("voice.accel_unsupported")
        pack, models = accel.CUDA, self.models_dir()
        download = Download(
            pack.url,
            models,
            pack.archive(models),
            pack.sha256,
            size=pack.size,
            on_progress=self._progress,
        )
        return self._start_jobs(
            pack.id, pack.size, [download, _Extract(pack, download.target, models)]
        )

    def _progress(self, progress: Any) -> None:
        self._download.update(done=progress.done, total=progress.total)

    def _start_jobs(self, what: str, size: int, jobs: list[Any]) -> dict[str, Any]:
        with self._lock:
            if self._download.get("state") == "running":
                raise ChatError("voice.download_running")
            self._download = {"state": "running", "model": what, "done": 0, "total": size}
            self._download_cancelled = False
        threading.Thread(target=self._run_download, args=(jobs,), daemon=True).start()
        return self.status()

    def _run_download(self, jobs: list[Any]) -> None:
        update: dict[str, Any] = {"state": "done"}
        for job in jobs:
            with self._lock:
                self._downloader = job
                if self._download_cancelled:
                    update = {"state": "idle"}  # отменено между шагами
                    break
            try:
                job.run()
            except DownloadError as error:
                update = {"state": "failed", "error": {"code": error.code, "params": error.params}}
                if error.code == "download.cancelled":
                    job.discard()
                    update = {"state": "idle"}
                break
            except OSError as error:
                update = {
                    "state": "failed",
                    "error": {"code": "download.failed", "params": {"detail": str(error)}},
                }
                break
        with self._lock:
            self._download.update(update)
            self._downloader = None

    def cancel_download(self) -> dict[str, Any]:
        with self._lock:
            self._download_cancelled = True
            if self._downloader is not None:
                self._downloader.cancel()
        return self.status()

    def close(self) -> None:
        self.cancel()
        self.cancel_download()
        with self._lock:
            self._cancel_idle()
            if self._transcriber is not None:
                self._transcriber.close()
                self._transcriber = None


class _Extract:
    """Шаг очереди скачивания: распаковать ускоритель после проверки хеша архива."""

    def __init__(self, pack: accel.AccelPack, archive: Path, models_dir: Path) -> None:
        self.pack, self.archive, self.models_dir = pack, archive, models_dir

    def run(self) -> None:
        accel.extract(self.pack, self.archive, self.models_dir)

    def cancel(self) -> None:
        pass  # распаковка короткая: дожидаемся её

    def discard(self) -> None:
        pass
