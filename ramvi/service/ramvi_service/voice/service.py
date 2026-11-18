"""5.2 VOI-01: запись → локальное распознавание → текст для поля ввода.

Состояния: `idle` → `recording` (после `start`) → `transcribing` (после `stop`) → `idle`.
`stop` возвращает текст; в чат он не отправляется — человек правит его в поле ввода.
Звук живёт только в памяти до конца распознавания (VOI-02).
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ramvi_service.chat import ChatError
from ramvi_service.downloads import Download, DownloadError
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
from ramvi_service.voice.stt import Transcriber, WhisperProcess

MIN_SECONDS = 0.3


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
        max_seconds: float = 120.0,
        idle_seconds: float = 300.0,
    ) -> None:
        self.models_dir, self.settings = models_dir, settings
        self.microphone = microphone or SoundDeviceMicrophone()
        self.make_transcriber = make_transcriber
        self.max_seconds, self.idle_seconds = max_seconds, idle_seconds
        self._lock = threading.Lock()
        self._state = "idle"
        self._recorder: Recorder | None = None
        self._limit = False
        self._transcriber: Transcriber | None = None
        self._transcriber_path: Path | None = None
        self._idle_timer: threading.Timer | None = None
        self._download: dict[str, Any] = {"state": "idle"}
        self._downloader: Download | None = None
        self._download_cancelled = False

    # --- модель ------------------------------------------------------------------

    def model(self) -> SttModel:
        return MODELS.get(self.settings().voice_model) or MODELS[DEFAULT_MODEL]

    def _language(self) -> str:
        language = self.settings().voice_language
        return language if language in LANGUAGES else "auto"

    def _model_path(self) -> Path:
        return self.model().path(self.models_dir())

    def _get_transcriber(self) -> Transcriber:
        """Процесс распознавания для текущей модели; смена модели — новый процесс."""
        path = self._model_path()
        if self._transcriber is not None and self._transcriber_path != path:
            self._transcriber.close()
            self._transcriber = None
        if self._transcriber is None:
            self._transcriber, self._transcriber_path = self.make_transcriber(path), path
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
            }
        return status

    def devices(self) -> dict[str, Any]:
        return {
            "devices": [{"name": d.name, "default": d.default} for d in self.microphone.devices()]
        }

    def start(self, device: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._state != "idle":
                raise ChatError("voice.busy", state=self._state)
            if not installed(self.model(), self.models_dir()):
                raise ChatError("voice.model_missing", model=self.model().id)
            name = device if device is not None else self.settings().voice_device
            self._limit = False
            self._recorder = Recorder(
                self.microphone, name or None, self.max_seconds, on_limit=self._on_limit
            )
            self._state = "recording"
            self._cancel_idle()
            transcriber = self._get_transcriber()
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

    def stop(self) -> dict[str, Any]:
        """Остановить запись и распознать; ответ — текст для поля ввода."""
        pcm, rate, loudest, seconds = self._take()
        try:
            if seconds < MIN_SECONDS:
                raise ChatError("voice.too_short")
            if loudest == 0:
                # Тишина из нулей: так macOS отдаёт звук без разрешения на микрофон.
                raise ChatError("voice.no_signal")
            with self._lock:
                transcriber = self._get_transcriber()
            started = time.monotonic()
            result = transcriber.transcribe(pcm, rate, self._language())
            if not str(result.get("text", "")).strip():
                raise ChatError("voice.no_speech")
            return {
                "text": str(result["text"]).strip(),
                "language": result.get("language"),
                "audio_s": round(seconds, 2),
                "proc_s": round(time.monotonic() - started, 2),
            }
        finally:
            del pcm
            with self._lock:
                self._state = "idle"
                self._limit = False
                self._schedule_idle()

    def cancel(self) -> dict[str, Any]:
        """Прекратить запись без распознавания; звук забывается."""
        with self._lock:
            recorder, self._recorder = self._recorder, None
            if self._state == "recording":
                self._state = "idle"
                self._limit = False
        if recorder is not None:
            recorder.finish()
        return self.status()

    # --- скачивание модели -------------------------------------------------------

    def download_model(self, model_id: str | None = None) -> dict[str, Any]:
        model = MODELS.get(model_id or self.model().id)
        if model is None:
            raise ChatError("voice.unknown_model", model=str(model_id))
        with self._lock:
            if self._download.get("state") == "running":
                raise ChatError("voice.download_running")
            downloads = [
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
                    (model, lambda p: self._download.update(done=p.done, total=p.total)),
                )
            ]
            self._download = {"state": "running", "model": model.id, "done": 0, "total": model.size}
            self._download_cancelled = False
        threading.Thread(target=self._run_download, args=(downloads,), daemon=True).start()
        return self.status()

    def _run_download(self, downloads: list[Download]) -> None:
        update: dict[str, Any] = {"state": "done"}
        for download in downloads:
            with self._lock:
                self._downloader = download
                if self._download_cancelled:
                    update = {"state": "idle"}  # отменено между файлами
                    break
            try:
                download.run()
            except DownloadError as error:
                update = {"state": "failed", "error": {"code": error.code, "params": error.params}}
                if error.code == "download.cancelled":
                    download.discard()
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
