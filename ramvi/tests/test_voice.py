"""5.2 VOI-01: запись → распознавание → текст; без микрофона, whisper и numpy.

Микрофон и распознаватель подставные; процесс распознавания проверяется настоящим
дочерним процессом с подставной «моделью» (общая память и RPC те же).
"""

import array
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ramvi_service import settings as st
from ramvi_service.app import build
from ramvi_service.chat import ChatError
from ramvi_service.downloads import DownloadError
from ramvi_service.voice import accel as vm_accel
from ramvi_service.voice import models as vm
from ramvi_service.voice.audio import Device, Recorder, peak
from ramvi_service.voice.service import RAM_NEED_MB, VoiceService
from ramvi_service.voice.stt import WhisperProcess
from test_chat import ENGINE, TOKEN, Client


def tone(seconds: float, rate: int = 16_000, amplitude: int = 8000) -> bytes:
    n = int(seconds * rate)
    return array.array("h", [amplitude if i % 40 < 20 else -amplitude for i in range(n)]).tobytes()


class FakeStream:
    def __init__(self, rate: int) -> None:
        self.rate, self.closed = rate, False

    def close(self) -> None:
        self.closed = True


class FakeMicrophone:
    def __init__(self, rate: int = 16_000, names: tuple[str, ...] = ("Mic A", "Mic B")) -> None:
        self.rate, self.names = rate, names
        self.on_block: Callable[[bytes], None] | None = None
        self.opened: list[str | None] = []
        self.stream: FakeStream | None = None

    def devices(self) -> list[Device]:
        return [Device(n, i == 0) for i, n in enumerate(self.names)]

    def open(self, device: str | None, on_block: Callable[[bytes], None]) -> FakeStream:
        if device is not None and device not in self.names:
            raise ChatError("voice.device_missing", device=device)
        self.opened.append(device)
        self.on_block = on_block
        self.stream = FakeStream(self.rate)
        return self.stream

    def say(self, pcm: bytes) -> None:
        assert self.on_block is not None
        step = self.rate // 10 * 2
        for i in range(0, len(pcm), step):
            self.on_block(pcm[i : i + step])


class FakeTranscriber:
    def __init__(self, path: Path, text: str = "привет мир") -> None:
        self.path, self.text = path, text
        self.calls: list[tuple[int, int, str]] = []
        self.prepared = self.closed = 0
        self.running_flag = False

    @property
    def running(self) -> bool:
        return self.running_flag

    def prepare(self) -> None:
        self.prepared += 1
        self.running_flag = True

    def transcribe(self, pcm: bytes, rate: int, language: str) -> dict[str, Any]:
        self.calls.append((len(pcm), rate, language))
        return {"text": f"  {self.text} ", "language": "ru" if language == "auto" else language}

    def close(self) -> None:
        self.closed += 1
        self.running_flag = False


@dataclass(frozen=True)
class VS:
    voice_device: str = ""
    voice_language: str = "auto"
    voice_model: str = vm.DEFAULT_MODEL


class Env:
    def __init__(self, tmp_path: Path, **options: Any) -> None:
        self.models = tmp_path / "models"
        self.settings = VS()
        self.mic = FakeMicrophone()
        self.transcribers: list[FakeTranscriber] = []

        def make(path: Path) -> FakeTranscriber:
            t = FakeTranscriber(path)
            self.transcribers.append(t)
            return t

        self.ram: int | None = None  # свободно RAM, МиБ (None — неизвестно)
        self.vram: int | None = None  # свободно видеопамяти NVIDIA, МиБ (None — NVIDIA нет)
        self.gpu: list[FakeTranscriber] = []

        def make_gpu(path: Path, folder: Path) -> FakeTranscriber:
            t = FakeTranscriber(path, "на видеокарте")
            t.single_use = True  # type: ignore[attr-defined]
            t.folder = folder  # type: ignore[attr-defined]
            self.gpu.append(t)
            return t

        defaults: dict[str, Any] = {
            "make_gpu_transcriber": make_gpu,
            "memory": lambda: (self.ram, self.vram),
            "has_nvidia": lambda: self.vram is not None,
            "platform": "darwin",
        }
        self.voice = VoiceService(
            lambda: self.models,
            lambda: self.settings,
            microphone=self.mic,
            make_transcriber=make,
            **{**defaults, **options},
        )

    def install_accel(self) -> Path:
        folder = vm_accel.CUDA.path(self.models)
        folder.mkdir(parents=True, exist_ok=True)
        for name in vm_accel.CUDA.files:
            (folder / name).write_bytes(b"x")
        return folder

    def install(self, model_id: str = vm.DEFAULT_MODEL, vad: bool = True) -> Path:
        path = vm.MODELS[model_id].path(self.models)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        if vad:
            vm.VAD.path(self.models).write_bytes(b"vad")
        return path


def code(call: Callable[[], Any]) -> str:
    with pytest.raises(ChatError) as error:
        call()
    return error.value.code


def wait(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


# --- запись ---------------------------------------------------------------------


def test_peak_of_int16_block() -> None:
    assert peak(array.array("h", [0, -300, 120]).tobytes()) == 300
    assert peak(b"") == 0


def test_recorder_keeps_audio_in_memory_until_limit() -> None:
    mic, hits = FakeMicrophone(), []
    recorder = Recorder(mic, None, max_seconds=0.95, on_limit=lambda: hits.append(1))
    mic.say(tone(1.5))  # блоки по 0,1 с: предел посреди блока
    assert recorder.seconds == 0.95 and hits == [1]  # лишнее не записано, сигнал один раз
    pcm, rate, loudest = recorder.finish()
    assert len(pcm) == 30_400 and rate == 16_000 and loudest == 8000
    assert mic.stream is not None and mic.stream.closed


class EagerMicrophone(FakeMicrophone):
    """Как PortAudio: блоки приходят ещё до того, как `open` вернул поток."""

    def __init__(self, pcm: bytes) -> None:
        super().__init__()
        self.pcm = pcm

    def open(self, device: str | None, on_block: Callable[[bytes], None]) -> FakeStream:
        stream = super().open(device, on_block)
        self.say(self.pcm)
        return stream


def test_blocks_before_open_returns_are_kept_and_limited() -> None:
    hits: list[int] = []
    recorder = Recorder(EagerMicrophone(tone(0.5)), None, max_seconds=10)
    assert recorder.seconds == 0.5
    recorder = Recorder(
        EagerMicrophone(tone(1.5)), None, max_seconds=0.95, on_limit=lambda: hits.append(1)
    )
    assert recorder.seconds == 0.95 and hits == [1]


class FakeSd:
    """Модуль sounddevice в миниатюре: интерфейсы, устройства, поток."""

    def __init__(self, fail_16k_without_convert: bool = True) -> None:
        self.default = type("D", (), {"device": [1, 3]})()
        self.fail = fail_16k_without_convert
        self.opened: list[tuple[int, int, Any]] = []

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [
            {"name": "MME", "default_input_device": 1},
            {"name": "Windows WASAPI", "default_input_device": 5},
        ]

    def query_devices(self, index: int | None = None) -> Any:
        devices = [
            {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
            {"name": "Microphone (Realtek(R) Au", "hostapi": 0, "max_input_channels": 2},
            {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
            {"name": "Speakers", "hostapi": 1, "max_input_channels": 0},
            {"name": "Mic Array (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2},
            {"name": "Microphone (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2},
        ]
        for d in devices:
            d["default_samplerate"] = 48000.0
        return devices if index is None else devices[index]

    def WasapiSettings(self, auto_convert: bool = False) -> str:  # noqa: N802
        return f"wasapi auto_convert={auto_convert}"

    def RawInputStream(self, **kw: Any) -> Any:  # noqa: N802
        if self.fail and kw["samplerate"] == 16_000 and kw["extra_settings"] is None:
            raise RuntimeError("Invalid sample rate")
        self.opened.append((kw["device"], kw["samplerate"], kw["extra_settings"]))
        return type(
            "S", (), {"start": lambda s: None, "stop": lambda s: None, "close": lambda s: None}
        )()


def test_windows_uses_wasapi_with_full_names_and_16k_convert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ramvi_service.voice.audio import SoundDeviceMicrophone

    sd = FakeSd()
    mic = SoundDeviceMicrophone(platform="win32")
    monkeypatch.setattr(mic, "_sd", lambda: sd)
    assert mic.devices() == [
        Device("Mic Array (Realtek(R) Audio)", False),
        Device("Microphone (Realtek(R) Audio)", True),
    ]
    stream = mic.open(None, lambda b: None)
    assert stream.rate == 16_000 and sd.opened == [(5, 16_000, "wasapi auto_convert=True")]
    mic.open("Mic Array (Realtek(R) Audio)", lambda b: None)
    assert sd.opened[-1][0] == 4


def test_other_systems_keep_default_host_and_fall_back_to_device_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ramvi_service.voice.audio import SoundDeviceMicrophone

    sd = FakeSd()
    mic = SoundDeviceMicrophone(platform="darwin")
    monkeypatch.setattr(mic, "_sd", lambda: sd)
    assert [d.name for d in mic.devices()] == [
        "Microsoft Sound Mapper - Input",
        "Microphone (Realtek(R) Au",
    ]
    assert mic.open(None, lambda b: None).rate == 48_000  # 16 кГц не открылась
    assert sd.opened == [(1, 48_000, None)]


# --- сервис -----------------------------------------------------------------------


def test_record_then_text_for_input_field(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    started = env.voice.start()
    assert started["state"] == "recording"
    env.mic.say(tone(1.0))
    wait(lambda: env.transcribers[0].prepared == 1)  # модель грузится, пока человек говорит
    assert env.voice.status()["seconds"] == 1.0
    result = env.voice.stop()
    assert result["text"] == "привет мир" and result["language"] == "ru"
    assert result["audio_s"] == 1.0
    assert env.transcribers[0].calls == [(32_000, 16_000, "auto")]
    assert env.voice.status()["state"] == "idle"
    assert env.mic.stream is not None and env.mic.stream.closed


@pytest.mark.parametrize("language", ["ru", "en", "uk"])
def test_language_from_settings(tmp_path: Path, language: str) -> None:
    env = Env(tmp_path)
    env.install()
    env.settings = VS(voice_language=language)
    env.voice.start()
    env.mic.say(tone(0.5))
    assert env.voice.stop()["language"] == language
    assert env.transcribers[0].calls[0][2] == language


def test_device_from_settings_or_request(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    env.settings = VS(voice_device="Mic B")
    env.voice.start()
    env.voice.cancel()
    env.voice.start("Mic A")
    env.voice.cancel()
    assert env.mic.opened == ["Mic B", "Mic A"]
    assert env.voice.devices() == {
        "devices": [{"name": "Mic A", "default": True}, {"name": "Mic B", "default": False}]
    }
    env.settings = VS(voice_device="Gone")
    assert code(env.voice.start) == "voice.device_missing"
    assert env.voice.status()["state"] == "idle"


def test_device_rate_is_passed_to_transcriber(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    env.mic.rate = 48_000
    env.voice.start()
    env.mic.say(tone(0.5, rate=48_000))
    env.voice.stop()
    assert env.transcribers[0].calls == [(48_000, 48_000, "auto")]


def test_missing_model_is_reported_before_recording(tmp_path: Path) -> None:
    env = Env(tmp_path)
    assert code(env.voice.start) == "voice.model_missing"
    env.install(vad=False)  # без детектора речи модель не готова: на тишине whisper выдумывает
    assert code(env.voice.start) == "voice.model_missing"
    assert env.mic.opened == [] and env.voice.status()["model"]["installed"] is False


def test_errors_return_to_idle(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    env.voice.start()
    assert code(env.voice.start) == "voice.busy"
    env.mic.say(tone(0.1))
    assert code(env.voice.stop) == "voice.too_short"
    env.voice.start()
    env.mic.say(bytes(16_000))  # нули: macOS без разрешения на микрофон
    assert code(env.voice.stop) == "voice.no_signal"
    env.voice.start()
    env.mic.say(tone(0.5))
    env.transcribers[0].text = "   "
    assert code(env.voice.stop) == "voice.no_speech"
    assert code(env.voice.stop) == "voice.not_recording"
    assert env.voice.status()["state"] == "idle"


def test_cancel_forgets_audio(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    env.voice.start()
    env.mic.say(tone(1.0))
    assert env.voice.cancel()["state"] == "idle"
    assert env.transcribers[0].calls == []
    assert code(env.voice.stop) == "voice.not_recording"


def test_limit_is_visible_in_status(tmp_path: Path) -> None:
    env = Env(tmp_path, max_seconds=0.5)
    env.install()
    env.voice.start()
    env.mic.say(tone(1.0))
    status = env.voice.status()
    assert status["limit_reached"] is True and status["seconds"] == 0.5
    assert env.voice.stop()["audio_s"] == 0.5


def test_model_change_replaces_process_and_idle_unloads(tmp_path: Path) -> None:
    env = Env(tmp_path, idle_seconds=0.05)
    env.install()
    env.install("small-q5_1")
    env.voice.start()
    env.mic.say(tone(0.5))
    env.voice.stop()
    wait(lambda: env.transcribers[0].closed == 1)  # без дела модель выгружается
    env.settings = VS(voice_model="small-q5_1")
    env.voice.start()
    env.mic.say(tone(0.5))
    env.voice.stop()
    assert [t.path.name for t in env.transcribers] == [
        "ggml-large-v3-turbo-q5_0.bin",
        "ggml-small-q5_1.bin",
    ]
    env.voice.close()
    assert env.transcribers[1].closed >= 1


def test_settings_validate_voice_fields() -> None:
    assert st.Settings(voice_language="uk", voice_model="small-q5_1").validate()
    for bad in ({"voice_language": "de"}, {"voice_model": "tiny"}, {"voice_device": 3}):
        with pytest.raises(st.SettingsError):
            st.Settings(**bad).validate()


# --- процесс распознавания ---------------------------------------------------------

FAKE_WORKER = """
import json, sys
from ramvi_service.voice.worker import read_shared
args = sys.argv[1:]
model = args[args.index("--model") + 1]
for line in sys.stdin:
    req = json.loads(line)
    p = req["payload"]
    if p.get("language") == "crash":
        sys.exit(3)
    pcm = read_shared(p["shm"], p["size"])
    text = f"{len(pcm)} {sum(pcm) % 251} {p['rate']} {p['language']} {model}"
    print(json.dumps({"id": req["id"], "result": {"text": text, "language": p["language"]}}),
          flush=True)
"""


def test_worker_process_gets_audio_through_shared_memory(tmp_path: Path) -> None:
    pcm = tone(2.0)
    process = WhisperProcess(
        tmp_path / "m.bin", threads=2, command=[sys.executable, "-c", FAKE_WORKER], timeout=30
    )
    try:
        result = process.transcribe(pcm, 16_000, "uk")
        assert result["text"] == f"{len(pcm)} {sum(pcm) % 251} 16000 uk {tmp_path / 'm.bin'}"
        assert process.running
        assert code(lambda: process.transcribe(pcm, 16_000, "crash")) == "voice.stt_failed"
        assert not process.running
        assert process.transcribe(b"\x01\x00", 8_000, "ru")["text"].startswith("2 1 8000 ru")
    finally:
        process.close()
    assert not process.running


def test_audio_never_reaches_disk(tmp_path: Path) -> None:
    """5.3 VOI-02: звук только в памяти — ни файлов, ни оставленной общей памяти."""
    env = Env(tmp_path)
    env.voice.make_transcriber = lambda path: WhisperProcess(
        path, threads=1, command=[sys.executable, "-c", FAKE_WORKER], timeout=30
    )
    env.install()
    before = {p: p.stat().st_size for p in tmp_path.rglob("*")}
    shm = Path("/dev/shm")
    shm_before = set(shm.glob("ramvi-stt-*")) if shm.is_dir() else set()
    try:
        env.voice.start()
        env.mic.say(tone(1.0))
        assert env.voice.stop()["text"].startswith("32000 ")
    finally:
        env.voice.close()
    assert {p: p.stat().st_size for p in tmp_path.rglob("*")} == before
    if shm.is_dir():
        assert set(shm.glob("ramvi-stt-*")) == shm_before


def test_whisper_wrapper_uses_cpu_vad_and_speech_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Обёртка whisper в процессе распознавания — на подставном pywhispercpp."""
    pytest.importorskip("numpy")
    import types

    from ramvi_service.voice import worker

    calls: list[dict[str, Any]] = []

    class Seg:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def __init__(self, path: str, **kw: Any) -> None:
            calls.append({"init": path, **kw})
            self._ctx = "ctx"
            vad_params = types.SimpleNamespace(speech_pad_ms=30)
            self._params = types.SimpleNamespace(vad_params=vad_params)

        def transcribe(self, audio: Any, **kw: Any) -> list[Seg]:
            calls.append({"samples": int(audio.size), **kw})
            return [Seg(" Привет, "), Seg("мир ")]

    pw = types.SimpleNamespace(whisper_full_lang_id=lambda ctx: 7, whisper_lang_str=lambda i: "uk")
    model_module = types.ModuleType("pywhispercpp.model")
    model_module.Model = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_pywhispercpp", pw)
    monkeypatch.setitem(sys.modules, "pywhispercpp", types.ModuleType("pywhispercpp"))
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)

    whisper = worker.Whisper("m.bin", 3, vad="v.bin")
    assert calls[0]["context_params"] == {"use_gpu": False} and calls[0]["n_threads"] == 3
    assert whisper.model._params.vad_params.speech_pad_ms == worker.SPEECH_PAD_MS == 1000
    result = whisper.transcribe(tone(1.0, 48_000), 48_000, "auto")
    assert result["text"] == "Привет, мир" and result["language"] == "uk"
    assert calls[1] == {
        "samples": 16_000,
        "language": "auto",
        "vad": True,
        "vad_model_path": "v.bin",
    }
    plain = worker.Whisper("m.bin", 1)
    plain.transcribe(tone(0.5), 16_000, "ru")
    assert calls[-1] == {"samples": 8_000, "language": "ru"}
    assert plain.model._params.vad_params.speech_pad_ms == 30


def test_resample_to_16k() -> None:
    np = pytest.importorskip("numpy")
    from ramvi_service.voice.worker import to_16k

    assert to_16k(tone(1.0, 48_000), 48_000).shape == (16_000,)
    assert to_16k(tone(1.0, 44_100), 44_100).shape == (16_000,)
    same = to_16k(tone(0.5), 16_000)
    assert same.dtype == np.float32 and abs(float(same.max()) - 8000 / 32768) < 1e-6


# --- HTTP ---------------------------------------------------------------------------


def test_voice_http_routes(tmp_path: Path) -> None:
    app, data = tmp_path / "app", tmp_path / "app" / "data"
    mic = FakeMicrophone()
    made: list[FakeTranscriber] = []

    def make(path: Path) -> FakeTranscriber:
        made.append(FakeTranscriber(path, "hello"))
        return made[-1]

    server, engines = build(
        data,
        TOKEN,
        engine_command=ENGINE,
        app_dir=app,
        voice_options={
            "microphone": mic,
            "make_transcriber": make,
            "memory": lambda: (None, None),
            "has_nvidia": lambda: False,
        },
    )
    server.start()
    try:
        client = Client(server.port)
        status = client.ok("GET", "/voice")
        assert status["state"] == "idle" and status["model"]["installed"] is False
        assert status["languages"] == ["auto", "ru", "en", "uk"]
        status_code, error = client.request("POST", "/voice/start", {})
        assert status_code == 400 and error["error"]["code"] == "voice.model_missing"
        vm.MODELS[vm.DEFAULT_MODEL].path(app / "models").parent.mkdir(parents=True)
        vm.MODELS[vm.DEFAULT_MODEL].path(app / "models").write_bytes(b"x")
        vm.VAD.path(app / "models").write_bytes(b"v")
        client.ok("POST", "/settings", {"voice_language": "en", "voice_device": "Mic B"})
        assert client.ok("POST", "/voice/start", {})["state"] == "recording"
        mic.say(tone(0.5))
        assert client.ok("POST", "/voice/stop", {}) == {
            "text": "hello",
            "language": "en",
            "audio_s": 0.5,
            "proc_s": pytest.approx(0, abs=0.5),
            "backend": "cpu",
            "model": "large-v3-turbo-q5_0",
        }
        assert mic.opened == ["Mic B"]
        assert client.ok("GET", "/voice/devices")["devices"][1]["name"] == "Mic B"
        status_code, error = client.request("POST", "/settings", {"voice_language": "de"})
        assert status_code == 400 and error["error"]["params"]["field"] == "voice_language"
    finally:
        server.close()
        engines.stop()
    assert made and made[0].closed >= 1  # закрытие сервиса останавливает распознавание


def test_download_gets_vad_then_model_over_https_with_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    gate = threading.Event()

    class FakeDownload:
        def __init__(self, url: str, models_dir: Path, name: str, sha256: str, **kw: Any) -> None:
            self.info = {"url": url, "name": name, "sha256": sha256, "size": kw.get("size")}
            self.target = Path(models_dir) / name
            self.cancelled = False

        def run(self) -> Path:
            seen.append(self.info)
            if str(self.info["name"]).endswith("turbo-q5_0.bin"):
                gate.wait(5)
                if self.cancelled:
                    raise DownloadError("download.cancelled")
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.write_bytes(b"x")
            return self.target

        def cancel(self) -> None:
            self.cancelled = True
            gate.set()

        def discard(self) -> None:
            pass

    monkeypatch.setattr("ramvi_service.voice.service.Download", FakeDownload)
    env = Env(tmp_path)
    assert env.voice.download_model()["download"]["state"] == "running"
    assert code(env.voice.download_model) == "voice.download_running"
    wait(lambda: len(seen) == 2)
    gate.set()
    wait(lambda: env.voice.status()["download"]["state"] == "done")
    model = vm.MODELS[vm.DEFAULT_MODEL]
    assert seen == [
        {
            "url": vm.VAD.url,
            "name": f"stt/{vm.VAD.file}",
            "sha256": vm.VAD.sha256,
            "size": vm.VAD.size,
        },
        {"url": model.url, "name": f"stt/{model.file}", "sha256": model.sha256, "size": model.size},
    ]
    assert (
        vm.VAD.url
        == "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
    )
    assert env.voice.status()["model"]["installed"] is True
    assert [(m["id"], m["installed"]) for m in env.voice.status()["models"]] == [
        ("large-v3-turbo-q5_0", True),
        ("small-q5_1", False),
    ]
    assert code(lambda: env.voice.download_model("nope")) == "voice.unknown_model"

    gate.clear()
    model.path(env.models).unlink()
    env.voice.download_model()
    wait(lambda: len(seen) == 4)
    env.voice.cancel_download()
    wait(lambda: env.voice.status()["download"]["state"] == "idle")
    assert env.voice.status()["model"]["installed"] is False


def test_worker_gets_vad_next_to_model(tmp_path: Path) -> None:
    model = tmp_path / "stt" / "m.bin"
    model.parent.mkdir()
    assert "--vad" not in WhisperProcess(model, threads=1).process.command
    vm.VAD.path(tmp_path).write_bytes(b"v")
    command = WhisperProcess(model, threads=1).process.command
    assert command[command.index("--vad") + 1] == str(vm.VAD.path(tmp_path))


# --- 5.4: где распознавать -------------------------------------------------------------


def record_and_stop(env: Env) -> dict[str, Any]:
    env.voice.start()
    env.mic.say(tone(0.5))
    return env.voice.stop()


def test_gpu_used_when_accel_installed_and_vram_free(tmp_path: Path) -> None:
    env = Env(tmp_path, platform="win32")
    env.install()
    folder = env.install_accel()
    env.vram = vm_accel.VRAM_NEED_MB["large-v3-turbo-q5_0"]
    env.voice.start()
    assert env.voice.status()["backend"] == "gpu"
    env.mic.say(tone(0.5))
    result = env.voice.stop()
    assert (result["text"], result["backend"]) == ("на видеокарте", "gpu")
    assert env.gpu[0].folder == folder and env.gpu[0].prepared == 1  # type: ignore[attr-defined]
    assert env.gpu[0].closed == 1  # видеопамять освобождена сразу после распознавания
    assert env.transcribers == []
    status = env.voice.status()
    assert status["last"]["backend"] == "gpu" and status["backend"] is None
    assert status["accel"]["offered"] and status["accel"]["installed"]


def test_cpu_when_vram_short_or_no_accel_or_not_windows(tmp_path: Path) -> None:
    env = Env(tmp_path, platform="win32")
    env.install()
    env.vram = 8000
    assert record_and_stop(env)["backend"] == "cpu"  # ускоритель не скачан
    env.install_accel()
    env.vram = vm_accel.VRAM_NEED_MB["large-v3-turbo-q5_0"] - 1  # занято LLM
    assert record_and_stop(env)["backend"] == "cpu"
    assert env.gpu == [] and len(env.transcribers) == 1  # процесс CPU переиспользуется
    mac = Env(tmp_path / "mac", platform="darwin")
    mac.install()
    mac.install_accel()
    mac.vram = 8000
    assert record_and_stop(mac)["backend"] == "cpu"
    assert mac.voice.status()["accel"]["offered"] is False


def test_low_ram_falls_back_to_small_or_refuses_before_recording(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.install()
    env.ram = RAM_NEED_MB["large-v3-turbo-q5_0"] - 1
    assert code(env.voice.start) == "voice.no_memory"
    assert env.mic.opened == []  # микрофон не открывался
    env.install("small-q5_1")
    result = record_and_stop(env)
    assert result["model"] == "small-q5_1" and result["backend"] == "cpu"
    env.ram = RAM_NEED_MB["small-q5_1"] - 1
    with pytest.raises(ChatError) as error:
        env.voice.start()
    assert error.value.code == "voice.no_memory"
    assert error.value.params == {"need_mb": RAM_NEED_MB["large-v3-turbo-q5_0"], "free_mb": env.ram}
    env.ram = None  # замер не удался — не мешать записи
    assert record_and_stop(env)["model"] == "large-v3-turbo-q5_0"


def test_accel_download_checks_hash_then_extracts_only_needed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    seen: list[dict[str, Any]] = []

    class FakeDownload:
        def __init__(self, url: str, models_dir: Path, name: str, sha256: str, **kw: Any) -> None:
            seen.append({"url": url, "name": name, "sha256": sha256, "size": kw.get("size")})
            self.target = Path(models_dir) / name

        def run(self) -> Path:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.target, "w") as zf:
                for name in (*vm_accel.CUDA.files, "wchess.exe", "nvblas64_12.dll"):
                    if name != drop:
                        zf.writestr(f"Release/{name}", name)
            return self.target

        def cancel(self) -> None:
            pass

        def discard(self) -> None:
            pass

    monkeypatch.setattr("ramvi_service.voice.service.Download", FakeDownload)
    assert code(Env(tmp_path / "mac").voice.download_accel) == "voice.accel_unsupported"
    env = Env(tmp_path, platform="win32")
    env.vram = 8000
    drop = "ggml-cuda.dll"  # испорченный архив: нет нужного файла
    env.voice.download_accel()
    wait(lambda: env.voice.status()["download"]["state"] != "running")
    status = env.voice.status()
    assert status["download"]["state"] == "failed" and not status["accel"]["installed"]
    assert not vm_accel.CUDA.path(env.models).exists()
    drop = ""
    env.voice.download_accel()
    wait(lambda: env.voice.status()["download"]["state"] == "done")
    folder = vm_accel.CUDA.path(env.models)
    assert sorted(p.name for p in folder.iterdir()) == sorted(vm_accel.CUDA.files)
    assert (folder / "ggml-cuda.dll").read_text() == "ggml-cuda.dll"
    assert not (env.models / vm_accel.CUDA.archive(env.models)).exists()  # архив удалён
    assert env.voice.status()["accel"]["installed"] is True
    assert seen[-1] == {
        "url": vm_accel.CUDA.url,
        "name": vm_accel.CUDA.archive(env.models),
        "sha256": vm_accel.CUDA.sha256,
        "size": vm_accel.CUDA.size,
    }
    assert vm_accel.CUDA.url.startswith("https://github.com/ggml-org/whisper.cpp/releases/")


def test_pure_python_resample_and_wav() -> None:
    import wave as wav_module

    from ramvi_service.voice.server import language_code, to_16k_pcm, wav_bytes

    assert len(to_16k_pcm(tone(1.0, 48_000), 48_000)) == 32_000
    assert len(to_16k_pcm(tone(1.0, 44_100), 44_100)) == 32_000
    same = tone(0.5)
    assert to_16k_pcm(same, 16_000) is same
    block = array.array("h", [3000] * 48).tobytes()
    assert set(array.array("h", to_16k_pcm(block, 48_000))) == {3000}
    import io

    with wav_module.open(io.BytesIO(wav_bytes(same))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
        assert w.readframes(w.getnframes()) == same
    assert language_code({"language_probabilities": {"ru": 0.9, "uk": 0.1}}, "auto") == "ru"
    assert language_code({"detected_language": "ukrainian"}, "auto") == "uk"
    assert language_code({"detected_language": "ukrainian"}, "en") == "en"


FAKE_SERVER = r"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
vad = "--vad" in args and args[args.index("-vp") + 1]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
        parts = {}
        for part in body.split(b"--" + boundary)[1:-1]:
            head, _, data = part.partition(b"\r\n\r\n")
            name = head.split(b'name="')[1].split(b'"')[0].decode()
            parts[name] = data[:-2]
        wav = parts["file"]
        lang = parts["language"].decode()
        reply = {"text": f" {len(wav)} {lang} vad={vad}\n", "detected_language": "ukrainian",
                 "language_probabilities": {"uk": 0.97, "ru": 0.03}}
        if lang == "ru":  # как у настоящего сервера: граница фрагмента посреди слова
            reply["text"] = " ночного вид\nения и\n"
            reply["segments"] = [{"text": " ночного вид"}, {"text": "ения и"}]
        out = json.dumps(reply).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def test_whisper_server_process_gets_wav_over_localhost(tmp_path: Path) -> None:
    from ramvi_service.voice.server import WhisperServer

    model = tmp_path / "stt" / "m.bin"
    model.parent.mkdir()
    vm.VAD.path(tmp_path).write_bytes(b"v")
    server = WhisperServer(
        model, tmp_path, threads=2, command=[sys.executable, "-c", FAKE_SERVER], start_timeout=30
    )
    try:
        command = server.process.command
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("-vp") + 1] == "1000"
        server.prepare()
        assert server.running
        result = server.transcribe(tone(1.0, 48_000), 48_000, "auto")
        assert result == {"text": "32044 auto vad=1000", "language": "uk"}  # WAV: 44 + 32000
        assert server.transcribe(tone(0.5), 16_000, "ru") == {
            "text": "ночного видения и",
            "language": "ru",
        }
    finally:
        server.close()
    assert not server.running


def test_whisper_server_that_never_starts_is_reported(tmp_path: Path) -> None:
    from ramvi_service.voice.server import WhisperServer

    server = WhisperServer(
        tmp_path / "m.bin", tmp_path, command=[sys.executable, "-c", "raise SystemExit(3)"]
    )
    assert code(server.prepare) == "voice.stt_failed"
    assert not server.running


def test_cpu_worker_lowers_its_priority() -> None:
    script = (
        "import os, sys; from ramvi_service.voice.worker import lower_priority; "
        "before = os.nice(0) if sys.platform != 'win32' else 0; lower_priority(); "
        "print(os.nice(0) - before if sys.platform != 'win32' else 'win')"
    )
    import os
    import subprocess

    service = Path(__file__).resolve().parents[1] / "service"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(service)},
    ).stdout.strip()
    assert out in ("5", "win")


def test_segments_join_without_breaking_words() -> None:
    assert vm.join_segments([" Они умеют видеть при помощи ночного вид", "ения и почти"]) == (
        "Они умеют видеть при помощи ночного видения и почти"
    )
    assert vm.join_segments([" Привет,", " как дела?\n", " Хорошо."]) == "Привет, как дела? Хорошо."
    assert vm.join_segments(["", "  "]) == ""


def test_gpu_also_needs_some_ram(tmp_path: Path) -> None:
    from ramvi_service.voice.service import GPU_RAM_NEED_MB

    env = Env(tmp_path, platform="win32")
    env.install()
    env.install_accel()
    env.vram = 8000
    env.ram = GPU_RAM_NEED_MB
    assert record_and_stop(env)["backend"] == "gpu"
    env.ram = GPU_RAM_NEED_MB - 1  # 14B занял почти всю RAM: ни видеокарта, ни CPU
    assert code(env.voice.start) == "voice.no_memory"
