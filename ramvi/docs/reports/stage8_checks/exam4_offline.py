"""8.6, экзамен 4 ТЗ на ноутбуке с RTX 5060: без сети работают чат (настоящая qwen3:8b),
голос (настоящий whisper, русская запись FLEURS) и файловые инструменты агента.

Сеть отрезана ловушкой во всех процессах Python комплекта (offline_guard_sitecustomize.py),
окружение как у оболочки: HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, PYTHONNOUSERSITE.
Запуск: runtime\\python\\python.exe exam4_offline.py (из папки проверки).
"""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

HERE = Path(sys.argv[0]).resolve().parent
BUNDLE = Path(os.environ["USERPROFILE"]) / "Desktop" / "RAMVI"
SPEECH = Path(r"C:\anm\voice_test\data\fleurs\ru_ru")
GUARD_LOG = HERE / "guard.log"
TOKEN = "offline-exam-" + "x" * 24
RESULTS: dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(*parts: object) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def save() -> None:
    (HERE / "exam4_results.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Ui:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}/ui/v1"

    def call(self, method: str, path: str, body: Any = None, timeout: float = 900) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {TOKEN}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: {error.code} {error.read()[:400]!r}") from None
        if raw.startswith("data:") or "\ndata:" in raw:
            return [json.loads(line[5:]) for line in raw.splitlines() if line.startswith("data:")]
        return json.loads(raw)


class Stream:
    def __init__(self, rate: int) -> None:
        self.rate = rate

    def close(self) -> None:
        pass


class WavMicrophone:
    """Микрофон, который «говорит» запись из файла."""

    def __init__(self) -> None:
        self.on_block: Any = None
        self.rate = 16_000

    def devices(self) -> list[Any]:
        from ramvi_service.voice.audio import Device

        return [Device("FLEURS", True)]

    def open(self, device: str | None, on_block: Any) -> Stream:
        self.on_block = on_block
        return Stream(self.rate)

    def play(self, path: Path) -> float:
        with wave.open(str(path)) as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2, "нужен моно 16 бит"
            self.rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
        step = self.rate // 10 * 2
        for i in range(0, len(pcm), step):
            self.on_block(pcm[i : i + step])
            time.sleep(0.02)
        return len(pcm) / 2 / self.rate


def memory() -> dict[str, int]:
    """Свободная RAM и рабочие наборы процессов python (МБ) — для находки о памяти."""
    import ctypes
    import subprocess

    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("avail", ctypes.c_ulonglong),
            ("pf_total", ctypes.c_ulonglong),
            ("pf_avail", ctypes.c_ulonglong),
            ("v_total", ctypes.c_ulonglong),
            ("v_avail", ctypes.c_ulonglong),
            ("ext", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
    ).stdout.decode("cp866", "replace")
    sets = sorted(
        (
            int("".join(c for c in line.split('","')[-1] if c.isdigit()) or 0) // 1024
            for line in out.splitlines()
            if line
        ),
        reverse=True,
    )
    return {"free_ram_mb": status.avail // 2**20, "python_processes_mb": sets}


def words(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def qwen_blob() -> Path:
    models = Path(os.environ["USERPROFILE"]) / ".ollama" / "models"
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "8b"
    layers = json.loads(manifest.read_text(encoding="utf-8"))["layers"]
    digest = next(x["digest"] for x in layers if x["mediaType"].endswith(".model"))
    return models / "blobs" / digest.replace(":", "-")


def guard_check() -> None:
    """Контроль: ловушка правда отрезает сеть (иначе пустой журнал ничего не значит)."""
    for url in ("https://huggingface.co/", "http://192.0.2.1/"):
        try:
            urllib.request.urlopen(url, timeout=5)
            RESULTS.setdefault("control_failed", []).append(url)
        except (OSError, urllib.error.URLError) as error:
            RESULTS.setdefault("control_blocked", []).append(f"{url}: {error}")


def main() -> None:
    if "guard on" not in GUARD_LOG.read_text(encoding="utf-8"):
        raise SystemExit("ловушка не загружена: нет sitecustomize в комплекте")
    guard_check()
    from ramvi_service.app import build

    work = HERE / "work"
    stt_link = work / "app" / "models" / "stt"
    if stt_link.exists():
        os.rmdir(stt_link)  # снять связь, не трогая модели комплекта
    shutil.rmtree(work, ignore_errors=True)
    app_dir, data = work / "app", work / "data"
    app_dir.mkdir(parents=True)
    (app_dir / "settings.json").write_text(
        json.dumps(
            {"data_dir": str(data), "models_dir": str(BUNDLE / "models"), "voice_language": "ru"}
        ),
        encoding="utf-8",
    )
    # Голос ищет модели в папке приложения (app/models/stt) — связать с моделями комплекта.
    import subprocess

    (app_dir / "models").mkdir()
    subprocess.run(
        [
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(app_dir / "models" / "stt"),
            str(BUNDLE / "models" / "stt"),
        ],
        check=True,
        capture_output=True,
    )
    mic = WavMicrophone()
    server, engines = build(data, TOKEN, app_dir=app_dir, voice_options={"microphone": mic})
    server.start()
    ui = Ui(server.port)
    try:
        model = ui.call("POST", "/models/import", {"path": str(qwen_blob())})["model"]["id"]
        RESULTS["model"] = model
        log("chat", model)
        chat = ui.call("POST", "/chats", {"model_id": model})["chat"]["id"]
        started = time.monotonic()
        events = ui.call(
            "POST", f"/chats/{chat}/send", {"text": "Сколько будет 17*23? Ответь только числом."}
        )
        answer = "".join(e.get("text", "") for e in events if e.get("type") == "text")
        RESULTS["chat"] = {
            "finish": events[-1].get("finish"),
            "error": events[-1].get("error"),
            "answer": answer[-200:],
            "has_391": "391" in answer,
            "seconds": round(time.monotonic() - started, 1),
        }
        RESULTS["memory_with_model"] = memory()
        # Голосу (whisper на CPU) нужно ~1,4 ГБ RAM; с загруженной 8B на этом ноутбуке их нет —
        # пользователь выгрузил бы модель кнопкой «Выгрузить». Делаем так же.
        if "--keep-model" in sys.argv:  # после исправления 8.6: голос при загруженной модели
            RESULTS["unload"] = "skipped (--keep-model)"
        else:
            RESULTS["unload"] = ui.call("POST", "/engine/unload", {}).get("unloaded")
        time.sleep(3)
        RESULTS["memory_after_unload"] = memory()
        save()
        log("voice")
        refs = dict(
            line.split("\t", 1)
            for line in (SPEECH / "ref.tsv").read_text(encoding="utf-8").splitlines()
            if "\t" in line
        )
        voice = []
        for name in list(refs)[:3]:
            ui.call("POST", "/voice/start", {})
            seconds = mic.play(SPEECH / "wav" / name)
            got = ui.call("POST", "/voice/stop", {})
            ref, hyp = words(refs[name]), words(got.get("text", ""))
            same = sum(1 for w in hyp if w in set(ref))
            voice.append(
                {
                    "file": name,
                    "audio_s": round(seconds, 1),
                    "ref": refs[name],
                    "text": got.get("text"),
                    "backend": got.get("backend"),
                    "words_matched": f"{same}/{len(ref)}",
                }
            )
        RESULTS["voice"] = voice
        save()
    finally:
        server.close()
        engines.stop()

    log("agent files")
    from ramvi_service.agent.changes import ChangeStore
    from ramvi_service.agent.files import FileLimits, register_file_tools
    from ramvi_service.agent.tools import ToolRunner
    from ramvi_service.agent.workspace import Permissions, Workspace

    class Journal:
        def begin(self, tool: str, args: dict[str, Any], permission: str) -> str:
            return "0"

        def finish(
            self, call_id: str, result: str, duration: float, permission: str | None = None
        ) -> None:
            pass

    project = work / "project"
    project.mkdir()
    (project / "main.py").write_bytes(b"def main():\n    return 41\n")
    workspace = Workspace(project, Permissions(write=True))
    runner = ToolRunner()
    register_file_tools(runner, workspace, FileLimits())
    read = runner.execute("read_file", {"path": "main.py"}, Journal())
    grep = runner.execute("grep", {"pattern": "return"}, Journal())
    edit = runner.execute(
        "edit_file",
        {"path": "main.py", "old_text": "return 41", "new_text": "return 42"},
        Journal(),
    )
    ChangeStore(work / "changes", workspace).apply(edit.data["proposal"])
    RESULTS["agent_files"] = {
        "read": read.ok,
        "grep": grep.ok,
        "diff": "+    return 42" in edit.data["proposal"].diff,
        "applied": (project / "main.py").read_bytes() == b"def main():\n    return 42\n",
    }

    lines = GUARD_LOG.read_text(encoding="utf-8").splitlines()
    processes = {line.split("\t")[0] for line in lines if line.endswith("guard on")}
    blocked = [line for line in lines if "BLOCKED" in line]
    controls = [b for b in blocked if "huggingface.co" in b or "192.0.2.1" in b]
    RESULTS["guard"] = {
        "processes_guarded": len(processes),
        "blocked_total": len(blocked),
        "blocked_besides_control": [b for b in blocked if b not in controls],
        "commands": sorted({line.split("\t")[1] for line in lines if line.endswith("guard on")}),
    }
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("done")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        RESULTS["crash"] = traceback.format_exc()
        save()
        raise
