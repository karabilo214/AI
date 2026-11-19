"""Owned Windows GPU check: UI HTTP -> Character worker -> SharedLLM -> Engine.

No user database or installed application is modified. Requires an external absolute
stop task, a free GPU and exclusive ownership marker. All own processes close in finally.
"""
# ruff: noqa: E402

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
bundle, source, work = args.bundle.resolve(), args.source.resolve(), args.work.resolve()
sys.path[:0] = [str(source / p) for p in ("service", "storage", "character")] + [
    str(bundle / "ramvi" / "engine")
]
os.environ.update(
    {
        "RAMVI_TOKENIZER": str(bundle / "runtime" / "tokenizers" / "Qwen3-8B"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUTF8": "1",
    }
)
os.environ["PATH"] = str(bundle / "runtime" / "python") + os.pathsep + os.environ.get("PATH", "")

from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.formats.gguf import find_ollama
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.planning.automatic import PlanningOptions
from ramvi_service.contracts.modules import Authority
from ramvi_service.engines import EngineManager, SwitchableProvider
from ramvi_service.modules.host import ModuleHost
from ramvi_service.resources import SharedLLM
from ramvi_service.sections import SectionRegistry
from ramvi_service.ui_api import UIServer
from ramvi_storage.repositories import Model, Models
from ramvi_storage.schema import open_core


def free():
    script = r"""
$all=Get-CimInstance Win32_Process
$own=New-Object 'System.Collections.Generic.HashSet[int]'
[void]$own.Add(PYTHONPID)
for($i=0;$i -lt 20;$i++){foreach($p in $all){if($own.Contains([int]$p.ParentProcessId)){[void]$own.Add([int]$p.ProcessId)}}}
$busy=@($all|Where-Object {(-not $own.Contains([int]$_.ProcessId)) -and ($_.Name -in @('WorldOfTanks.exe','WorldOfTanks64.exe','ollama.exe','RAMVI.exe') -or ($_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -like '*ramvi_engine*'))})
if($busy.Count){'BUSY'}else{'FREE'}
""".replace("PYTHONPID", str(os.getpid()))
    p = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=15
    )
    return p.returncode == 0 and "FREE" in p.stdout


def main():
    work.mkdir(parents=True, exist_ok=True)
    lock = work.parent / "ramvi_gpu_owner.json"
    if not free():
        raise RuntimeError("another GPU session is active")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        json.dump({"owner": "character-dialogue", "root": str(work), "pid": os.getpid()}, f)
    server = host = manager = db = None
    results = []
    try:
        if (
            not free()
            or subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            != "0"
        ):
            raise RuntimeError("GPU no longer free")
        db = open_core(work / ("data_smoke" if args.smoke else "data_full"))
        models = Models(db)
        for name in ["qwen3:8b", "qwen3:14b"]:
            models.upsert(Model(name, "local", "qwen3", str(find_ollama(name)), time.time()))

        def make_engine(model, draft=None, maxt=8192, resident=None):
            return EngineProcess(
                ModelConfig(
                    model.path,
                    maxt=maxt,
                    tokenizer=os.environ["RAMVI_TOKENIZER"],
                    planning=PlanningOptions(draft="off"),
                ),
                startup_timeout=180,
                cwd=work,
            )

        manager = EngineManager(models, work / "run", make_engine=make_engine, context=lambda: 8192)
        authority = Authority()
        host = ModuleHost(
            work / ("data_smoke" if args.smoke else "data_full"),
            llm=SharedLLM(authority, SwitchableProvider(manager)),
        )
        registry = SectionRegistry(host)
        registry.discover(source / "modules")
        if registry.problems:
            raise RuntimeError(registry.problems)
        token = "character-live-" + os.urandom(16).hex()
        server = UIServer(None, token, sections=registry).start()

        def request(path, body=None):
            c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=20)
            try:
                c.request(
                    "POST",
                    "/ui/v1" + path,
                    json.dumps(body or {}).encode(),
                    {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                )
                r = c.getresponse()
                value = json.loads(r.read())
                if r.status != 200:
                    raise RuntimeError(value)
                return value
            finally:
                c.close()

        def cmd(operation, **arguments):
            r = request(
                "/sections/character/commands/dialogue",
                {"operation": operation, "arguments": arguments},
            )
            if "error" in r:
                raise RuntimeError(r)
            return r

        assert request("/sections/character/open")["section"]["state"] == "active"
        cmd("profile", id="friendly")
        cmd("boundary", name="teasing", allowed=False)

        runtime_files = sorted(
            p
            for p in (source / "character").rglob("*")
            if p.suffix in {".py", ".json"} and "assets" not in p.parts
        )
        runtime_hash = hashlib.sha256(
            b"".join(
                p.relative_to(source).as_posix().encode() + b"\0" + p.read_bytes()
                for p in runtime_files
            )
        ).hexdigest()

        def reply(label, prompt, expected, model="qwen3:8b"):
            if not free():
                raise RuntimeError("parallel GPU session detected")
            started = time.monotonic()
            identifier = cmd("start", text=prompt, model=model)["response_id"]
            cursor = 0
            events = []
            while time.monotonic() - started < 180:
                job = cmd("poll", response_id=identifier, cursor=cursor)["job"]
                events.extend(job["events"])
                cursor = job["cursor"]
                if job["state"] != "running":
                    break
                time.sleep(0.1)
            output = "\n".join(e["text"] for e in events if e["type"] == "segment")
            item = {
                "id": label,
                "runtime_sha256": runtime_hash,
                "model": model,
                "state": job["state"],
                "fallback": job["fallback"],
                "text": output,
                "expected": expected,
                "accepted": job["state"] == "completed" and expected in output,
                "elapsed_s": round(time.monotonic() - started, 3),
                "events": events,
            }
            results.append(item)
            (work / "dialogue_results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps({k: v for k, v in item.items() if k != "events"}, ensure_ascii=False),
                flush=True,
            )
            if label == "solvable_control" and not item["accepted"]:
                raise RuntimeError("solvable control failed")

        reply("solvable_control", "Сколько будет 2 + 2? Ответь коротко.", "4")
        for run in range(0 if args.smoke else 3):
            a = 100037 + run * 127
            b = 700019 + run * 239
            reply(
                "fresh_sum_" + str(run),
                f"В коробке {a} синих и {b} красных деталей. Сколько всего? Напиши точное число.",
                str(a + b),
            )
        for run in range(3):
            nonce = f"МАЯК-{101903 + run * 331}"
            reply(f"fresh_copy_{run}", f"Повтори ровно этот код: {nonce}", nonce)
        reply(
            "remember",
            "Мой выдуманный маяк называется ЛИСТ-731. Повтори название коротко.",
            "ЛИСТ-731",
        )
        # Stop before a late record can be presented; no TTS/renderer exists in the worker.
        identifier = cmd(
            "start", text="Напиши очень длинный рассказ о море на 3000 слов.", model="qwen3:8b"
        )["response_id"]
        time.sleep(0.25)
        cmd("cancel")
        time.sleep(1)
        cancelled = cmd("poll", response_id=identifier)["job"]
        assert cancelled["state"] == "cancelled" and not cancelled["events"], cancelled
        host.suspend("character")
        host.resume("character")
        restored = cmd("status")
        assert restored["character"]["profile_id"] == "friendly"
        assert any("ЛИСТ-731" in row["assistant"] for row in restored["history"])
        reply(
            "different_model_remembers",
            "Как назывался мой выдуманный маяк? Ответь только названием.",
            "ЛИСТ-731",
            "qwen3:14b",
        )
        reply("switch_back", "Повтори ровно код: ВЕТЕР-900731", "ВЕТЕР-900731")
        status = host.invoke("character", "status")
        assert not status["devices_open"] and status["presentation"] is None
        assert not status["personality"]["boundaries"]["teasing"]
        (work / "dialogue_lifecycle.json").write_text(
            json.dumps(
                {
                    "cancelled": cancelled,
                    "restored": restored["character"],
                    "model": manager.model_id,
                    "devices_open": status["devices_open"],
                    "presentation": status["presentation"],
                    "teasing": status["personality"]["boundaries"]["teasing"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if server:
            server.close()
        if host:
            host.close()
        if manager:
            manager.stop()
        if db:
            db.close()
        try:
            owner = json.loads(lock.read_text())
            if owner.get("pid") == os.getpid():
                lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
