"""Live Character understanding checks using an owned portable Engine instance."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--suite", choices=("development", "blind", "phonetic"), default="development")
parser.add_argument("--model", default="qwen3:8b")
args = parser.parse_args()
bundle, source, work = args.bundle.resolve(), args.source.resolve(), args.work.resolve()
work.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(source / "character"),
    str(bundle / "ramvi" / "service"),
    str(bundle / "ramvi" / "engine"),
]
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"})
os.environ["PATH"] = str(bundle / "runtime" / "python") + os.pathsep + os.environ.get("PATH", "")
from context_blind_cases import cases as blind_cases
from context_cases import cases as development_cases
from context_cases import evaluate
from ramvi_character.context import Budget, ContextOrchestrator, tokenizer_counter
from ramvi_character.emotions import EmotionalState
from ramvi_character.personality import Personality
from ramvi_engine.engine_process import EngineProcess
from ramvi_engine.model_loader import ModelConfig
from ramvi_engine.planning.automatic import PlanningOptions
from ramvi_service.contracts.modules import Authority
from ramvi_service.providers import LocalEngineProvider
from ramvi_service.resources import SharedLLM


def busy_gpu() -> bool:
    # Both parallel sessions must honor this marker; never kill somebody else's job.
    script = r"""
$all=Get-CimInstance Win32_Process
$own=New-Object 'System.Collections.Generic.HashSet[int]'
[void]$own.Add(PYTHONPID)
for($i=0;$i -lt 20;$i++) {foreach($p in $all) {if($own.Contains([int]$p.ParentProcessId)){[void]$own.Add([int]$p.ProcessId)}}}
$busy=@($all | Where-Object {(-not $own.Contains([int]$_.ProcessId)) -and ($_.Name -in @('WorldOfTanks.exe','WorldOfTanks64.exe','ollama.exe','RAMVI.exe') -or ($_.Name -in @('python.exe','pythonw.exe') -and $_.CommandLine -like '*ramvi_engine*'))})
if($busy.Count -gt 0){'BUSY'}else{'FREE'}
""".replace("PYTHONPID", str(os.getpid()))
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=15
    )
    if result.returncode:
        raise RuntimeError("cannot determine shared GPU ownership")
    return "BUSY" in result.stdout


def main() -> None:
    lock = work.parent / "ramvi_gpu_owner.json"
    if busy_gpu():
        raise RuntimeError("shared GPU is busy; check deferred")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"owner": "character-context", "root": str(work), "pid": os.getpid()}, handle)
    engine = None
    records = []
    try:
        used = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if used != "0" or busy_gpu():
            raise RuntimeError("shared GPU became busy; check deferred")
        from transformers import AutoTokenizer

        tokenizer_path = bundle / "runtime" / "tokenizers" / "Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        counter = tokenizer_counter(tokenizer, "Qwen3-8B-local-template")
        engine = EngineProcess(
            ModelConfig(
                args.model,
                maxt=8192,
                tokenizer=str(tokenizer_path),
                planning=PlanningOptions(draft="off"),
            ),
            startup_timeout=180,
            cwd=work,
        )
        engine.start()
        authority = Authority()
        provider = LocalEngineProvider(
            engine.base_url, args.model, api_key=engine.api_key, timeout=90
        )
        llm = SharedLLM(authority, provider)
        owner = authority.issue("character", "acceptance-user", "understanding-check")
        llm.retain(owner)
        cancel = threading.Event()
        deadline = time.monotonic() + 1800
        try:
            case_source = blind_cases if args.suite == "blind" else development_cases
            files = sorted(
                p
                for p in (source / "character" / "ramvi_character").rglob("*")
                if p.is_file() and p.suffix in {".py", ".json"} and "assets" not in p.parts
            )
            runtime_hash = hashlib.sha256(
                b"".join(
                    p.relative_to(source).as_posix().encode() + b"\0" + p.read_bytes()
                    for p in files
                )
            ).hexdigest()
            for run in range(args.runs):
                for case in case_source(run)[:2] if args.suite == "phonetic" else case_source(run):
                    if time.monotonic() >= deadline or busy_gpu():
                        raise RuntimeError(
                            "deadline or another GPU session detected; own engine stopped"
                        )
                    personality = Personality()
                    personality.select("sarcastic")
                    if not case.get("boundary"):
                        personality.boundary("teasing", False)
                    orchestrator = ContextOrchestrator("acceptance-user", counter)
                    memory = (
                        [
                            {
                                "id": "injected",
                                "scope": "acceptance-user",
                                "text": case["memory"],
                                "relevance": 1,
                                "confidence": 0.1,
                                "status": "unconfirmed",
                            }
                        ]
                        if "memory" in case
                        else []
                    )
                    prepared = orchestrator.build(
                        task=case["task"],
                        user_input=case["text"],
                        personality=personality.view(),
                        state=EmotionalState().view(),
                        recent=case.get("history", []),
                        memories=memory,
                        budget=Budget(),
                    )
                    usage = {}
                    raw_response = []

                    def complete(request, usage=usage, raw_response=raw_response):
                        response = llm.complete(owner, request, timeout=90, cancel=cancel)
                        usage.update(response.get("usage", {}))
                        raw_response.append(
                            response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        )
                        return response

                    before = personality.checkpoint()
                    started = time.monotonic()
                    try:
                        result = orchestrator.understand(
                            prepared, complete, model=args.model, cancel=cancel.is_set
                        )
                        accepted = evaluate(case, result)
                        accepted = accepted and personality.checkpoint() == before
                        accepted = (
                            accepted and usage.get("prompt_tokens") == result["prompt_tokens"]
                        )
                        item = {
                            "run": run + 1,
                            "case": case["id"],
                            "accepted": bool(accepted),
                            "analysis": result,
                            "seconds": round(time.monotonic() - started, 2),
                            "counted_input": prepared.accounting["input"],
                            "usage": usage,
                        }
                    except Exception as error:
                        item = {
                            "run": run + 1,
                            "case": case["id"],
                            "accepted": False,
                            "failure": type(error).__name__,
                            "detail": str(error)[:200],
                            "raw": raw_response[-1] if raw_response else "",
                        }
                    item["suite"] = (
                        "blind"
                        if args.suite == "blind"
                        else (
                            "validation_reused"
                            if case["id"].startswith("holdout_")
                            else "development"
                        )
                    )
                    if args.suite == "phonetic":
                        item["suite"] = "phonetic"
                    item["model"] = args.model
                    item["runtime_sha256"] = runtime_hash
                    records.append(item)
                    (
                        work
                        / (
                            "context_blind_results.json"
                            if args.suite == "blind"
                            else "context_phonetic_results.json"
                            if args.suite == "phonetic"
                            else "context_results.json"
                        )
                    ).write_text(
                        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(
                        json.dumps(
                            {k: v for k, v in item.items() if k not in {"analysis", "usage"}},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if run == 0 and case["id"] == "literal_control" and not item["accepted"]:
                        raise RuntimeError(
                            "solvable control failed; stop before comparative claims"
                        )
        finally:
            cancel.set()
            llm.release_owner(owner)
            authority.revoke(owner)
    finally:
        if engine is not None:
            engine.close()
        try:
            record = json.loads(lock.read_text(encoding="utf-8"))
            if record.get("pid") == os.getpid() and record.get("root") == str(work):
                lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
