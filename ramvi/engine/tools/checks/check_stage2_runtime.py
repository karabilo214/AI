"""Section 2 hardware integration: real isolated Engine HTTP, no throughput benchmark.

Requires an absolute RAMVI_TEST_DEADLINE. Run only in an explicitly authorized
hardware session; machine-busy checks belong to the outer owner-controlled runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Tools are repository-only; importing this file never imports torch/CUDA.
_ENGINE_ROOT = Path(__file__).resolve().parents[2]
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
_SERVICE_ROOT = _ENGINE_ROOT.parent / "service"
if _SERVICE_ROOT.is_dir() and str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from ramvi_engine.engine_process import EngineProcess  # noqa: E402
from ramvi_engine.model_loader import ModelConfig  # noqa: E402
from ramvi_engine.planning.automatic import PlanningOptions  # noqa: E402
from ramvi_engine.protocol import API_VERSION, check_api_version, validate_event  # noqa: E402

ECHO = {
    "messages": [{"role": "user", "content": "Reply with exactly this digit: 4"}],
    "max_tokens": 24,
    "temperature": 0,
    "stream": False,
}


def absolute_deadline(value: str | None) -> dt.datetime:
    if not value:
        raise ValueError("RAMVI_TEST_DEADLINE is required")
    deadline = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline requires an explicit timezone")
    deadline = deadline.astimezone(dt.UTC)
    if deadline <= dt.datetime.now(dt.UTC):
        raise ValueError("deadline already expired")
    return deadline


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
        ) -> None:
            raise urllib.error.HTTPError(req.full_url, code, "redirect disabled", headers, fp)

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


class RuntimeCheck:
    def __init__(
        self, model: str, maxt: int, out: Path, deadline: dt.datetime, require_service: bool
    ) -> None:
        self.deadline = deadline
        self.out = out.resolve()
        self.data = self.out.parent / (self.out.stem + "-worker-data")
        self.require_service = require_service
        self.worker = EngineProcess(
            ModelConfig(model, maxt=maxt, planning=PlanningOptions()),
            startup_timeout=min(180.0, max(0.1, self.remaining() - 5)),
            env={
                "RAMVI_DATA_DIR": str(self.data),
                "RAMVI_PREFILL": "legacy",
                "RAMVI_DRAFT_GATE": "0",
            },
        )
        self._opener = _no_redirect_opener()
        self._threads: list[threading.Thread] = []
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "complete": False,
            "scope": "section2_runtime_integration",
            "hardware_benchmark": False,
            "model": model,
            "maxt": maxt,
            "started": dt.datetime.now(dt.UTC).isoformat(),
            "deadline": deadline.isoformat(),
            "parent_pid": os.getpid(),
            "checks": [],
        }
        self._watchdog = threading.Timer(self.remaining(), self._hard_deadline)
        self._watchdog.daemon = True

    def remaining(self) -> float:
        return max(0.0, (self.deadline - dt.datetime.now(dt.UTC)).total_seconds())

    def budget(self, requested: float = 30.0) -> float:
        left = self.remaining()
        if left <= 1:
            raise TimeoutError("absolute hardware deadline")
        return min(requested, left - 0.5)

    def _hard_deadline(self) -> None:
        # Windows Job handles close on owner exit and kill the entire owned tree.
        # POSIX children have their own process group, so reap that group first.
        if sys.platform != "win32":
            supervisor = self.worker._supervisor
            child = supervisor._process if supervisor is not None else None
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        os._exit(124)

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.out.with_suffix(self.out.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.out)

    def passed(self, name: str, **facts: Any) -> None:
        self.report["checks"].append({"name": name, "passed": True, **facts})
        self.save()
        print(json.dumps({"check": name, "passed": True}), flush=True)

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        auth: bool = True,
        request_id: str | None = None,
        api_version: str | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if auth:
            # The check owns this random credential; it is never printed or saved.
            headers["Authorization"] = "Bearer " + self.worker._key
        if request_id is not None:
            headers["X-Request-ID"] = request_id
        if api_version is not None:
            headers["X-RAMVI-API-Version"] = api_version
        request = urllib.request.Request(
            self.worker.base_url + path,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers=headers,
        )
        try:
            response = self._opener.open(request, timeout=self.budget(timeout))
        except urllib.error.HTTPError as exc:
            with exc:
                body = exc.read(2**21 + 1)
            value = json.loads(body)
            return exc.code, value
        with response:
            body = response.read(2**21 + 1)
            status = response.status
        if len(body) > 2**21:
            raise AssertionError("integration response exceeded bounded size")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise AssertionError("HTTP result is not an object")
        return status, value

    def call(
        self, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        status, value = self.request(path, payload, timeout=timeout)
        assert status == 200, (path, status, value)
        return value

    def wait(self, predicate: Callable[[], Any], label: str, timeout: float = 15.0) -> Any:
        end = time.monotonic() + self.budget(timeout)
        while time.monotonic() < end:
            value = predicate()
            if value:
                return value
            time.sleep(min(0.01, max(0.0, end - time.monotonic())))
        raise TimeoutError(label)

    def thread(
        self, action: Callable[[], Any], name: str
    ) -> tuple[threading.Thread, dict[str, Any]]:
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["value"] = action()
            except BaseException as exc:
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    result["http_status"] = exc.code

        thread = threading.Thread(target=run, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()
        return thread, result

    def join(self, thread: threading.Thread) -> None:
        thread.join(self.budget(60.0))
        assert not thread.is_alive(), thread.name

    def echo(self, request_id: str | None = None) -> dict[str, Any]:
        status, body = self.request("/v1/chat/completions", dict(ECHO), request_id=request_id)
        assert status == 200, body
        assert body["choices"][0]["message"]["content"].strip() == "4", body
        metrics = body["ramvi_metrics"]
        assert metrics["outcome"] == "completed"
        assert metrics["output_tokens"] > 0
        return body

    def driver_memory(self) -> int:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=self.budget(10.0),
            check=True,
        )
        return int(result.stdout.strip())

    def verify_protocol_metrics(self) -> None:
        started = self.worker.start()
        assert started["state"] == "ready"
        assert "torch" not in sys.modules, "parent unexpectedly imported CUDA backend"
        health = self.call("/health")
        check_api_version(health["versions"]["api"])
        assert health["versions"]["api"] == API_VERSION
        assert self.request("/engine/status", auth=False)[0] == 401
        assert self.request("/engine/events", auth=False)[0] == 401
        assert self.request("/engine/metrics", auth=False)[0] == 401
        assert self.request("/engine/versions", api_version="99.0")[0] == 426
        assert self.call("/engine/versions")["versions"] == health["versions"]
        lifecycle = self.call("/engine/status")
        assert lifecycle["state"] == "ready" and lifecycle["loaded"]
        request_id = "runtime-echo-" + uuid.uuid4().hex
        body = self.echo(request_id)
        record = body["ramvi_metrics"]
        assert record["request_id"] == request_id
        memory = record["memory"]
        assert isinstance(memory["peak_ram_bytes"], int) and memory["peak_ram_bytes"] > 0
        assert isinstance(memory["peak_vram_allocated_bytes"], int)
        assert memory["peak_vram_allocated_bytes"] > 0
        assert memory["peak_vram_reserved_bytes"] >= memory["peak_vram_allocated_bytes"]
        assert record["ttft_ms"] is not None and record["engine_first_token_ms"] is not None
        status = self.call("/engine/metrics")
        assert status["metrics"]["last"]["generation_id"] == record["generation_id"]
        assert status["metrics"]["active"] == 0
        assert status["plan"]["layers"] == record["plan"]["layers"]
        events = self.call("/engine/events?request_id=" + urllib.parse.quote(request_id, safe=""))
        envelopes = [row["event"] for row in events["events"]]
        assert envelopes and envelopes[0]["type"] == "generation.started"
        assert envelopes[-1]["type"] == "generation.completed"
        for event in envelopes:
            validate_event(event)
            assert event["request_id"] == request_id
        assert [event["sequence"] for event in envelopes] == list(range(1, len(envelopes) + 1))
        assert [event["monotonic_ns"] for event in envelopes] == sorted(
            event["monotonic_ns"] for event in envelopes
        )
        journals = sorted((self.data / "planning").glob("session-*.jsonl"))
        assert journals, "automatic placement did not create its journal"
        rows = [
            json.loads(line)
            for file in journals
            for line in file.read_text(encoding="utf-8").splitlines()
        ]
        decisions = [row["payload"]["decision"] for row in rows if row["event"] == "decision"]
        assert decisions and decisions[-1]["mode"] == "auto"
        assert decisions[-1]["resources"]["kind"] == "estimate"
        actual = [
            row["payload"]["placement"] for row in rows if row["event"] in ("loaded", "actual")
        ]
        assert actual and actual[-1] == record["plan"]["layers"]
        self.passed(
            "isolated_http_auth_versions_events_metrics_plan",
            metrics=record,
            event_count=len(envelopes),
            planned_resident=decisions[-1]["resident"],
            actual_resident=record["plan"]["counts"].get("gpu", 0),
        )

    def service_classes(self) -> tuple[Any, Any, Any] | None:
        try:
            authority = importlib.import_module("ramvi_service.contracts.modules").Authority
            provider = importlib.import_module("ramvi_service.providers").LocalEngineProvider
            shared = importlib.import_module("ramvi_service.resources").SharedLLM
        except ModuleNotFoundError as exc:
            if self.require_service or exc.name != "ramvi_service":
                raise
            self.report["service_checks"] = "skipped_optional_package_absent"
            self.save()
            return None
        return authority, provider, shared

    def verify_shared_queue(self, classes: tuple[Any, Any, Any]) -> None:
        Authority, Provider, Shared = classes
        authority = Authority()
        a = authority.issue("runtime_chat", "integration", "active")
        b = authority.issue("runtime_character", "integration", "queued")
        c = authority.issue("runtime_agent", "integration", "fresh")
        provider = Provider(
            self.worker.base_url,
            os.path.basename(self.worker.config.model.rstrip("/\\")),
            api_key=self.worker._key,
            timeout=self.budget(120.0),
        )
        entered, proceed = threading.Event(), threading.Event()
        outer = self

        class FirstRequestGate:
            # Deterministic service scheduling barrier; every completed call still
            # executes the real HTTP provider and native model.
            capabilities = provider.capabilities
            first = True

            def complete(self, request: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
                if self.first:
                    self.first = False
                    entered.set()
                    if not proceed.wait(outer.budget(30.0)):
                        raise TimeoutError("service queue scheduling barrier")
                if cancel.is_set():
                    raise InterruptedError("owner stopped before HTTP")
                value = provider.complete(request, cancel)
                assert isinstance(value, dict)
                return value

            def cancel(self) -> None:
                proceed.set()
                provider.cancel()

            def unload(self) -> None:
                provider.unload()

        shared = Shared(authority, FirstRequestGate(), on_last_release=provider.unload)
        for owner in (a, b, c):
            shared.retain(owner)
        long_request = dict(ECHO)
        first: threading.Thread | None = None
        queued: threading.Thread | None = None
        try:
            first, first_result = self.thread(
                lambda: shared.complete(a, long_request, timeout=self.budget(120.0)),
                "stage2-owner-active",
            )
            assert entered.wait(self.budget(30.0)), "first owner never entered provider"
            queued, queued_result = self.thread(
                lambda: shared.complete(b, dict(ECHO), timeout=self.budget(120.0)),
                "stage2-owner-queued",
            )
            self.wait(lambda: shared.status()["queue"]["queued"] == 1, "second owner never queued")
            shared.release_owner(b)
            proceed.set()
            self.join(first)
            self.join(queued)
            assert "error" not in first_result, first_result
            assert first_result["value"]["choices"][0]["message"]["content"].strip() == "4"
            assert queued_result.get("error_type") == "InterruptedError", queued_result
            fresh = shared.complete(c, dict(ECHO), timeout=self.budget(120.0))
            assert fresh["choices"][0]["message"]["content"].strip() == "4"
            shared.release_owner(a)
            assert self.call("/engine/status")["loaded"], "releasing A unloaded C's retained model"
            shared.release_owner(c)
            assert not self.call("/engine/status")["loaded"]
            assert shared.status()["queue"]["leases"] == 0
            self.passed(
                "shared_llm_fifo_queued_owner_cancel_fresh_owner",
                owners_after=shared.status()["owners"],
            )
        finally:
            proceed.set()
            for owner in (b, a, c):
                shared.release_owner(owner)
            for thread in (first, queued):
                if thread is not None:
                    self.join(thread)
        self.call("/engine/load", {}, timeout=120.0)

    def verify_targeted_native_queue(self, classes: tuple[Any, Any, Any]) -> None:
        for attempt in range(3):
            if self._targeted_native_attempt(classes, attempt):
                return
        raise AssertionError(
            "could not observe a real queued native request within bounded retries"
        )

    def _targeted_native_attempt(self, classes: tuple[Any, Any, Any], attempt: int) -> bool:
        Authority, Provider, Shared = classes
        authority = Authority()
        old = authority.issue("runtime_character", "integration", "old")
        fresh = authority.issue("runtime_agent", "integration", "new")
        provider = Provider(
            self.worker.base_url,
            os.path.basename(self.worker.config.model.rstrip("/\\")),
            api_key=self.worker._key,
            timeout=self.budget(120.0),
        )
        shared = Shared(authority, provider)
        shared.retain(old)
        shared.retain(fresh)
        unrelated_id = "unrelated-" + uuid.uuid4().hex
        size = min(
            self.worker.config.maxt - 96, int(self.worker.config.maxt * (0.6 + 0.1 * attempt))
        )
        unrelated: threading.Thread | None = None
        pending: threading.Thread | None = None
        try:
            unrelated, unrelated_result = self.thread(
                lambda: self.request(
                    "/v1/completions",
                    {
                        "prompt": [1000 + attempt * 31] * size,
                        "max_tokens": 64,
                        "temperature": 0,
                        "stream": False,
                    },
                    request_id=unrelated_id,
                    timeout=120.0,
                ),
                "stage2-unrelated-active",
            )
            self.wait(
                lambda: self.call("/engine/status", timeout=2.0).get("request_id") == unrelated_id,
                "unrelated native request did not become active",
            )
            pending, pending_result = self.thread(
                lambda: shared.complete(old, dict(ECHO), timeout=self.budget(120.0)),
                "stage2-native-queued",
            )

            def queued_behind_other() -> Any:
                if unrelated is not None and not unrelated.is_alive():
                    return "window_closed"
                with provider._lock:
                    request_id = provider._active_request
                if request_id is None:
                    return None
                events = self.call("/engine/events?request_id=" + request_id, timeout=2.0)
                state = self.call("/engine/status", timeout=2.0)
                if events["events"] and state.get("request_id") == unrelated_id:
                    return request_id
                return None

            target = self.wait(
                queued_behind_other, "provider did not queue behind unrelated request"
            )
            if target == "window_closed":
                shared.release_owner(old)
                self.join(unrelated)
                self.join(pending)
                return False
            shared.release_owner(old)
            self.join(unrelated)
            self.join(pending)
            assert "error" not in unrelated_result, unrelated_result
            unrelated_status, unrelated_body = unrelated_result["value"]
            assert (
                unrelated_status == 200
                and unrelated_body["ramvi_metrics"]["outcome"] == "completed"
            ), unrelated_body
            assert pending_result.get("error_type") in ("InterruptedError", "HTTPError"), (
                pending_result
            )
            if pending_result.get("error_type") == "HTTPError":
                assert pending_result.get("http_status") == 409, pending_result
            response = shared.complete(fresh, dict(ECHO), timeout=self.budget(120.0))
            assert response["choices"][0]["message"]["content"].strip() == "4"
            assert self.call("/engine/status")["state"] == "ready"
            self.passed(
                "targeted_cancel_queued_native_request_preserves_other_and_fresh_owner",
                attempt=attempt + 1,
            )
            return True
        finally:
            shared.release_owner(old)
            shared.release_owner(fresh)
            for thread in (unrelated, pending):
                if thread is not None:
                    self.join(thread)
        return False

    def verify_restart_and_worker_crash(self) -> None:
        print("STAGE2_PHASE begin_lifecycle", flush=True)
        original_instance = self.call("/health")["instance_id"]
        print("STAGE2_PHASE unload", flush=True)
        self.call("/engine/unload", {})
        assert not self.call("/engine/status")["loaded"]
        assert self.request("/v1/chat/completions", dict(ECHO))[0] == 503
        self.call("/engine/load", {}, timeout=120.0)
        self.echo()
        assert self.call("/health")["instance_id"] == original_instance
        print("STAGE2_PHASE http_restart", flush=True)
        self.call("/engine/restart", {}, timeout=120.0)
        self.echo()
        assert self.call("/health")["instance_id"] == original_instance
        print("STAGE2_PHASE force_crash", flush=True)
        supervisor = self.worker._supervisor
        assert supervisor is not None
        if sys.platform == "win32":
            assert supervisor._job is not None
            supervisor._job.close()  # forcible termination of this exact owned worker tree
        else:
            child = supervisor._process
            assert child is not None
            os.killpg(child.pid, signal.SIGKILL)
        self.wait(lambda: self.worker.status()["state"] == "failed", "worker crash not diagnosed")
        assert os.getpid() == self.report["parent_pid"], "service parent was replaced"
        diagnosed = self.worker.status()
        assert diagnosed["restart_available"]
        print("STAGE2_PHASE process_restart", flush=True)
        self.worker.restart()
        self.echo()
        assert self.call("/health")["instance_id"] != original_instance
        self.passed(
            "native_worker_crash_parent_survives_and_restarts",
            diagnosed=diagnosed,
            parent_alive=True,
            new_worker=self.worker.status()["process"],
        )

    def _driver_back_to(self, baseline: int) -> dict[str, int] | None:
        value = self.driver_memory()
        self.report.setdefault("driver_reclaim_samples", []).append(
            {"monotonic_s": time.monotonic(), "mib": value}
        )
        return {"mib": value} if value <= baseline else None

    def run(self) -> dict[str, Any]:
        self.save()
        self._watchdog.start()
        failure: BaseException | None = None
        try:
            self.report["driver_before_mib"] = self.driver_memory()
            self.verify_protocol_metrics()
            classes = self.service_classes()
            if classes is not None:
                self.verify_shared_queue(classes)
                self.verify_targeted_native_queue(classes)
            self.verify_restart_and_worker_crash()
        except BaseException as exc:
            failure = exc
            self.report["error_type"] = type(exc).__name__
            self.report["error"] = str(exc)
            self.report["diagnostics"] = self.worker.diagnostics()
        finally:
            try:
                self.worker.close()
                for thread in self._threads:
                    thread.join(timeout=3)
                leftovers = [thread.name for thread in self._threads if thread.is_alive()]
                assert not leftovers, leftovers
                assert self.worker.status()["state"] == "stopped"
                baseline = self.report.get("driver_before_mib")
                if isinstance(baseline, int):
                    after = self.wait(
                        lambda: self._driver_back_to(baseline),
                        "driver memory did not return after child exit",
                        timeout=60.0,
                    )
                    self.report["driver_after_mib"] = after["mib"]
            except BaseException as exc:
                self.report["cleanup_error"] = type(exc).__name__ + ": " + str(exc)
                failure = failure or exc
            self._watchdog.cancel()
            self._watchdog.join(timeout=1)
            self.report["finished"] = dt.datetime.now(dt.UTC).isoformat()
            self.report["complete"] = failure is None
            self.report["worker_after"] = self.worker.status()
            self.save()
        if failure is not None:
            raise failure
        print("STAGE2_RUNTIME_COMPLETE", flush=True)
        return self.report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--maxt", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--require-service", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error("timeout must be finite and within 0..3600")
    if not 1024 <= args.maxt <= 16384:
        parser.error("maxt must be within 1024..16384 for a real queued-prefill check")
    try:
        deadline = absolute_deadline(os.environ.get("RAMVI_TEST_DEADLINE"))
    except ValueError as exc:
        parser.error(str(exc))
    deadline = min(deadline, dt.datetime.now(dt.UTC) + dt.timedelta(seconds=args.timeout))
    RuntimeCheck(args.model, args.maxt, args.out, deadline, args.require_service).run()


if __name__ == "__main__":
    main()
