# ruff: noqa: E402
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description="Native UIA acceptance on an isolated WPF fixture")
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
ROOT = args.work.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE = args.bundle.resolve()
SOURCE = args.source.resolve()
sys.path[:0] = [str(SOURCE / "service"), str(SOURCE / "storage"), str(BUNDLE / "ramvi" / "engine")]
from ramvi_service.agent.approval import ApprovalGate, Decision
from ramvi_service.agent.changes import ChangeStore
from ramvi_service.agent.runtime import MemoryJournal
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.tools import ToolAccess
from ramvi_service.agent.ui import UIControl
from ramvi_service.agent.workspace import Workspace
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority, ProviderCapabilities
from ramvi_service.system_context import SystemContextAdapter

results = []


def record(name, **data):
    item = {"test": name, **data}
    results.append(item)
    (ROOT / "ui_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(item), flush=True)


def read(name):
    for _ in range(50):
        try:
            return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            time.sleep(0.1)
    raise TimeoutError(name)


def focus(where):
    (ROOT / "focus.txt").write_text(where)
    time.sleep(0.5)


def main():
    for filename in ("fixture_ready.json", "fixture_state.json", "close.txt"):
        (ROOT / filename).unlink(missing_ok=True)
    fixture = subprocess.Popen(
        [
            "powershell.exe",
            "-STA",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path(__file__).with_name("fixture.ps1")),
            "-Root",
            str(ROOT),
        ],
        stdout=open(ROOT / "fixture.log", "w"),
        stderr=subprocess.STDOUT,
    )
    (ROOT / "pids.txt").write_text(f"{os.getpid()}\n{fixture.pid}\n")
    try:
        target = read("fixture_ready.json")
        time.sleep(0.5)
        adapter = SystemContextAdapter()
        current = adapter.current_target()
        assert current == target, (current, target)
        authority = Authority()
        owner = authority.issue("character", "test_user", "ui_test")
        grants = GrantStore(authority)
        context = ContextProvider(
            authority, grants, adapter, min_interval=0, max_bytes_per_minute=32 * 2**20
        )
        app, window = target["application"], target["window_id"]
        for permission in (
            "context.ui",
            "context.screen",
            "agent.control",
            "egress.text",
            "egress.image",
        ):
            grants.allow(owner, permission, app)
        snap = context.capture_best(owner, app, window)
        nodes = snap.structure["elements"]
        assert snap.text == "reference initial", "selected text must be returned by UI Automation"
        record(
            "ui_capture",
            elements=len(nodes),
            fingerprint=bool(snap.structure["fingerprint"]),
            nodes=nodes,
        )
        field = next(e["id"] for e in nodes if e["name"] == "Agent input field")
        button = next(e["id"] for e in nodes if e["name"] == "Agent apply button")
        adapter.act(
            app,
            window,
            field,
            "set_value",
            "reference verified",
            snap.structure["fingerprint"],
            threading.Event(),
        )
        time.sleep(0.3)
        assert read("fixture_state.json")["text"] == "reference verified"
        record("solvable_reference_set_value", ok=True)
        snap = context.capture_best(owner, app, window)
        adapter.act(
            app, window, button, "invoke", None, snap.structure["fingerprint"], threading.Event()
        )
        time.sleep(0.3)
        assert read("fixture_state.json")["clicks"] == 1
        record("solvable_reference_invoke", ok=True)
        screen = context.capture(
            owner, app, window, kind="screen", capabilities=ProviderCapabilities(image_input=True)
        )
        png = base64.b64decode(screen.image.split(",", 1)[1])
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 500
        (ROOT / "window_capture.png").write_bytes(png)
        record("window_screen", png_bytes=len(png), ok=True)
        ws = Workspace(ROOT / "ui_workspace")
        access = ToolAccess()
        answers = []
        gate = ApprovalGate(
            lambda req: answers.pop(0)(req), ChangeStore(ROOT / "ui_changes", ws), access
        )
        tools = build_tools(ws, access, gate=gate)
        control = UIControl(context, owner, app, window, adapter, max_age=30)
        control.register(tools)
        journal = MemoryJournal()

        def action(value):
            snap = json.loads(tools.execute("ui_context", {}, journal).content)
            return {
                "snapshot_id": snap["snapshot_id"],
                "element_id": field,
                "action": "set_value",
                "value": value,
                "plan": "Fill the isolated test field",
            }

        for run, value in enumerate(("fresh 17", "fresh 913", "fresh 100032")):
            args = action(value)
            answers.append(lambda req: Decision("approve"))
            result = tools.execute("ui_action", args, journal)
            assert result.ok, result.content
            time.sleep(0.2)
            assert read("fixture_state.json")["text"] == value
            record("approved_fresh_" + str(run), ok=True)
        args = action("must not be applied")
        before = read("fixture_state.json")
        answers.append(lambda req: Decision("deny", reason="keep"))
        assert not tools.execute("ui_action", args, journal).ok
        assert read("fixture_state.json") == before
        record("deny_keeps_state", ok=True)
        args = action("wrong window must not change")

        def switch(req):
            focus("other")
            return Decision("approve")

        answers.append(switch)
        assert not tools.execute("ui_action", args, journal).ok
        assert read("fixture_state.json") == before
        record("focus_changed_during_approval", ok=True)
        focus("main")
        args = action("changed target must not overwrite")

        def mutate(req):
            (ROOT / "fixture_mutate.txt").write_text("changed by fixture")
            time.sleep(0.5)
            return Decision("approve")

        answers.append(mutate)
        assert not tools.execute("ui_action", args, journal).ok
        assert read("fixture_state.json")["text"] == "changed by fixture"
        record("value_changed_during_approval", ok=True)
        before = read("fixture_state.json")
        args = action("revoked must not change")

        def revoke(req):
            grants.revoke(owner, "agent.control", app)
            return Decision("approve")

        answers.append(revoke)
        assert not tools.execute("ui_action", args, journal).ok
        assert read("fixture_state.json") == before
        record("control_revoked_during_approval", ok=True)
        grants.allow(owner, "agent.control", app)
        args = action("expired must not change")

        def expire(req):
            control.max_age = -1
            return Decision("approve")

        answers.append(expire)
        assert not tools.execute("ui_action", args, journal).ok
        assert read("fixture_state.json") == before
        record("expired_during_approval", ok=True)
    finally:
        (ROOT / "close.txt").write_text("close")
        try:
            fixture.wait(4)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(fixture.pid), "/T", "/F"], capture_output=True, timeout=10
            )


try:
    main()
except BaseException as error:
    record(
        "failure", error=type(error).__name__, detail=str(error), traceback=traceback.format_exc()
    )
    raise
