# ruff: noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Portable Windows workspace and Job Object acceptance")
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
ROOT = args.work.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE = args.bundle.resolve()
SOURCE = args.source.resolve()
sys.path[:0] = [str(SOURCE / "service"), str(SOURCE / "storage"), str(BUNDLE / "ramvi" / "engine")]
from ramvi_service.agent.process import ProcessLimits, run_process
from ramvi_service.agent.runtime import MemoryJournal
from ramvi_service.agent.toolbox import build_tools
from ramvi_service.agent.workspace import Permissions, Workspace

project = ROOT / "safety_workspace"
project.mkdir(exist_ok=True)
(project / ".env").write_text("PRIVATE_DO_NOT_LEAK_100032")
(ROOT / "outside.txt").write_text("PRIVATE_OUTSIDE_913")
ws = Workspace(project, Permissions(write=True, terminal=True))
tools = build_tools(ws)
journal = MemoryJournal()
out = []
for name, args in [
    ("read_file", {"path": "../outside.txt"}),
    ("read_file", {"path": ".env"}),
    ("write_file", {"path": "../outside.txt", "content": "must not overwrite"}),
    ("run_command", {"command": 'python -c "print(123)"'}),
]:
    result = tools.execute(name, args, journal)
    assert not result.ok and "PRIVATE_" not in result.content
    out.append(
        {
            "test": name,
            "args": {k: v for k, v in args.items() if k == "path"},
            "error": result.error,
            "ok": True,
        }
    )
assert (ROOT / "outside.txt").read_text() == "PRIVATE_OUTSIDE_913"
assert all(e["permission"] == "denied" or e["result"]["ok"] is False for e in journal.entries)
started = time.monotonic()
result = run_process(
    [sys.executable, "-c", "import time; time.sleep(30)"], ROOT, ProcessLimits(timeout=0.4)
)
assert result.timed_out and result.exit_code is None
out.append(
    {"test": "windows_job_timeout", "ok": True, "seconds": round(time.monotonic() - started, 2)}
)
from ramvi_service.agent.gateway import AgentGateway
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority
from ramvi_service.modules.host import ModuleHost


class FixedContext:
    def capture(self, *args):
        return {"text": "private-test", "structure": {"ok": True}}


authority = Authority()
owner = authority.issue("test", "user", "session")
grants = GrantStore(authority)
context = ContextProvider(authority, grants, FixedContext(), min_interval=0)
for permission in ("context.ui", "egress.text"):
    grants.allow(owner, permission, "app")
snapshot = context.capture(owner, "app", "window")
grants.release_owner(owner)
for permission in ("context.ui", "egress.text"):
    grants.allow(owner, permission, "app")
try:
    context.authorize_egress(owner, snapshot)
except PermissionError:
    pass
else:
    raise AssertionError("released snapshot revived")
gateway = AgentGateway(authority, grants)
host = ModuleHost(ROOT / "late_host", agent=gateway)
assert host.authority is authority and host.grants is grants
host.close()
gateway.close()
out.extend(
    [
        {"test": "released_snapshot_not_revived", "ok": True},
        {"test": "injected_agent_host_identity", "ok": True},
    ]
)
(ROOT / "safety_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
