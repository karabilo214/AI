"""6.12: UI-действия проходят schema → journal → approval → повторную проверку цели."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from ramvi_service.agent.approval import Decision
from ramvi_service.agent.runtime import AgentRuntime, MemoryJournal, ModelReply
from ramvi_service.agent.ui import UIControl
from ramvi_service.context import ContextProvider, GrantStore
from ramvi_service.contracts.modules import Authority
from ramvi_storage.database import Database
from test_agent_runtime import Env, Script, call, done, reply
from test_agent_runtime import db as db  # noqa: F401

PRIVATE = "private selected 913"
TYPED = "private new 100032"


class Adapter:
    def __init__(self) -> None:
        self.window = "window"
        self.fingerprint = "version-1"
        self.applied: list[tuple[str, str | None]] = []

    def capture(self, app: str, window: str, kind: str, cancel: Any) -> dict[str, Any]:
        return {
            "text": PRIVATE,
            "structure": {
                "fingerprint": self.fingerprint,
                "elements": [{"id": "entry", "name": PRIVATE}],
            },
        }

    def act(
        self,
        app: str,
        window: str,
        element: str,
        action: str,
        value: Any,
        fingerprint: str,
        cancel: Any,
    ) -> dict[str, Any]:
        if cancel.is_set() or window != self.window or fingerprint != self.fingerprint:
            raise ValueError("stale target")
        self.applied.append((action, value))
        return {"applied": True}


class Setup:
    def __init__(self, tmp: Path, db: Database, external: bool = False) -> None:
        self.authority = Authority()
        self.owner = self.authority.issue("character", "user", "session")
        self.grants = GrantStore(self.authority)
        self.adapter = Adapter()
        self.context = ContextProvider(self.authority, self.grants, self.adapter, min_interval=0)
        self.control = UIControl(
            self.context, self.owner, "app", "window", self.adapter, external=external
        )
        self.env = Env(tmp, db, Script(done("ok")))
        self.control.register(self.env.tools)
        self.journal = MemoryJournal()

    def grant(self, control: bool = True) -> None:
        self.grants.allow(self.owner, "context.ui", "app")
        if control:
            self.grants.allow(self.owner, "agent.control", "app")

    def args(self) -> dict[str, Any]:
        result = self.env.tools.execute("ui_context", {}, self.journal)
        assert result.ok
        snapshot = json.loads(result.content)
        return {
            "snapshot_id": snapshot["snapshot_id"],
            "element_id": "entry",
            "action": "set_value",
            "value": TYPED,
            "plan": "Fill selected field",
        }


def test_solvable_action_has_reviewable_proposal_and_single_use_snapshot(
    tmp_path: Path, db: Database
) -> None:
    setup = Setup(tmp_path, db)
    setup.grant()
    args = setup.args()
    seen = []

    def ask(request: Any) -> Decision:
        assert setup.adapter.applied == []
        assert request.kind == "control" and request.application == "app"
        assert request.value == TYPED and request.element_id == "entry"
        assert request.plan == "Fill selected field" and PRIVATE in request.title
        seen.append(request)
        return Decision("approve")

    setup.env.gate.ask = ask
    result = setup.env.tools.execute("ui_action", args, setup.journal)
    assert result.ok and setup.adapter.applied == [("set_value", TYPED)] and len(seen) == 1
    assert not setup.env.tools.execute("ui_action", args, setup.journal).ok
    assert len(setup.adapter.applied) == 1


@pytest.mark.parametrize(
    "change",
    ["window", "tree", "revoke_control", "revoke_read", "age", "snapshot", "cancel", "deny"],
)
def test_change_while_waiting_for_approval_does_not_act(
    tmp_path: Path, db: Database, change: str
) -> None:
    setup = Setup(tmp_path, db)
    setup.grant()
    args = setup.args()
    cancel = threading.Event()
    setup.env.gate.cancel = cancel

    def ask(request: Any) -> Decision:
        if change == "window":
            setup.adapter.window = "other"
        elif change == "tree":
            setup.adapter.fingerprint = "version-2"
        elif change == "revoke_control":
            setup.grants.revoke(setup.owner, "agent.control", "app")
        elif change == "revoke_read":
            setup.grants.revoke(setup.owner, "context.ui", "app")
        elif change == "age":
            setup.control.max_age = -1
        elif change == "snapshot":
            setup.context.capture(setup.owner, "app", "window")
        elif change == "cancel":
            cancel.set()
        return Decision("deny" if change == "deny" else "approve")

    setup.env.gate.ask = ask
    if change == "cancel":
        from ramvi_service.agent.tools import StopRequested

        with pytest.raises(StopRequested):
            setup.env.tools.execute("ui_action", args, setup.journal)
    else:
        assert not setup.env.tools.execute("ui_action", args, setup.journal).ok
    assert setup.adapter.applied == []


def test_context_access_does_not_grant_control_and_disabled_tool_cannot_act(
    tmp_path: Path, db: Database
) -> None:
    setup = Setup(tmp_path, db)
    setup.grant(control=False)
    args = setup.args()
    assert "ui_action" not in [t["function"]["name"] for t in setup.env.tools.openai_tools()]
    result = setup.env.tools.execute("ui_action", args, setup.journal)
    assert not result.ok and setup.adapter.applied == []
    setup.grants.allow(setup.owner, "agent.control", "app")
    from dataclasses import replace

    setup.env.tools.access = replace(setup.env.tools.access, disabled=frozenset({"ui_action"}))
    assert not setup.env.tools.execute("ui_action", args, setup.journal).ok
    assert setup.adapter.applied == []


def test_private_ui_content_not_in_persistent_journals(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    setup.grant()

    def choose(messages: Any) -> ModelReply:
        args = json.loads(messages[-1]["content"])
        return reply(
            PRIVATE,
            call(
                "ui_action",
                snapshot_id=args["snapshot_id"],
                element_id="entry",
                action="set_value",
                value=TYPED,
                plan=PRIVATE,
            ),
        )

    class Strategy:
        def decide(self, messages: Any, tools: Any, cancel: Any, max_tokens: Any) -> ModelReply:
            if messages[-1]["role"] == "user":
                return reply(PRIVATE, call("ui_context"))
            if '"snapshot_id"' in messages[-1]["content"]:
                return choose(messages)
            return done(TYPED)

    setup.env.gate.ask = lambda request: Decision("approve")
    result = AgentRuntime(Strategy(), setup.env.tools, setup.env.runs).run("Fill field")
    assert result.status == "done" and setup.adapter.applied == [("set_value", TYPED)]
    stored = json.dumps(
        {
            "events": setup.env.runs.events(result.run_id),
            "calls": [vars(c) for c in setup.env.runs.tool_calls(result.run_id)],
            "run": vars(setup.env.runs.get(result.run_id)),
        },
        default=str,
    )
    assert PRIVATE not in stored and TYPED not in stored
    assert "ui_action" in stored and "entry" in stored and "approved" in stored


def test_external_model_needs_separate_egress_and_revocation_interrupts_it(
    tmp_path: Path, db: Database
) -> None:
    setup = Setup(tmp_path, db, external=True)
    setup.grant()
    setup.args()
    with pytest.raises(PermissionError):
        with setup.env.tools.model_context(threading.Event()):
            raise AssertionError("must not reach external provider")
    setup.grants.allow(setup.owner, "egress.text", "app")
    with pytest.raises(InterruptedError):
        with setup.env.tools.model_context(threading.Event()) as cancel:
            setup.grants.revoke(setup.owner, "egress.text", "app")
            assert cancel.is_set() and cancel.wait(0)


def test_read_reallow_cannot_send_revoked_history(tmp_path: Path, db: Database) -> None:
    setup = Setup(tmp_path, db)
    setup.grant()
    setup.args()
    setup.grants.revoke(setup.owner, "context.ui", "app")
    setup.grants.allow(setup.owner, "context.ui", "app")
    with pytest.raises(PermissionError):
        with setup.env.tools.model_context(threading.Event()):
            raise AssertionError("old history must not be reused")


def test_ui_confirmation_and_file_permissions_are_independent(tmp_path: Path, db: Database) -> None:
    from dataclasses import replace

    from ramvi_service.agent.workspace import Permissions

    setup = Setup(tmp_path, db)
    setup.grant()
    # UI Control is a separate grant; file read/write/terminal rights are absent.
    setup.env.ws.permissions = Permissions(read=False, write=False, terminal=False)
    args = setup.args()
    setup.env.gate.access = replace(
        setup.env.gate.access, confirm_writes=False, confirm_commands=False
    )
    asked = []

    def deny(request: Any) -> Decision:
        asked.append(request.kind)
        return Decision("deny")

    setup.env.gate.ask = deny
    assert not setup.env.tools.execute("ui_action", args, setup.journal).ok
    assert asked == ["control"] and setup.adapter.applied == []
    setup.env.gate.access = replace(setup.env.gate.access, confirm_controls=False)
    assert setup.env.tools.execute("ui_action", args, setup.journal).ok
    assert setup.adapter.applied == [("set_value", TYPED)]
    assert asked == ["control"]
