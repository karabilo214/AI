"""6.12: UI-предложения через общий ToolRunner и ApprovalGate.

Только конкретные UIA-элементы выбранного окна; не глобальные клавиши/координаты.
Контекст/грант/фокус/свежесть проверяются повторно после пользовательского approve.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from ramvi_service.agent.tools import ToolError, ToolResult, ToolRunner, ToolSpec
from ramvi_service.cancellation import LinkedCancellation
from ramvi_service.context import ContextProvider
from ramvi_service.contracts.modules import Owner


class ControlAdapter(Protocol):
    def act(
        self,
        application: str,
        window: str,
        element_id: str,
        action: str,
        value: str | None,
        fingerprint: str,
        cancel: threading.Event,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UIProposal:
    application: str
    window_id: str
    snapshot_id: str
    element_id: str
    action: str
    plan: str = field(repr=False)
    value: str | None = field(repr=False)
    target_label: str = field(repr=False)
    apply: Callable[[threading.Event | None], ToolResult] = field(repr=False, compare=False)


class UIControl:
    def __init__(
        self,
        context: ContextProvider,
        owner: Owner,
        application: str,
        window_id: str,
        adapter: ControlAdapter,
        *,
        max_age: float | None = 30,
        external: bool = False,
    ) -> None:
        self.context, self.owner = context, owner
        self.external = external
        self._observed_epoch: int | None = None
        self.application, self.window_id, self.adapter, self.max_age = (
            application,
            window_id,
            adapter,
            max_age,
        )

    def require(self) -> None:
        self.context.grants.require(self.owner, "agent.control", self.application)

    def propose(self, args: dict[str, Any]) -> ToolResult:
        self.require()
        snapshot = self.context.snapshot(self.owner, args["snapshot_id"])
        self.context.validate_snapshot(self.owner, snapshot, max_age=self.max_age)
        if snapshot.application != self.application or snapshot.window_id != self.window_id:
            raise ToolError("agent.ui.target_changed")
        structure = snapshot.structure or {}
        elements = structure.get("elements", [])
        if not any(isinstance(e, dict) and e.get("id") == args["element_id"] for e in elements):
            raise ToolError("agent.ui.element_missing")
        fingerprint = structure.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ToolError("agent.ui.no_fingerprint")
        action, value = args["action"], args.get("value")
        if (action == "set_value") != (value is not None):
            raise ToolError("agent.tool.invalid_args")

        def apply(cancel: threading.Event | None) -> ToolResult:
            self.require()
            self.context.validate_snapshot(self.owner, snapshot, max_age=self.max_age)
            rights = [("agent.control", self.application), ("context.ui", self.application)]
            revoked = self.context.grants.watch(self.owner, rights)

            linked = LinkedCancellation(revoked, *([cancel] if cancel is not None else []))
            try:
                if linked.is_set():
                    raise InterruptedError("UI action cancelled")
                self.context.validate_snapshot(self.owner, snapshot, max_age=self.max_age)
                # System adapter rechecks native window, fingerprint and element immediately
                # before InvokePattern/ValuePattern. No global input goes to another app.
                outcome = self.adapter.act(
                    self.application,
                    self.window_id,
                    args["element_id"],
                    action,
                    value,
                    fingerprint,
                    linked,
                )
                if outcome.get("applied") is not True:
                    raise ValueError("UI action outcome is unknown")
                if linked.is_set():
                    raise InterruptedError("UI action outcome uncertain after cancellation")
                # A snapshot is single-use after any action, including failed/uncertain ones.
                return ToolResult(
                    True, "UI action completed; capture fresh UI Context before the next action."
                )
            except (
                PermissionError,
                TimeoutError,
                NotImplementedError,
                InterruptedError,
                ValueError,
            ):
                raise ToolError("agent.ui.not_applied_or_unknown") from None
            finally:
                self.context.release_snapshot(self.owner, snapshot.snapshot_id)
                self.context.grants.unwatch(revoked)

        proposal = UIProposal(
            self.application,
            self.window_id,
            snapshot.snapshot_id,
            args["element_id"],
            action,
            args["plan"],
            value,
            next(
                str(e.get("name", ""))
                for e in elements
                if isinstance(e, dict) and e.get("id") == args["element_id"]
            ),
            apply,
        )
        return ToolResult(True, "Proposed UI action; waiting for approval.", {"proposal": proposal})

    @contextmanager
    def model_context(self, cancel: threading.Event) -> Iterator[threading.Event]:
        if self._observed_epoch is None:
            yield cancel
            return
        rights = [("context.ui", self.application)]
        if self.external:
            rights.append(("egress.text", self.application))
        revoked = self.context.grants.watch(self.owner, rights)
        linked = LinkedCancellation(cancel, revoked)
        try:
            if (
                self.context.grants.epoch(self.owner, "context.ui", self.application)
                != self._observed_epoch
            ):
                raise PermissionError("history contains revoked context; start a new task")
            if linked.is_set():
                raise InterruptedError("UI context revoked")
            yield linked
            if linked.is_set():
                raise InterruptedError("UI context revoked")
        finally:
            self.context.grants.unwatch(revoked)

    def register(self, runner: ToolRunner) -> None:
        runner.private = True
        runner.model_context = self.model_context
        runner.authorize_with("agent.control", self.require)
        runner.authorize_with(
            "context.ui",
            lambda: self.context.grants.require(self.owner, "context.ui", self.application),
        )

        def capture(args: dict[str, Any]) -> ToolResult:
            snapshot = self.context.capture_best(self.owner, self.application, self.window_id)
            epoch = self.context.grants.epoch(self.owner, "context.ui", self.application)
            self.context.validate_snapshot(self.owner, snapshot)
            if self._observed_epoch is None:
                self._observed_epoch = epoch
            import json
            from dataclasses import asdict

            return ToolResult(True, json.dumps(asdict(snapshot), ensure_ascii=False))

        runner.register(
            ToolSpec(
                "ui_context",
                "Read fresh structured UI Context of the user-selected window. "
                "Use element IDs and snapshot_id in ui_action; screen/coordinates are not actions.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                "context.ui",
            ),
            capture,
        )
        runner.register(
            ToolSpec(
                "ui_action",
                "Propose an action on an element in a fresh UI Context snapshot. "
                "Write a short plan before calling. Approval is required by default. "
                "After any action capture fresh context. Never retry an uncertain action blindly.",
                {
                    "type": "object",
                    "properties": {
                        "snapshot_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "element_id": {"type": "string", "minLength": 1, "maxLength": 512},
                        "action": {"type": "string", "enum": ["invoke", "set_value"]},
                        "value": {"type": "string", "maxLength": 8192},
                        "plan": {"type": "string", "minLength": 1, "maxLength": 2000},
                    },
                    "required": ["snapshot_id", "element_id", "action", "plan"],
                    "additionalProperties": False,
                },
                "agent.control",
                effect="control",
                risk="critical",
            ),
            self.propose,
        )
