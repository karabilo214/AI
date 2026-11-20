"""CHAR-21: an explicit owner task handed to Agent through the public Agent Gateway.

Character never imports Agent and never grants itself rights. The workspace binding
(folder, permissions, tools, model) is made by trusted host code when the owner turns
delegation on; Character only proposes the goal. Agent runs it with its own ToolRunner,
confirmations and journal. Character follows the bounded event feed, reports the result
and can cancel only its own task. Without Agent the ordinary dialogue goes on.
"""

import copy
import threading
import time
from collections.abc import Callable
from typing import Any

GOAL_LIMIT = 4096
RESULT_LIMIT = 2048
POLL = 0.25
# Gateway errors that mean "Agent is not there for Character right now".
UNAVAILABLE = (NotImplementedError, PermissionError)


class Delegation:
    def __init__(
        self,
        services: Any,
        on_finished: Callable[[dict[str, Any]], None],
        *,
        poll: float = POLL,
    ) -> None:
        self.services, self.on_finished, self.poll = services, on_finished, poll
        self.lock = threading.RLock()
        self.binding: str | None = None
        self.task: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()

    def bind(self, binding_id: str | None) -> None:
        """Trusted UI hands over the opaque binding made by the host (None turns it off)."""
        if binding_id is not None and (
            not isinstance(binding_id, str) or not 1 <= len(binding_id) <= 128
        ):
            raise ValueError("invalid agent binding")
        with self.lock:
            self.binding = binding_id

    def running(self) -> bool:
        with self.lock:
            return self.task is not None and self.task["state"] == "running"

    def start(self, goal: str) -> dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip() or len(goal) > GOAL_LIMIT:
            raise ValueError("invalid agent goal")
        with self.lock:
            if self.services is None or self.binding is None:
                return self._refuse("agent_off")
            if self.running():
                return self._refuse("agent_busy")
            try:
                created = self.services.call(
                    "agent.create", {"goal": goal.strip(), "binding_id": self.binding}
                )
            except UNAVAILABLE:
                # The binding was revoked (settings changed, Agent removed or restarted).
                self.binding = None
                return self._refuse("agent_unavailable")
            except RuntimeError:
                return self._refuse("agent_busy")
            self.task = {
                "task_id": str(created["task_id"]),
                "goal": goal.strip(),
                "state": "running",
                "phase": "running",
                "steps": 0,
                "result": "",
                "stop_reason": "",
                "error": None,
                "cursor": 0,
                "started": time.monotonic(),
            }
            self.stopping.clear()
            self.thread = threading.Thread(
                target=self._follow,
                args=(self.task["task_id"],),
                name="character-agent",
                daemon=True,
            )
            self.thread.start()
            return self.view()

    def _refuse(self, code: str) -> dict[str, Any]:
        view = self.view()
        view["error"] = code
        return view

    def cancel(self) -> dict[str, Any]:
        """Stop Character's own delegated task; other tasks are not reachable from here."""
        with self.lock:
            task = self.task
        if task is not None and task["state"] == "running":
            try:
                self.services.call("agent.cancel", {"task_id": task["task_id"]})
            except UNAVAILABLE:
                with self.lock:
                    task["state"], task["error"] = "cancelled", "agent_unavailable"
        return self.view()

    def view(self) -> dict[str, Any]:
        with self.lock:
            task = copy.deepcopy(self.task)
            if task is not None:
                task.pop("started", None)
                task.pop("cursor", None)
            return {"enabled": self.binding is not None, "task": task, "error": None}

    def stop(self, wait: float = 5) -> None:
        """Module deactivation: cancel the own task and do not leave the follower behind."""
        self.stopping.set()  # first: a result after shutdown is not written into the dialogue
        self.cancel()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(wait)

    def _follow(self, task_id: str) -> None:
        finished: dict[str, Any] | None = None
        try:
            while not self.stopping.is_set():
                with self.lock:
                    after = self.task["cursor"] if self.task else 0
                feed = self.services.call("agent.events", {"task_id": task_id, "after": after})
                status = self.services.call("agent.status", {"task_id": task_id})
                with self.lock:
                    task = self.task
                    if task is None or task["task_id"] != task_id:
                        return
                    task["cursor"] = int(feed["cursor"])
                    task["phase"] = str(status.get("phase") or "running")
                    task["steps"] += sum(1 for e in feed["events"] if e.get("type") == "agent.step")
                    if feed["done"] and status["state"] != "running":
                        result = status.get("result") or {}
                        task["state"] = str(status["state"])
                        task["stop_reason"] = str(result.get("stop_reason", ""))
                        task["result"] = str(result.get("result", ""))[:RESULT_LIMIT]
                        task["steps"] = int(result.get("steps", task["steps"]))
                        task["error"] = status.get("error")
                        finished = copy.deepcopy(task)
                        break
                self.stopping.wait(self.poll)
        except UNAVAILABLE:
            with self.lock:
                if self.task is not None and self.task["task_id"] == task_id:
                    self.task["state"], self.task["error"] = "failed", "agent_unavailable"
                    finished = copy.deepcopy(self.task)
        except Exception as error:  # the feed broke; never crash the module worker
            with self.lock:
                if self.task is not None and self.task["task_id"] == task_id:
                    self.task["state"], self.task["error"] = "failed", type(error).__name__
                    finished = copy.deepcopy(self.task)
        if finished is None:
            return
        try:
            self.services.call("agent.forget", {"task_id": task_id})
        except Exception:
            pass  # the handle expires on its own; the journal stays with Agent
        finished.pop("started", None)
        finished.pop("cursor", None)
        if not self.stopping.is_set():
            self.on_finished(finished)
