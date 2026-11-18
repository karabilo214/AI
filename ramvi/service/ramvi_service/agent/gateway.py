"""6.10: Host-owned Agent Gateway. Модуль предлагает задачу, не выдаёт себе права."""

import copy
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ramvi_service.agent.readiness import ModelFacts
from ramvi_service.agent.runtime import AgentRuntime, RunResult, _bounded
from ramvi_service.context import GrantStore
from ramvi_service.contracts.modules import Authority, Owner

RuntimeFactory = Callable[[], AgentRuntime]


@dataclass
class Binding:
    owner: Owner
    factory: RuntimeFactory
    model_id: str | None = None
    facts: ModelFacts | None = None
    context_data: str = ""


@dataclass
class Task:
    owner: Owner
    binding_id: str
    cancel: threading.Event
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    thread: threading.Thread | None = None
    result: RunResult | None = None
    error: str | None = None
    finished: float | None = None
    seq: int = 0
    phase: str = "running"


class AgentGateway:
    """Публичные create/status/events/cancel; bind/revoke — только доверенный Host.

    Фабрика создаёт свежий runtime с теми же ToolRunner/ApprovalGate, что у текстового
    клиента. JSON модуля не принимает пути, права, стратегию или решение approve.
    Долговечный журнал — у runtime; ограниченные handles живут в сессии сервиса.
    """

    def __init__(self, authority: Authority, grants: GrantStore, *, timeout: float = 10) -> None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid shutdown timeout")
        if grants.authority is not authority:
            raise ValueError("different authorities")
        self.authority, self.grants, self.timeout = authority, grants, timeout
        self._bindings: dict[str, Binding] = {}
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()
        self._closed = False

    def bind(
        self,
        owner: Owner,
        factory: RuntimeFactory,
        *,
        model_id: str | None = None,
        facts: ModelFacts | None = None,
        context_data: str = "",
    ) -> str:
        """Явный выбор рабочей папки/политики пользователем; этого метода нет в RPC."""
        self.authority.require(owner)
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent Gateway closed")
            identifier = uuid.uuid4().hex
            self._bindings[identifier] = Binding(owner, factory, model_id, facts, context_data)
            return identifier

    def revoke(self, owner: Owner, binding_id: str) -> None:
        with self._lock:
            binding = self._bindings.get(binding_id)
            if binding is None or binding.owner != owner:
                raise PermissionError("not your workspace binding")
            del self._bindings[binding_id]
            for task in self._tasks.values():
                if task.binding_id == binding_id:
                    task.cancel.set()

    def create(self, owner: Owner, goal: str, binding_id: str) -> str:
        self.grants.require(owner, "agent")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 32_000:
            raise ValueError("invalid agent goal")
        with self._lock:
            if self._closed:
                raise RuntimeError("Agent Gateway closed")
            binding = self._bindings.get(binding_id)
            if binding is None or binding.owner != owner:
                raise PermissionError("not your workspace binding")
            for identifier, old in list(self._tasks.items()):
                if old.finished is not None and time.monotonic() - old.finished > 1800:
                    del self._tasks[identifier]
            if len(self._tasks) >= 32:
                raise RuntimeError("agent task handle limit; forget completed tasks")
            if any(t.binding_id == binding_id and t.finished is None for t in self._tasks.values()):
                raise RuntimeError("workspace binding already has a task")
            cancel = self.grants.watch(owner, [("agent", "*")])
            task_id = uuid.uuid4().hex
            task = Task(owner, binding_id, cancel)
            self._tasks[task_id] = task
            task.thread = threading.Thread(
                target=self._run, args=(task_id, task, binding, goal), name="ramvi-agent-task"
            )
            try:
                task.thread.start()
            except BaseException:
                self._tasks.pop(task_id)
                self.grants.unwatch(cancel)
                raise
            return task_id

    def _run(self, task_id: str, task: Task, binding: Binding, goal: str) -> None:
        try:
            if task.cancel.is_set():
                raise InterruptedError("agent task cancelled")
            runtime = binding.factory()
            previous = runtime._emit

            def emit(event: dict[str, Any]) -> None:
                with self._lock:
                    task.seq += 1
                    if event["type"] in {"agent.tool_approval_required", "agent.limit_reached"}:
                        task.phase = "waiting_approval"
                    elif event["type"] in {"agent.tool_approval_decided", "agent.limit_decided"}:
                        task.phase = "running"
                    task.events.append(_bounded(copy.deepcopy({**event, "cursor": task.seq})))
                if previous is not None:
                    previous(event)

            runtime._emit = emit
            try:
                result = runtime.run(
                    goal,
                    workspace=str(runtime.tools.workspace.root) if runtime.tools.workspace else "",
                    model_id=binding.model_id,
                    facts=binding.facts,
                    context_data=binding.context_data,
                    cancel=task.cancel,
                    run_id=task_id,
                    origin={
                        "principal_id": task.owner.principal_id,
                        "module_id": task.owner.module_id,
                        "session_id": task.owner.session_id,
                    },
                )
            finally:
                runtime._emit = previous
            with self._lock:
                task.result = result
        except Exception as error:
            # Не выдавать файлы/секреты из текста исключения клиенту.
            with self._lock:
                task.error = type(error).__name__
                task.phase = "failed"
                task.seq += 1
                task.events.append(
                    {
                        "type": "agent.gateway_failed",
                        "task_id": task_id,
                        "cursor": task.seq,
                        "error": task.error,
                    }
                )
        finally:
            self.grants.unwatch(task.cancel)
            with self._lock:
                task.finished = time.monotonic()

    def _task(self, owner: Owner, task_id: str) -> Task:
        self.authority.require(owner)
        task = self._tasks.get(task_id)
        if task is None or task.owner != owner:
            raise PermissionError("not your agent task")
        return task

    def status(self, owner: Owner, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._task(owner, task_id)
            return {
                "task_id": task_id,
                "principal_id": owner.principal_id,
                "source_module": owner.module_id,
                "state": (task.result.status if task.result else "failed")
                if task.finished is not None
                else "running",
                "result": asdict(task.result) if task.result else None,
                "error": task.error,
                "phase": task.result.status if task.result else task.phase,
            }

    def events(self, owner: Owner, task_id: str, after: int = 0) -> dict[str, Any]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("invalid event cursor")
        with self._lock:
            task = self._task(owner, task_id)
            first = task.events[0]["cursor"] if task.events else task.seq + 1
            return {
                "events": copy.deepcopy([e for e in task.events if e["cursor"] > after]),
                "cursor": task.seq,
                "gap": after < first - 1,
                "done": task.finished is not None,
            }

    def cancel(self, owner: Owner, task_id: str) -> None:
        with self._lock:
            self._task(owner, task_id).cancel.set()

    def forget(self, owner: Owner, task_id: str) -> None:
        with self._lock:
            task = self._task(owner, task_id)
            if task.finished is None:
                raise RuntimeError("task is still running")
            del self._tasks[task_id]

    def release_owner(self, owner: Owner) -> None:
        with self._lock:
            tasks = [t for t in self._tasks.values() if t.owner == owner]
            self._bindings = {k: b for k, b in self._bindings.items() if b.owner != owner}
            for task in tasks:
                task.cancel.set()
        self._join(tasks)
        with self._lock:
            self._tasks = {k: t for k, t in self._tasks.items() if t.owner != owner}

    def _join(self, tasks: list[Task]) -> None:
        deadline = time.monotonic() + self.timeout
        for task in tasks:
            if task.thread is not None:
                task.thread.join(max(0, deadline - time.monotonic()))
                if task.thread.is_alive():
                    raise TimeoutError("agent task did not stop")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel.set()
        self._join(tasks)
        with self._lock:
            self._tasks.clear()
            self._bindings.clear()
