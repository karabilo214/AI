"""6.7 (AG-07): журнал агента — сводка, хронология, «почему изменился файл», очистка.

Данные пишет runtime: `agent_runs` (задача, статус, токены, итог, причина),
`tool_calls` (каждый вызов: очищенные аргументы, разрешение, результат, время),
`agent_events` (план, шаги с токенами и занятым контекстом, вызовы, запросы
подтверждения с diff, решения пользователя, контрольные точки, итог). Длинные
строки в событиях обрезаны (`runtime.MAX_EVENT_TEXT`), результаты вызовов — до 4000
символов, длинные аргументы — длина, хеш и начало.

`why_changed(path)` восстанавливает причину изменения файла: задача запуска → план
на этом шаге → вызов и его аргументы → кто и как подтвердил → diff → номер
изменения и можно ли его откатить.

Очистка обязательна, иначе журнал и снимки разрастаются: `maintain(retention)` —
автоматически по сроку и общему размеру (настройки вкладки «Агент», `None` — без
ограничения), `clear(before)` — по кнопке (всё или старше даты). Вместе с запуском
удаляются снимки его изменений: откатить их после этого нельзя. Идущие запуски не
трогаются. После удаления база сжимается (`Database.compact`).
"""

import json
import posixpath
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DAY = 86_400.0


@dataclass(frozen=True)
class JournalRetention:
    """Автоочистка журнала (меняет пользователь; None — без ограничения)."""

    max_age_days: float | None = 30.0
    max_bytes: int | None = 500 * 1024 * 1024  # база журнала + снимки изменений


def _dir_bytes(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def _norm(path: str) -> str:
    """`./src\\calc.py` и `src/calc.py` — один путь."""
    return posixpath.normpath(path.replace("\\", "/")).lstrip("/") if path else ""


class AgentJournal:
    """Чтение и очистка журнала агента."""

    def __init__(self, runs: Any, changes_root: Path | None = None) -> None:
        """`runs` — `AgentRuns`; `changes_root` — папка снимков `ChangeStore`."""
        self.runs, self.changes_root = runs, changes_root

    # --- чтение -----------------------------------------------------------------------

    def summary(self, run_id: str) -> dict[str, Any]:
        """Сводка для экрана: статус, время, токены, контекст, вызовы, ошибки."""
        run = self.runs.get(run_id)
        calls = self.runs.tool_calls(run_id)
        events = self.runs.events(run_id)
        usage = [e["payload"] for e in events if e["kind"] == "agent.usage"]
        errors = self.errors(run_id)
        end = run.ended_at if run.ended_at is not None else time.time()
        return {
            "id": run.id,
            "objective": run.objective,
            "workspace": run.workspace,
            "status": run.status,
            "stop_reason": run.stop_reason,
            "result": run.result,
            "steps": run.steps,
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "duration": max(0.0, end - run.started_at),
            "tokens": run.token_usage or {},
            "context_peak": max((u.get("context", 0) for u in usage), default=0),
            "tool_calls": len(calls),
            "changes": [
                e["payload"]["change_id"]
                for e in events
                if e["kind"] == "agent.tool_result" and e["payload"].get("change_id")
            ],
            "errors": len(errors),
        }

    def timeline(self, run_id: str) -> list[dict[str, Any]]:
        """Хронология запуска по порядку: всё, что показывает экран агента."""
        return list(self.runs.events(run_id))

    def experience(self, run_id: str) -> list[dict[str, Any]]:
        """AG-08: опыт только выбранного запуска; не передаётся модели автоматически.

        Ссылки state.run_id/step и action.call_id ведут к исходной хронологии.
        Удаляется вместе с журналом; прошлые разрешения повторно не используются.
        """
        return [e["payload"] for e in self.runs.events(run_id) if e["kind"] == "agent.experience"]

    def errors(self, run_id: str) -> list[dict[str, Any]]:
        """Таймлайн ошибок: ошибки инструментов, отказы, остановка с неудачей."""
        found: list[dict[str, Any]] = []
        for event in self.runs.events(run_id):
            payload = event["payload"]
            failed_tool = event["kind"] == "agent.tool_result" and payload.get("error")
            failed_run = event["kind"] == "agent.done" and payload.get("status") != "done"
            if failed_tool or failed_run:
                found.append(event)
        return found

    def why_changed(self, path: str) -> list[dict[str, Any]]:
        """Все изменения файла агентом, от старых к новым, с причиной каждого."""
        target = _norm(path)
        answers: list[dict[str, Any]] = []
        name = target.rsplit("/", 1)[-1]  # без разделителей: в JSON «\\» экранируется
        for hit in self.runs.find_events("agent.tool_result", name):
            applied = [_norm(p) for p in hit["payload"].get("applied") or []]
            if target not in applied:
                continue
            run_id, call_id = hit["run_id"], hit["payload"].get("call_id")
            events = [e for e in self.runs.events(run_id) if e["seq"] <= hit["seq"]]
            call = next(
                (
                    e
                    for e in reversed(events)
                    if e["kind"] == "agent.tool_call" and e["payload"].get("call_id") == call_id
                ),
                None,
            )
            start = call["seq"] if call else 0
            between = [e for e in events if e["seq"] > start]
            request = next(
                (
                    e["payload"]["request"]
                    for e in between
                    if e["kind"] == "agent.tool_approval_required"
                ),
                None,
            )
            decision = next(
                (
                    e["payload"]["decision"]
                    for e in between
                    if e["kind"] == "agent.tool_approval_decided"
                ),
                None,
            )
            plans = [e for e in events if e["kind"] == "agent.plan"]
            run = self.runs.get(run_id)
            change_id = hit["payload"].get("change_id")
            answers.append(
                {
                    "run_id": run_id,
                    "objective": run.objective,
                    "at": hit["created_at"],
                    "step": hit["step"],
                    "plan": plans[-1]["payload"].get("text") if plans else None,
                    "tool": hit["payload"].get("tool"),
                    "args": None if call is None else call["payload"].get("args"),
                    "approval": decision or {"action": "auto"},
                    "diff": None if request is None else request.get("diff"),
                    "change_id": change_id,
                    "rollback": self._rollback_state(change_id),
                }
            )
        return answers

    def _rollback_state(self, change_id: str | None) -> str:
        if not change_id or self.changes_root is None:
            return "unknown"
        manifest = self.changes_root / change_id / "manifest.json"
        if not manifest.exists():
            return "snapshot_removed"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return "rolled_back" if data.get("rolled_back") else "available"

    # --- размер и очистка -------------------------------------------------------------

    def _change_ids(self, run_id: str) -> list[str]:
        return [
            e["payload"]["change_id"]
            for e in self.runs.events(run_id, ("agent.tool_result",))
            if e["payload"].get("change_id")
        ]

    def size(self) -> dict[str, int]:
        runs = self.runs.finished()
        database = sum(self.runs.run_bytes(r.id) for r in runs)
        snapshots = 0 if self.changes_root is None else _dir_bytes(self.changes_root)
        return {"runs": len(runs), "database": database, "snapshots": snapshots}

    def _delete(self, run_ids: list[str]) -> int:
        if not run_ids:
            return 0
        for run_id in run_ids:
            if self.changes_root is not None:
                for change_id in self._change_ids(run_id):
                    if change_id.isalnum():
                        shutil.rmtree(self.changes_root / change_id, ignore_errors=True)
        deleted = int(self.runs.delete(run_ids))
        self.runs.db.compact()
        return deleted

    def clear(self, before: float | None = None) -> int:
        """По кнопке «Очистить»: все завершённые запуски или закончившиеся раньше `before`."""
        return self._delete([r.id for r in self.runs.finished(before)])

    def maintain(self, retention: JournalRetention, now: float | None = None) -> int:
        """Автоочистка: старше срока, затем самые старые — пока не уложится в размер."""
        now = time.time() if now is None else now
        doomed: list[str] = []
        if retention.max_age_days is not None:
            doomed += [r.id for r in self.runs.finished(now - retention.max_age_days * DAY)]
        if retention.max_bytes is not None:
            left = [r for r in self.runs.finished() if r.id not in doomed]
            sizes = {r.id: self.runs.run_bytes(r.id) for r in left}
            snapshot_sizes = {
                r.id: sum(
                    _dir_bytes(self.changes_root / c)
                    for c in self._change_ids(r.id)
                    if self.changes_root is not None and c.isalnum()
                )
                for r in left
            }
            total = sum(sizes.values()) + sum(snapshot_sizes.values())
            for run in left:  # от старых к новым
                if total <= retention.max_bytes:
                    break
                doomed.append(run.id)
                total -= sizes[run.id] + snapshot_sizes[run.id]
        return self._delete(doomed)
