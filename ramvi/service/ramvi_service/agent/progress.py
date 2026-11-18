"""6.1: распознавание зацикливания по журналу запуска, а не по одинаковым вызовам.

Повтор вызова сам по себе нормален (перечитать файл после правки). Петля — когда
работа не движется:

* **правка тех же строк** — одни и те же строки одного файла меняются снова и снова
  (`same_lines`, после `max_same_lines` правок);
* **возврат к прежнему виду** — файл стал таким, каким уже был в этом запуске: агент
  отменяет сам себя (`oscillation`, сразу);
* **повтор без новой информации** — тот же вызов вернул тот же результат, и между
  ними в файлах ничего не менялось (ни записи, ни команды) (`no_progress`, после
  `max_no_progress` таких вызовов подряд).

Найденная петля — не обрыв: runtime спрашивает пользователя, продолжать ли.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def changed_lines(diff: str) -> dict[str, list[int]]:
    """Номера изменённых строк исходного файла (начало каждого блока) по unified diff."""
    result: dict[str, list[int]] = {}
    path: str | None = None
    old = 0
    in_change = False
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/")
            result.setdefault(path, [])
            continue
        if line.startswith("--- "):
            continue
        match = _HUNK.match(line)
        if match:
            old, in_change = int(match.group(1)), False
            continue
        if path is None or line.startswith("\\"):
            continue
        if line.startswith(("-", "+")):
            if not in_change:
                result[path].append(old)
                in_change = True
            if line.startswith("-"):
                old += 1
        else:
            old += 1
            in_change = False
    return result


@dataclass
class ProgressMonitor:
    """Сравнивает каждый результат с уже записанным в этом запуске."""

    max_no_progress: int | None = 3
    max_same_lines: int | None = 3
    _state: int = 0  # меняется при каждой применённой записи и выполненной команде
    _seen: set[tuple[str, str, str, int]] = field(default_factory=set)
    _no_progress: int = 0
    _line_edits: dict[tuple[str, int], int] = field(default_factory=dict)
    _file_states: dict[str, set[str]] = field(default_factory=dict)

    def observe(self, tool: str, arguments: str, result: Any) -> tuple[str, str] | None:
        """Вернуть (вид петли, подробность) или None."""
        files = result.data.get("applied") if result.ok else None
        if files:  # применённая запись
            self._state += 1
            self._no_progress = 0
            return self._writes(files)
        if result.ok and result.data.get("exit_code") is not None:
            self._state += 1  # команда могла изменить файлы
        digest = hashlib.sha256(result.content.encode()).hexdigest()
        signature = (tool, arguments, digest, self._state)
        if signature in self._seen:
            self._no_progress += 1
        else:
            self._seen.add(signature)
            self._no_progress = 0
        if self.max_no_progress is not None and self._no_progress >= self.max_no_progress - 1:
            return "no_progress", tool
        return None

    def _writes(self, files: list[dict[str, Any]]) -> tuple[str, str] | None:
        found: tuple[str, str] | None = None
        for change in files:
            path = change["path"]
            states = self._file_states.setdefault(path, set())
            if change.get("before_sha256"):
                states.add(change["before_sha256"])
            if change["after_sha256"] in states and found is None:
                found = ("oscillation", path)
            states.add(change["after_sha256"])
            for line in change.get("lines", []):
                key = (path, line)
                self._line_edits[key] = self._line_edits.get(key, 0) + 1
                limit = self.max_same_lines
                if limit is not None and self._line_edits[key] >= limit and found is None:
                    found = ("same_lines", f"{path}:{line}")
        return found

    def forgive(self, kind: str, detail: str) -> None:
        """Пользователь разрешил продолжить — начать счёт этой петли заново."""
        if kind == "no_progress":
            self._no_progress = 0
            self._seen.clear()
        elif kind == "same_lines":
            path, _, line = detail.rpartition(":")
            self._line_edits[(path, int(line))] = 0
        elif kind == "oscillation":
            self._file_states.pop(detail, None)
