"""6.2 (AG-02): файловые инструменты агента поверх `Workspace`.

Чтение: `list_dir`, `read_file`, `file_info`, `find_files`, `grep` (ограниченный), `diff`.
Запись: `write_file`, `edit_file`, `apply_patch` только готовят предложение с diff —
файлы не меняются; применение после подтверждения пользователя — 6.4. `apply_patch`
собирает несколько правок (в том числе в разных файлах) в одно предложение: оно
применяется целиком или не применяется вовсе.

Каждый путь проходит `Workspace.resolve`/`open_read`, поэтому права, границы папки
и скрытые секреты действуют одинаково для всех инструментов. Обход папок не идёт по
ссылкам и пропускает `.git` и секреты; у каждого инструмента свои лимиты.
"""

import difflib
import fnmatch
import hashlib
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ramvi_service.agent.tools import ToolError, ToolResult, ToolRunner, ToolSpec
from ramvi_service.agent.workspace import AccessDenied, Workspace

SKIP_DIRS = frozenset({".git"})


@dataclass(frozen=True)
class FileLimits:
    """Лимиты файловых инструментов (защита контекста модели и времени)."""

    max_list_entries: int = 500
    max_text_bytes: int = 1024 * 1024  # чтение, правка и grep одного файла
    max_read_lines: int = 2000
    max_find_results: int = 200
    max_walk_entries: int = 20_000
    max_grep_matches: int = 500
    max_grep_bytes: int = 32 * 1024 * 1024  # суммарно за один grep
    max_pattern_chars: int = 500
    max_line_chars: int = 300  # строка совпадения в выводе
    max_scan_line_chars: int = 10_000  # дальше строка не проверяется регуляркой


@dataclass(frozen=True)
class WriteProposal:
    """Предложенная запись: полное новое содержимое, хеш исходного файла и diff."""

    path: Path
    display: str
    before_sha256: str | None  # None — файла не было
    after: bytes
    diff: str


@dataclass(frozen=True)
class PatchProposal:
    """Атомарный патч: несколько предложенных записей, применяются все или ни одна."""

    files: tuple[WriteProposal, ...]
    diff: str


def unified_diff(before: str, after: str, display: str, *, new: bool = False) -> str:
    """Unified diff с пометкой «нет перевода строки в конце», как у git."""
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="/dev/null" if new else f"a/{display}",
        tofile=f"b/{display}",
    )
    out: list[str] = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


class FileTools:
    """Обработчики файловых инструментов одной рабочей папки."""

    def __init__(self, workspace: Workspace, limits: FileLimits | None = None) -> None:
        self.ws = workspace
        self.limits = limits or FileLimits()

    # --- общие части ------------------------------------------------------------------

    def _read_text(self, path: str, limit: int) -> tuple[str, bool]:
        with self.ws.open_read(path) as handle:
            data = handle.read(limit + 1)
        if _is_binary(data):
            raise ToolError("agent.tool.not_text", path=path)
        truncated = len(data) > limit
        return data[:limit].decode("utf-8", errors="replace"), truncated

    def _walk(self, base: Path) -> Iterator[Path]:
        """Файлы под `base` без ссылок, `.git` и секретов; не больше лимита записей."""
        seen = 0
        for top, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if d.lower() not in SKIP_DIRS and not self.ws.is_secret(d)
            )
            for name in sorted(files):
                seen += 1
                if seen > self.limits.max_walk_entries:
                    raise ToolError("agent.tool.too_many_files", limit=self.limits.max_walk_entries)
                if not self.ws.is_secret(name):
                    yield Path(top) / name

    def _base(self, path: str) -> Path:
        base = self.ws.resolve(path)
        if not base.is_dir():
            raise ToolError("agent.tool.not_directory", path=path)
        return base

    # --- чтение -----------------------------------------------------------------------

    def list_dir(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        base = self._base(path)
        rows: list[str] = []
        truncated = False
        with os.scandir(base) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if self.ws.is_secret(entry.name):
                    continue
                if len(rows) >= self.limits.max_list_entries:
                    truncated = True
                    break
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or os.path.isjunction(entry.path):
                    rows.append(f"l {entry.name}")
                elif stat.S_ISDIR(info.st_mode):
                    rows.append(f"d {entry.name}/")
                else:
                    rows.append(f"f {entry.name} ({info.st_size} B)")
        header = f"{self.ws.relative(base)}: {len(rows)} entries" + (
            " (truncated)" if truncated else ""
        )
        return ToolResult(True, "\n".join([header, *rows]), {"entries": len(rows)}, None, truncated)

    def read_file(self, args: dict[str, Any]) -> ToolResult:
        path = args["path"]
        start = args.get("start_line", 1)
        count = min(args.get("max_lines", 400), self.limits.max_read_lines)
        text, cut = self._read_text(path, self.limits.max_text_bytes)
        lines = text.splitlines()
        chosen = lines[start - 1 : start - 1 + count]
        end = start + len(chosen) - 1
        body = "\n".join(f"{start + i:6}\t{line}" for i, line in enumerate(chosen))
        more = end < len(lines) or cut
        footer = f"\n[lines {start}-{end} of {len(lines)}{'+' if cut else ''}]"
        data = {"start_line": start, "end_line": end, "total_lines": len(lines)}
        return ToolResult(True, body + footer, data, None, more)

    def file_info(self, args: dict[str, Any]) -> ToolResult:
        target = self.ws.resolve(args["path"])
        info = os.lstat(target)
        if stat.S_ISLNK(info.st_mode) or os.path.isjunction(target):
            kind = "link"
        elif stat.S_ISDIR(info.st_mode):
            kind = "directory"
        elif stat.S_ISREG(info.st_mode):
            kind = "file"
        else:
            kind = "other"
        data = {
            "path": self.ws.relative(target),
            "type": kind,
            "size": info.st_size,
            "modified": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(timespec="seconds"),
            "mode": stat.filemode(info.st_mode),
            "writable": os.access(target, os.W_OK),
        }
        content = "\n".join(f"{key}: {value}" for key, value in data.items())
        return ToolResult(True, content, data)

    def find_files(self, args: dict[str, Any]) -> ToolResult:
        pattern = args["pattern"].lower()
        base = self._base(args.get("path", "."))
        found: list[str] = []
        truncated = False
        for file in self._walk(base):
            rel = file.relative_to(base).as_posix().lower()
            target = rel if "/" in pattern else file.name.lower()
            if fnmatch.fnmatchcase(target, pattern):
                if len(found) >= self.limits.max_find_results:
                    truncated = True
                    break
                found.append(self.ws.relative(file))
        lines = list(found) or ["(no files)"]
        if truncated:
            lines.append(f"[first {len(found)} results]")
        return ToolResult(True, "\n".join(lines), {"files": found}, None, truncated)

    def grep(self, args: dict[str, Any]) -> ToolResult:
        if len(args["pattern"]) > self.limits.max_pattern_chars:
            raise ToolError("agent.tool.pattern_too_long", limit=self.limits.max_pattern_chars)
        try:
            regex = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as error:
            raise ToolError("agent.tool.bad_pattern", error=str(error)) from None
        limit = min(args.get("max_matches", 100), self.limits.max_grep_matches)
        glob = (args.get("glob") or "").lower()
        target = self.ws.resolve(args.get("path", "."))
        files = [target] if target.is_file() else self._walk(target)
        matches: list[str] = []
        scanned, skipped, truncated = 0, 0, False
        for file in files:
            if glob and not fnmatch.fnmatchcase(file.name.lower(), glob):
                continue
            rel = self.ws.relative(file)
            try:
                with self.ws.open_read(rel) as handle:
                    data = handle.read(self.limits.max_text_bytes + 1)
            except AccessDenied:
                skipped += 1
                continue
            if len(data) > self.limits.max_text_bytes or _is_binary(data):
                skipped += 1
                continue
            scanned += len(data)
            if scanned > self.limits.max_grep_bytes:
                truncated = True
                break
            text = data.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line[: self.limits.max_scan_line_chars]):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append(f"{rel}:{number}: {line[: self.limits.max_line_chars]}")
            if truncated:
                break
        lines = list(matches) or ["(no matches)"]
        if truncated:
            lines.append(f"[stopped at {len(matches)} matches or size limit]")
        if skipped:
            lines.append(f"[skipped {skipped} binary, large or inaccessible files]")
        data_out = {"matches": len(matches), "skipped": skipped}
        return ToolResult(True, "\n".join(lines), data_out, None, truncated)

    def diff(self, args: dict[str, Any]) -> ToolResult:
        before, cut_a = self._read_text(args["path"], self.limits.max_text_bytes)
        after, cut_b = self._read_text(args["other_path"], self.limits.max_text_bytes)
        if cut_a or cut_b:
            raise ToolError("agent.tool.too_large", limit=self.limits.max_text_bytes)
        a, b = args["path"], args["other_path"]
        lines = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{a}",
            tofile=f"b/{b}",
        )
        text = "".join(x if x.endswith("\n") else x + "\n" for x in lines)
        return ToolResult(True, text or "(files are identical)", {"identical": not text})

    # --- подготовка записи ------------------------------------------------------------

    def _current(self, path: str) -> tuple[Path, bytes | None]:
        target = self.ws.resolve(path, permission="write", must_exist=False)
        if not os.path.lexists(target):
            return target, None
        with self.ws.open_read(path) as handle:
            data = handle.read(self.limits.max_text_bytes + 1)
        if len(data) > self.limits.max_text_bytes:
            raise ToolError("agent.tool.too_large", limit=self.limits.max_text_bytes)
        if _is_binary(data):
            raise ToolError("agent.tool.not_text", path=path)
        return target, data

    def _proposal(self, path: str, target: Path, before: bytes | None, after: str) -> WriteProposal:
        encoded = after.encode("utf-8")
        if len(encoded) > self.limits.max_text_bytes:
            raise ToolError("agent.tool.too_large", limit=self.limits.max_text_bytes)
        if before is not None and encoded == before:
            raise ToolError("agent.tool.no_change", path=path)
        display = self.ws.relative(target)
        try:
            old = "" if before is None else before.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("agent.tool.not_text", path=path) from None
        diff = unified_diff(old, after, display, new=before is None)
        return WriteProposal(
            path=target,
            display=display,
            before_sha256=None if before is None else hashlib.sha256(before).hexdigest(),
            after=encoded,
            diff=diff,
        )

    def _propose(self, path: str, target: Path, before: bytes | None, after: str) -> ToolResult:
        proposal = self._proposal(path, target, before, after)
        display, diff = proposal.display, proposal.diff
        content = f"Proposed change to {display} (awaiting user approval):\n{diff}"
        return ToolResult(True, content, {"proposal": proposal})

    def write_file(self, args: dict[str, Any]) -> ToolResult:
        target, before = self._current(args["path"])
        return self._propose(args["path"], target, before, args["content"])

    def edit_file(self, args: dict[str, Any]) -> ToolResult:
        path, old, new = args["path"], args["old_text"], args["new_text"]
        target, before = self._current(path)
        if before is None:
            raise AccessDenied("agent.access.not_found", path=path)
        try:
            text = before.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("agent.tool.not_text", path=path) from None
        found = text.count(old)
        if found == 0:
            raise ToolError("agent.tool.edit_not_found", path=path)
        if found > 1:
            raise ToolError("agent.tool.edit_ambiguous", path=path, occurrences=found)
        return self._propose(path, target, before, text.replace(old, new, 1))

    def apply_patch(self, args: dict[str, Any]) -> ToolResult:
        """Собрать правки в одно атомарное предложение; любая ошибка отменяет всё."""
        files: dict[str, list[Any]] = {}  # реальный путь → [путь модели, путь, исходник, текст]
        for index, edit in enumerate(args["edits"]):
            path, old, new = edit["path"], edit["old_text"], edit["new_text"]
            key = str(self.ws.resolve(path, permission="write", must_exist=False))
            if key not in files:
                target, before = self._current(path)
                try:
                    text = None if before is None else before.decode("utf-8")
                except UnicodeDecodeError:
                    raise ToolError("agent.tool.not_text", path=path, edit=index) from None
                files[key] = [path, target, before, text]
            entry = files[key]
            if entry[3] is None:  # новый файл: только создание целиком
                if old:
                    raise ToolError("agent.tool.edit_not_found", path=path, edit=index)
                entry[3] = new
                continue
            if not old:
                raise ToolError("agent.tool.edit_empty", path=path, edit=index)
            found = entry[3].count(old)
            if found == 0:
                raise ToolError("agent.tool.edit_not_found", path=path, edit=index)
            if found > 1:
                raise ToolError(
                    "agent.tool.edit_ambiguous", path=path, edit=index, occurrences=found
                )
            entry[3] = entry[3].replace(old, new, 1)
        proposals: list[WriteProposal] = []
        for path, target, before, text in files.values():
            if before is not None and text.encode("utf-8") == before:
                continue
            proposals.append(self._proposal(path, target, before, text))
        if not proposals:
            raise ToolError("agent.tool.no_change")
        diff = "".join(p.diff for p in proposals)
        patch = PatchProposal(tuple(proposals), diff)
        names = ", ".join(p.display for p in proposals)
        content = f"Proposed patch for {names} (awaiting user approval):\n{diff}"
        return ToolResult(True, content, {"proposal": patch})


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_PATH = {"type": "string", "maxLength": 4096, "description": "Path inside the workspace"}
_DIR = {"type": "string", "maxLength": 4096, "description": "Folder; default is the workspace"}


def register_file_tools(
    runner: ToolRunner, workspace: Workspace, limits: FileLimits | None = None
) -> FileTools:
    """Зарегистрировать файловые инструменты одной рабочей папки."""
    tools = FileTools(workspace, limits)
    lim = tools.limits
    specs = [
        (
            ToolSpec(
                "list_dir",
                "List files and folders in a folder (d = folder, f = file, l = link).",
                _schema({"path": _DIR}, []),
                "read",
            ),
            tools.list_dir,
        ),
        (
            ToolSpec(
                "read_file",
                "Read a text file; lines are numbered. Use start_line to page.",
                _schema(
                    {
                        "path": _PATH,
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": lim.max_read_lines,
                        },
                    },
                    ["path"],
                ),
                "read",
            ),
            tools.read_file,
        ),
        (
            ToolSpec(
                "file_info",
                "File or folder metadata: type, size, modification time, permissions.",
                _schema({"path": _PATH}, ["path"]),
                "read",
            ),
            tools.file_info,
        ),
        (
            ToolSpec(
                "find_files",
                "Find files by glob pattern (e.g. '*.py' or 'src/*/test_*.py').",
                _schema(
                    {"pattern": {"type": "string", "minLength": 1, "maxLength": 500}, "path": _DIR},
                    ["pattern"],
                ),
                "read",
            ),
            tools.find_files,
        ),
        (
            ToolSpec(
                "grep",
                "Search text files for a regular expression; returns file:line: text.",
                _schema(
                    {
                        "pattern": {"type": "string", "minLength": 1},
                        "path": _DIR,
                        "glob": {"type": "string", "maxLength": 200},
                        "ignore_case": {"type": "boolean"},
                        "max_matches": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": lim.max_grep_matches,
                        },
                    },
                    ["pattern"],
                ),
                "read",
            ),
            tools.grep,
        ),
        (
            ToolSpec(
                "diff",
                "Show a unified diff between two text files.",
                _schema({"path": _PATH, "other_path": _PATH}, ["path", "other_path"]),
                "read",
            ),
            tools.diff,
        ),
        (
            ToolSpec(
                "write_file",
                "Create or replace a text file. The user sees the diff and must approve it.",
                _schema({"path": _PATH, "content": {"type": "string"}}, ["path", "content"]),
                "write",
                "write",
                "high",
            ),
            tools.write_file,
        ),
        (
            ToolSpec(
                "edit_file",
                "Replace one exact, unique fragment of a text file. The user must approve it.",
                _schema(
                    {
                        "path": _PATH,
                        "old_text": {"type": "string", "minLength": 1},
                        "new_text": {"type": "string"},
                    },
                    ["path", "old_text", "new_text"],
                ),
                "write",
                "write",
                "high",
            ),
            tools.edit_file,
        ),
        (
            ToolSpec(
                "apply_patch",
                "Several exact edits, possibly in different files, as one atomic patch. "
                "Each old_text must be unique in its file; empty old_text creates a new file. "
                "The user sees the whole diff and approves it or not.",
                _schema(
                    {
                        "edits": {
                            "type": "array",
                            "maxItems": 50,
                            "items": _schema(
                                {
                                    "path": _PATH,
                                    "old_text": {"type": "string"},
                                    "new_text": {"type": "string"},
                                },
                                ["path", "old_text", "new_text"],
                            ),
                        }
                    },
                    ["edits"],
                ),
                "write",
                "write",
                "high",
            ),
            tools.apply_patch,
        ),
    ]
    for spec, handler in specs:
        runner.register(spec, handler)
    return tools
