"""6.4 (AG-04): применение подтверждённых записей со снимками и откат.

`ChangeStore.apply` применяет `WriteProposal`/`PatchProposal` только целиком:

1. каждый путь заново проходит `Workspace.resolve` (отзыв, подмена ссылкой);
2. текущий файл сверяется с хешем, который видел агент: файл успели изменить или
   создать — изменение не применяется (`agent.change.conflict`);
3. исходное содержимое сохраняется снимком в папке хранилища (`<id>/manifest.json`
   и копии файлов) — до первой записи;
4. запись атомарная: временный файл рядом, fsync, `os.replace`; права исходного
   файла сохраняются. Ошибка на любом файле откатывает уже записанные.

`rollback(id)` возвращает файлы из снимков (созданные — удаляет вместе с пустыми
созданными папками). Если файл после изменения успели поменять ещё раз, откат не
перетирает чужую правку (`agent.change.conflict`), пока не передан `force=True`.
Откат — действие пользователя: ему не нужны права агента, но границы путей те же.
Снимки переживают перезапуск приложения.
"""

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ramvi_service.agent.files import PatchProposal, WriteProposal
from ramvi_service.agent.tools import ToolError
from ramvi_service.agent.workspace import Permissions, Workspace

MANIFEST = "manifest.json"


@dataclass(frozen=True)
class FileChange:
    display: str
    before_sha256: str | None  # None — файла не было
    after_sha256: str
    snapshot: str | None  # имя копии исходника в папке изменения
    created_dirs: tuple[str, ...]  # созданные папки (для отката), от внешней к внутренней


@dataclass(frozen=True)
class ChangeRecord:
    id: str
    files: tuple[FileChange, ...]
    created_at: float
    rolled_back: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ChangeRecord":
        files = tuple(
            FileChange(**{**f, "created_dirs": tuple(f["created_dirs"])}) for f in data["files"]
        )
        return cls(data["id"], files, data["created_at"], data.get("rolled_back", False))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(target: Path, data: bytes, mode: int | None) -> None:
    temp = target.with_name(f".{target.name}.ramvi-{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temp, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp, stat.S_IMODE(mode))
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


class ChangeStore:
    """Снимки и журнал применённых изменений одной рабочей папки."""

    def __init__(self, root: Path, workspace: Workspace) -> None:
        self.root, self.ws = Path(root), workspace
        self.root.mkdir(parents=True, exist_ok=True)

    # --- общие части ------------------------------------------------------------------

    def _read(self, ws: Workspace, display: str) -> bytes | None:
        target = ws.resolve(display, permission="write", must_exist=False)
        if not os.path.lexists(target):
            return None
        with ws.open_read(display) as handle:
            return handle.read()

    def _save(self, record: ChangeRecord) -> None:
        folder = self.root / record.id
        data = json.dumps(record.to_json(), ensure_ascii=False, indent=1).encode()
        _atomic_write(folder / MANIFEST, data, None)

    def get(self, change_id: str) -> ChangeRecord:
        if not change_id.isalnum():
            raise ToolError("agent.change.not_found", change=change_id)
        try:
            data = json.loads((self.root / change_id / MANIFEST).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ToolError("agent.change.not_found", change=change_id) from None
        return ChangeRecord.from_json(data)

    def history(self) -> list[ChangeRecord]:
        records = [self.get(p.name) for p in self.root.iterdir() if (p / MANIFEST).exists()]
        return sorted(records, key=lambda r: r.created_at)

    # --- применение -------------------------------------------------------------------

    def apply(self, proposal: WriteProposal | PatchProposal) -> ChangeRecord:
        """Применить предложение целиком; при любой ошибке файлы остаются прежними."""
        writes = proposal.files if isinstance(proposal, PatchProposal) else (proposal,)
        originals: list[bytes | None] = []
        for write in writes:  # сначала все проверки — до первой записи
            target = self.ws.resolve(write.display, permission="write", must_exist=False)
            if target != write.path:
                raise ToolError("agent.change.conflict", path=write.display)
            current = self._read(self.ws, write.display)
            if (None if current is None else _sha(current)) != write.before_sha256:
                raise ToolError("agent.change.conflict", path=write.display)
            originals.append(current)
        change_id = uuid.uuid4().hex
        folder = self.root / change_id
        folder.mkdir()
        files: list[FileChange] = []
        for index, (write, original) in enumerate(zip(writes, originals, strict=True)):
            snapshot = None
            if original is not None:
                snapshot = f"{index}.orig"
                _atomic_write(folder / snapshot, original, None)
            files.append(
                FileChange(write.display, write.before_sha256, _sha(write.after), snapshot, ())
            )
        record = ChangeRecord(change_id, tuple(files), time.time())
        self._save(record)  # снимки на диске до первой записи в рабочую папку
        done: list[FileChange] = []
        try:
            for i, write in enumerate(writes):
                created: list[str] = []
                parent = write.path.parent
                missing: list[Path] = []
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created.append(self.ws.relative(directory))
                files[i] = FileChange(**{**asdict(files[i]), "created_dirs": tuple(created)})
                done.append(files[i])  # при сбое записи откат уберёт и созданные папки
                mode = os.stat(write.path).st_mode if originals[i] is not None else None
                _atomic_write(write.path, write.after, mode)
        except BaseException:
            self._restore(done, folder, self.ws)
            shutil.rmtree(folder, ignore_errors=True)
            raise
        record = ChangeRecord(change_id, tuple(files), record.created_at)
        self._save(record)
        return record

    # --- откат ------------------------------------------------------------------------

    def _restore(self, files: list[FileChange], folder: Path, ws: Workspace) -> None:
        for change in reversed(files):
            target = ws.resolve(change.display, permission="write", must_exist=False)
            if change.snapshot is None:
                if os.path.lexists(target):
                    target.unlink()
                for directory in reversed(change.created_dirs):
                    path = ws.resolve(directory, permission="write", must_exist=False)
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
            else:
                data = (folder / change.snapshot).read_bytes()
                mode = os.stat(target).st_mode if os.path.lexists(target) else None
                _atomic_write(target, data, mode)

    def rollback(self, change_id: str, *, force: bool = False) -> ChangeRecord:
        """Вернуть файлы изменения из снимков (действие пользователя)."""
        record = self.get(change_id)
        if record.rolled_back:
            raise ToolError("agent.change.already_rolled_back", change=change_id)
        # Права агента могли быть отозваны — откат делает пользователь, границы путей те же.
        ws = Workspace(self.ws.root, Permissions(read=True, write=True), self.ws.restrictions)
        if not force:
            for change in record.files:
                current = self._read(ws, change.display)
                if current is None or _sha(current) != change.after_sha256:
                    raise ToolError("agent.change.conflict", path=change.display)
        self._restore(list(record.files), self.root / change_id, ws)
        record = ChangeRecord(record.id, record.files, record.created_at, rolled_back=True)
        self._save(record)
        return record
