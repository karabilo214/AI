"""Проверки навигации по фактической документации проекта без GPU."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def test_local_documentation_links_resolve() -> None:
    """Перенос документа не должен оставлять ссылки на отсутствующие файлы."""
    documents = [
        ROOT.parent / "README.md",
        ROOT / "README.md",
        ROOT / "engine" / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    missing: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            link = urlsplit(target.strip("<>"))
            if link.scheme or link.netloc or not link.path:
                continue
            destination = document.parent / unquote(link.path)
            if not destination.exists():
                missing.append(f"{document.relative_to(ROOT.parent)}: {target}")
    assert not missing, "Неразрешённые ссылки:\n" + "\n".join(missing)
