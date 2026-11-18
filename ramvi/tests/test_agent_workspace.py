"""6.3 (AG-03): агент не выходит за рабочую папку и не видит секретов."""

import os
from pathlib import Path

import pytest
from ramvi_service.agent.workspace import AccessDenied, Permissions, Workspace


def _code(call: object) -> str:
    with pytest.raises(AccessDenied) as error:
        call()  # type: ignore[operator]
    return error.value.code


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    root, outside = tmp_path / "project", tmp_path / "outside"
    (root / "src").mkdir(parents=True)
    outside.mkdir()
    (root / "src" / "main.py").write_text("print(1)\n")
    (root / "README.md").write_text("readme\n")
    (outside / "secret.txt").write_text("outside\n")
    return root, outside


def test_normal_paths_resolve_inside(tree: tuple[Path, Path]) -> None:
    root, _ = tree
    ws = Workspace(root)
    real = Path(os.path.realpath(root))
    assert ws.resolve("src/main.py") == real / "src" / "main.py"
    assert ws.resolve("./src//main.py") == real / "src" / "main.py"
    assert ws.resolve("src\\main.py") == real / "src" / "main.py"
    assert ws.resolve("") == real and ws.resolve(".") == real
    assert ws.relative(ws.resolve("src/main.py")) == "src/main.py"
    assert ws.relative(ws.resolve(".")) == "."
    with ws.open_read("src/main.py") as handle:
        assert handle.read() == b"print(1)\n"


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("../outside/secret.txt", "agent.access.parent"),
        ("src/../../outside/secret.txt", "agent.access.parent"),
        ("src/..", "agent.access.parent"),
        ("/etc/passwd", "agent.access.absolute"),
        ("\\Windows\\win.ini", "agent.access.absolute"),
        ("C:\\Windows\\win.ini", "agent.access.absolute"),
        ("C:file.txt", "agent.access.absolute"),
        ("\\\\server\\share\\x", "agent.access.absolute"),
        ("src/main.py:stream", "agent.access.invalid"),
        ("...", "agent.access.invalid"),
        (". .", "agent.access.invalid"),
        ("src/con.txt", "agent.access.invalid"),
        ("a\0b", "agent.access.invalid"),
        ("missing.txt", "agent.access.not_found"),
        ("README.md/x", "agent.access.not_directory"),
    ],
)
def test_escape_attempts_are_denied(tree: tuple[Path, Path], path: str, code: str) -> None:
    root, _ = tree
    assert _code(lambda: Workspace(root).resolve(path)) == code


def test_symlinks_are_denied_even_inside(tree: tuple[Path, Path]) -> None:
    root, outside = tree
    (root / "out").symlink_to(outside, target_is_directory=True)
    (root / "out_file").symlink_to(outside / "secret.txt")
    (root / "inner").symlink_to(root / "src", target_is_directory=True)
    ws = Workspace(root, Permissions(write=True))
    assert _code(lambda: ws.resolve("out/secret.txt")) == "agent.access.symlink"
    assert _code(lambda: ws.resolve("out_file")) == "agent.access.symlink"
    assert _code(lambda: ws.open_read("out_file")) == "agent.access.symlink"
    assert _code(lambda: ws.resolve("inner/main.py")) == "agent.access.symlink"
    # Создание нового файла через ссылку на папку тоже закрыто.
    write = lambda: ws.resolve("out/new.txt", permission="write", must_exist=False)  # noqa: E731
    assert _code(write) == "agent.access.symlink"
    assert not (outside / "new.txt").exists()


def test_root_given_through_symlink_is_canonical(tree: tuple[Path, Path]) -> None:
    root, _ = tree
    alias = root.parent / "alias"
    alias.symlink_to(root, target_is_directory=True)
    ws = Workspace(alias)
    assert ws.root == Path(os.path.realpath(root))
    assert ws.resolve("README.md").read_text() == "readme\n"


def test_hardlink_to_outside_file_is_denied(tree: tuple[Path, Path]) -> None:
    root, outside = tree
    os.link(outside / "secret.txt", root / "linked.txt")
    ws = Workspace(root)
    assert _code(lambda: ws.resolve("linked.txt")) == "agent.access.hardlink"
    assert _code(lambda: ws.open_read("linked.txt")) == "agent.access.hardlink"


@pytest.mark.parametrize(
    "name",
    [".env", ".ENV", ".env.local", "server.pem", "id_rsa", "id_ed25519.pub", ".ssh/config"],
)
def test_secrets_are_hidden_at_any_depth(tree: tuple[Path, Path], name: str) -> None:
    root, _ = tree
    target = root / "src" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("token\n")
    ws = Workspace(root, Permissions(write=True))
    assert _code(lambda: ws.resolve(f"src/{name}")) == "agent.access.secret"
    assert _code(lambda: ws.open_read(f"src/{name}")) == "agent.access.secret"
    write = lambda: ws.resolve(f"src/{name}", permission="write")  # noqa: E731
    assert _code(write) == "agent.access.secret"
    # Хвостовые точки Windows не обходят исключение.
    assert ws.is_secret(".env.") and ws.is_secret(".env ")
    assert not ws.is_secret("environment.py")


def test_git_is_readable_but_not_writable(tree: tuple[Path, Path]) -> None:
    root, _ = tree
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    ws = Workspace(root, Permissions(write=True))
    assert ws.resolve(".git/HEAD").name == "HEAD"
    hook = lambda: ws.resolve(".Git/hooks/pre-commit", permission="write", must_exist=False)  # noqa: E731
    assert _code(hook) == "agent.access.git_write"


def test_permissions_are_separate_and_revocable(tree: tuple[Path, Path]) -> None:
    root, _ = tree
    reader = Workspace(root)
    assert reader.resolve("README.md")
    write = lambda: reader.resolve("README.md", permission="write")  # noqa: E731
    assert _code(write) == "agent.access.permission"
    for permission in ("terminal", "network"):
        assert _code(lambda: reader.require(permission)) == "agent.access.permission"  # noqa: B023
    writer = Workspace(root, Permissions(read=False, write=True))
    assert _code(lambda: writer.resolve("README.md")) == "agent.access.permission"
    assert writer.resolve("new/dir/file.txt", permission="write", must_exist=False) == (
        writer.root / "new" / "dir" / "file.txt"
    )
    reader.revoke()
    assert reader.revoked
    assert _code(lambda: reader.resolve("README.md")) == "agent.access.revoked"
    assert _code(lambda: reader.open_read("README.md")) == "agent.access.revoked"


def test_grant_root_must_be_an_existing_folder(tree: tuple[Path, Path]) -> None:
    root, _ = tree
    assert _code(lambda: Workspace("project")) == "agent.access.root_relative"
    assert _code(lambda: Workspace(root / "nope")) == "agent.access.root_missing"
    assert _code(lambda: Workspace(root / "README.md")) == "agent.access.root_not_directory"
    assert _code(lambda: Workspace(Path(root.anchor))) == "agent.access.root_is_drive"


def test_open_read_rejects_directories_and_swapped_files(
    tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, outside = tree
    ws = Workspace(root)
    assert _code(lambda: ws.open_read("src")) == "agent.access.not_file"
    # Подмена между проверкой и открытием: открылся другой файл — чтение отменено.
    real_open = os.open

    def swapped(path: object, flags: int, *args: object) -> int:
        return real_open(outside / "secret.txt", flags)

    monkeypatch.setattr(os, "open", swapped)
    assert _code(lambda: ws.open_read("README.md")) == "agent.access.changed"
