"""8.3 (PORT-04): ключ выпуска и подписанный пакет обновления RAMVI.

    python packaging/make_update.py keygen --out release-key.hex
    python packaging/make_update.py build --bundle <папка комплекта> --version 0.8.1 \
        --key release-key.hex --out RAMVI-update-0.8.1.zip [--previous <старый комплект>]

Закрытый ключ хранится только у того, кто выпускает обновления, и никогда не кладётся
в репозиторий. Открытый ключ (печатается при keygen) вписывается в
`ramvi_service.updates.TRUSTED_KEYS`. В пакет попадают только разрешённые части
комплекта (`ramvi/`, `runtime/`, `modules/`, `RAMVI.exe`, `BUILD.json`, `LICENSE`);
с `--previous` — только изменившиеся файлы и список удалённых.
"""

import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ramvi_service import ed25519  # noqa: E402
from ramvi_service.updates import ALLOWED_DIRS, ALLOWED_FILES, PRODUCT, SCHEMA  # noqa: E402


def bundle_files(bundle: Path) -> dict[str, Path]:
    """Файлы комплекта, которые может менять обновление (без кешей Python)."""
    found: dict[str, Path] = {}
    for name in ALLOWED_FILES:
        if (bundle / name).is_file():
            found[name] = bundle / name
    for top in ALLOWED_DIRS:
        for path in sorted((bundle / top).rglob("*")) if (bundle / top).is_dir() else []:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                found[path.relative_to(bundle).as_posix()] = path
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(
    bundle: Path,
    version: str,
    secret: bytes,
    out: Path,
    previous: Path | None = None,
    notes: str = "",
) -> dict[str, object]:
    current = bundle_files(bundle)
    old = bundle_files(previous) if previous is not None else {}
    files, chosen = [], []
    for path, source in current.items():
        digest = _sha256(source)
        if path in old and _sha256(old[path]) == digest:
            continue
        files.append({"path": path, "sha256": digest, "size": source.stat().st_size})
        chosen.append((path, source))
    manifest = {
        "schema": SCHEMA,
        "product": PRODUCT,
        "version": version,
        "files": files,
        "remove": sorted(set(old) - set(current)),
        "notes": notes,
    }
    raw = json.dumps(manifest, ensure_ascii=False, indent=1).encode()
    signature = base64.b64encode(ed25519.sign(secret, raw)).decode("ascii")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", raw)
        package.writestr("manifest.sig", signature)
        for path, source in chosen:
            package.write(source, "payload/" + path)
    return {"version": version, "files": len(files), "remove": len(manifest["remove"])}  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(prog="make_update")
    sub = parser.add_subparsers(dest="command", required=True)
    keygen = sub.add_parser("keygen")
    keygen.add_argument("--out", required=True, type=Path)
    build = sub.add_parser("build")
    build.add_argument("--bundle", required=True, type=Path)
    build.add_argument("--version", required=True)
    build.add_argument("--key", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--previous", type=Path)
    build.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.command == "keygen":
        secret = secrets.token_bytes(32)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secret.hex())
        print(
            json.dumps(
                {"public_key": ed25519.public_key(secret).hex(), "secret_file": str(args.out)}
            )
        )
        return
    secret = bytes.fromhex(args.key.read_text().strip())
    print(
        json.dumps(
            build_package(args.bundle, args.version, secret, args.out, args.previous, args.notes)
        )
    )


if __name__ == "__main__":
    main()
