"""Content manifests: a SHA-256 and size for every file under a directory."""

import hashlib
from pathlib import Path
from typing import TypedDict

from constellation.core.config import config_hash
from constellation.core.errors import ManifestError

MANIFEST_NAME = "manifest.json"


class ManifestEntry(TypedDict):
    sha256: str
    size: int


Manifest = dict[str, ManifestEntry]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.relative_to(root).as_posix() != MANIFEST_NAME
    )


def build_manifest(root: str | Path) -> Manifest:
    """Hash every file under `root`. Hidden files and `manifest.json` are skipped."""
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in _files(root)
    }


def manifest_hash(manifest: Manifest) -> str:
    """A single hash that identifies the whole manifest."""
    return config_hash(dict(manifest))


def verify_manifest(root: str | Path, manifest: Manifest) -> None:
    """Raise `ManifestError` if the files under `root` differ from `manifest`."""
    current = build_manifest(root)
    missing = sorted(set(manifest) - set(current))
    extra = sorted(set(current) - set(manifest))
    changed = sorted(
        name for name in set(manifest) & set(current) if manifest[name] != current[name]
    )
    if missing or extra or changed:
        raise ManifestError(
            f"files under {root} do not match the manifest: "
            f"missing={missing}, unexpected={extra}, changed={changed}"
        )
