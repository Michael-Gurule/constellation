"""Common types for the labeled anomaly benchmarks, and their download logic."""

import hashlib
import json
import os
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from pydantic import BaseModel

from constellation.core.config import load_config
from constellation.core.errors import ConfigError, StoreError

Interval = tuple[int, int]

CHUNK = 1 << 20


@dataclass(frozen=True)
class BenchmarkSeries:
    """One channel of a benchmark, split into training and test data.

    `test_labels` are half-open `(start, stop)` positions into `test`. `meta`
    carries what a consumer needs to place samples in time: `cadence_s` and
    `test_start` (ISO 8601, UTC), plus dataset-specific details.
    """

    dataset: str
    asset_id: str
    channel: str
    train: np.ndarray
    test: np.ndarray
    test_labels: list[Interval]
    meta: dict[str, Any] = field(default_factory=dict)


class Benchmark(Protocol):
    """A benchmark whose files are already on disk."""

    name: str

    def load(self, channels: Sequence[str] | None = None) -> list[BenchmarkSeries]:
        """Load the series, optionally only the named channels."""
        ...


class ManualDownloadError(ConfigError):
    """A file has no download URL and is not present: the owner must supply it."""


class BenchmarkFile(BaseModel):
    """One file of a benchmark download."""

    name: str
    url: str | None = None
    md5: str | None = None
    sha256: str | None = None
    size: int | None = None
    extract: bool = False
    note: str = ""
    # Directories that stand in for this file when it was delivered already extracted.
    provided_as: list[str] = []


class BenchmarkSource(BaseModel):
    """Where a benchmark comes from, and how to verify what was fetched."""

    title: str
    doi: str
    citation: str
    licence: str
    files: list[BenchmarkFile]
    manual: str = ""


class BenchmarkConfig(BaseModel):
    """`configs/data/benchmarks.yaml`."""

    benchmarks: dict[str, BenchmarkSource]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return load_config(path, BenchmarkConfig)


def file_digest(path: Path, algorithm: str) -> str:
    """Hex digest of a file, read in chunks."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: BenchmarkFile) -> dict[str, str | int]:
    """Check size, md5 and sha256 where the config provides them.

    Returns the observed `sha256`, `md5` and `size`. Raises `StoreError` on any mismatch.
    """
    size = path.stat().st_size
    if spec.size is not None and spec.size != size:
        raise StoreError(f"{path.name}: expected {spec.size} bytes, found {size}")
    observed: dict[str, str | int] = {"size": size}
    for algorithm in ("md5", "sha256"):
        value = file_digest(path, algorithm)
        expected = getattr(spec, algorithm)
        if expected is not None and expected.lower() != value:
            raise StoreError(
                f"{path.name}: {algorithm} mismatch (expected {expected}, got {value})"
            )
        observed[algorithm] = value
    return observed


def safe_extract(archive: Path, target: Path) -> list[str]:
    """Extract a zip file, refusing members that would land outside `target`."""
    target = target.resolve()
    names: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise StoreError(
                    f"{archive.name}: unsafe path in archive: {member.filename}"
                )
            names.append(member.filename)
        bundle.extractall(target)
    return names


def tree_digests(
    directory: Path, subdirectories: Sequence[str]
) -> dict[str, str | int]:
    """One sha256 over the files below `subdirectories`, with their count and size.

    The hash covers every file's relative path and content hash in sorted order, so
    it changes if a file is added, removed, renamed or altered. Hidden files such as
    `.DS_Store` are ignored.
    """
    digest = hashlib.sha256()
    count = 0
    size = 0
    for name in sorted(subdirectories):
        for path in sorted(p for p in (directory / name).rglob("*") if p.is_file()):
            relative = path.relative_to(directory)
            if any(part.startswith(".") for part in relative.parts):
                continue
            digest.update(
                f"{relative.as_posix()}:{file_digest(path, 'sha256')}\n".encode()
            )
            count += 1
            size += path.stat().st_size
    return {"sha256": digest.hexdigest(), "size": size, "files": count}


@dataclass(frozen=True)
class FetchedFile:
    """A verified file, or (with `tree=True`) a set of extracted directories."""

    name: str
    path: Path
    downloaded: bool
    digests: dict[str, str | int]
    tree: bool = False


def _download(
    client: httpx.Client,
    url: str,
    destination: Path,
    progress: Callable[[int], None] | None,
) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_bytes(CHUNK):
                    handle.write(block)
                    if progress is not None:
                        progress(len(block))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(
    name: str,
    source: BenchmarkSource,
    directory: Path,
    *,
    client: httpx.Client | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[FetchedFile]:
    """Download, verify and (for archives) extract the files of one benchmark.

    Files go to `directory` (normally `data/external/<name>/`). A file that is
    already there and passes verification is not downloaded again. A file
    without a URL must be placed there by hand; if it is missing,
    `ManualDownloadError` carries the instructions.
    """
    directory.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0, read=300.0))
    results: list[FetchedFile] = []
    try:
        for spec in source.files:
            path = directory / spec.name
            downloaded = False
            if not path.exists():
                if spec.provided_as and all(
                    (directory / d).is_dir() for d in spec.provided_as
                ):
                    label = "+".join(spec.provided_as)
                    results.append(
                        FetchedFile(
                            label,
                            directory,
                            False,
                            tree_digests(directory, spec.provided_as),
                            tree=True,
                        )
                    )
                    continue
                if spec.url is None:
                    raise ManualDownloadError(
                        f"{name}: {spec.name} must be downloaded by hand and saved to {path}"
                        + (
                            f" (or extracted so that {', '.join(spec.provided_as)} sit beside it)"
                            if spec.provided_as
                            else ""
                        )
                        + f". {source.manual or spec.note}"
                    )
                _download(http, spec.url, path, progress)
                downloaded = True
            try:
                digests = verify_file(path, spec)
            except StoreError:
                if downloaded:
                    path.unlink(
                        missing_ok=True
                    )  # never keep a download that failed its check
                raise
            if spec.extract and path.suffix == ".zip":
                safe_extract(path, directory)
            results.append(FetchedFile(spec.name, path, downloaded, digests))
    finally:
        if owns_client:
            http.close()
    return results


def update_manifest(
    path: Path, name: str, files: Sequence[FetchedFile]
) -> dict[str, Any]:
    """Record the checksums of a benchmark's downloaded files in the manifest."""
    manifest: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    manifest[name] = {
        item.name: {
            key: item.digests[key]
            for key in ("md5", "sha256", "size", "files")
            if key in item.digests
        }
        for item in files
    }
    manifest = dict(sorted(manifest.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return manifest


def load_manifest(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}
