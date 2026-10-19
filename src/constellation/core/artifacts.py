"""Model artifacts: a serialized model plus a `metadata.json` that describes it.

Layout: `<models_dir>/<name>/<version>/model.joblib` (or `model.pt` for a torch
module, stored as a `state_dict`) and `metadata.json`. Pickle is never used
directly: scikit-learn style objects go through joblib and torch modules
through their state dict.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field

from constellation.core.config import canonical_json, config_hash
from constellation.core.errors import ArtifactError, FeatureMismatchError

METADATA_FILE = "metadata.json"
JOBLIB_FILE = "model.joblib"
TORCH_FILE = "model.pt"

_TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "lightgbm",
    "torch",
    "pyarrow",
    "constellation",
)


class ArtifactMetadata(BaseModel):
    """What is needed to trust, reproduce and safely reuse a saved model."""

    name: str
    version: str
    created_at: datetime
    git_sha: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str
    data_manifest_hashes: dict[str, str] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    feature_set_hash: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class LoadedArtifact:
    """A model object (or torch `state_dict`) and its metadata."""

    model: Any
    metadata: ArtifactMetadata


def feature_set_hash(feature_names: Sequence[str]) -> str:
    """Hash of the ordered feature list."""
    return hashlib.sha256(
        canonical_json({"features": list(feature_names)}).encode()
    ).hexdigest()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for library in _TRACKED_LIBRARIES:
        try:
            versions[library] = importlib_metadata.version(library)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def create_metadata(
    name: str,
    *,
    config: Mapping[str, Any] | BaseModel,
    feature_names: Sequence[str],
    data_manifest_hashes: Mapping[str, str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    metrics: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> ArtifactMetadata:
    """Build metadata for a new artifact, filling in version, hashes and libraries."""
    created = created_at or datetime.now(UTC)
    digest = config_hash(config)
    config_dict = (
        config.model_dump(mode="json")
        if isinstance(config, BaseModel)
        else dict(config)
    )
    return ArtifactMetadata(
        name=name,
        version=f"{created:%Y%m%d-%H%M%S}-{digest[:8]}",
        created_at=created,
        git_sha=_git_sha(),
        config=config_dict,
        config_hash=digest,
        data_manifest_hashes=dict(data_manifest_hashes or {}),
        feature_names=list(feature_names),
        feature_set_hash=feature_set_hash(feature_names),
        thresholds=dict(thresholds or {}),
        metrics=dict(metrics or {}),
        library_versions=_library_versions(),
    )


def artifact_dir(models_dir: str | Path, metadata: ArtifactMetadata) -> Path:
    """The canonical directory for an artifact: `<models_dir>/<name>/<version>`."""
    return Path(models_dir) / metadata.name / metadata.version


def save_artifact(obj: Any, directory: str | Path, metadata: ArtifactMetadata) -> Path:
    """Write the model and its metadata into `directory`, which must not exist yet.

    The files are written to a temporary sibling directory and moved into place,
    so a reader never sees a half-written artifact.
    """
    directory = Path(directory)
    if directory.exists():
        raise ArtifactError(f"artifact directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(obj, torch.nn.Module):
            torch.save(obj.state_dict(), staging / TORCH_FILE)
        else:
            joblib.dump(obj, staging / JOBLIB_FILE)
        (staging / METADATA_FILE).write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


def read_metadata(directory: str | Path) -> ArtifactMetadata:
    """Read an artifact's metadata without loading the model."""
    path = Path(directory) / METADATA_FILE
    try:
        return ArtifactMetadata.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise ArtifactError(
            f"cannot read artifact metadata at {path}: {error}"
        ) from error


def check_features(expected: Sequence[str], actual: Sequence[str]) -> None:
    """Raise `FeatureMismatchError` unless both lists match, in order."""
    if list(expected) == list(actual):
        return
    missing = [f for f in expected if f not in actual]
    unexpected = [f for f in actual if f not in expected]
    detail = (
        f"missing={missing}, unexpected={unexpected}"
        if missing or unexpected
        else "same features in a different order"
    )
    raise FeatureMismatchError(
        f"input features do not match the trained features: {detail}"
    )


def load_artifact(
    directory: str | Path, expected_features: Sequence[str] | None = None
) -> LoadedArtifact:
    """Load an artifact, optionally checking the input feature list first."""
    directory = Path(directory)
    metadata = read_metadata(directory)
    if expected_features is not None:
        check_features(metadata.feature_names, expected_features)

    joblib_path = directory / JOBLIB_FILE
    torch_path = directory / TORCH_FILE
    if joblib_path.exists():
        model: Any = joblib.load(joblib_path)
    elif torch_path.exists():
        import torch

        model = torch.load(torch_path, map_location="cpu", weights_only=True)
    else:
        raise ArtifactError(f"no model file in {directory}")
    return LoadedArtifact(model=model, metadata=metadata)
