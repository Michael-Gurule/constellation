"""YAML configuration loading and stable configuration hashes."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from constellation.core.errors import ConfigError


def load_config[ModelT: BaseModel](path: str | Path, model: type[ModelT]) -> ModelT:
    """Read a YAML file and validate it against a pydantic model."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must contain a mapping at the top level")
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid config {path}:\n{error}") from error


def canonical_json(value: BaseModel | Mapping[str, Any]) -> str:
    """Serialize to JSON with sorted keys and no insignificant whitespace."""
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(value: BaseModel | Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form. Independent of key order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
