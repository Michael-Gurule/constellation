"""The feature registry: one declaration of every feature, for training and serving.

A `FeatureSet` maps channel names to channel *types* (temperature, speed, ...)
and lists `FeatureSpec`s. A spec names a window function and the channel types
it may be applied to; applying it to a channel of another type is an error, so
a feature built for one kind of signal cannot silently run on another.

`FeatureSet.feature_names(channels)` and `compute(...)` share one planning step,
so the column order is a pure function of the feature set and the channels.
"""

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from constellation import orbit
from constellation.core.errors import ConfigError
from constellation.features import windows

Kind = Literal["unary", "binary", "asset"]


def _level(x: pd.Series) -> pd.Series:
    return x


def _difference(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


@dataclass(frozen=True)
class FunctionInfo:
    function: Callable[..., Any]
    kind: Kind


FUNCTIONS: dict[str, FunctionInfo] = {
    "level": FunctionInfo(_level, "unary"),
    "roll_mean": FunctionInfo(windows.roll_mean, "unary"),
    "roll_std": FunctionInfo(windows.roll_std, "unary"),
    "roll_min": FunctionInfo(windows.roll_min, "unary"),
    "roll_max": FunctionInfo(windows.roll_max, "unary"),
    "roll_slope": FunctionInfo(windows.roll_slope, "unary"),
    "robust_z": FunctionInfo(windows.robust_z, "unary"),
    "rate_per_s": FunctionInfo(windows.rate_per_s, "unary"),
    "roll_changes": FunctionInfo(windows.roll_changes, "unary"),
    "safe_ratio": FunctionInfo(windows.safe_ratio, "binary"),
    "difference": FunctionInfo(_difference, "binary"),
    "orbit_sunlit": FunctionInfo(lambda f: f["sunlit"], "asset"),
    "orbit_eclipse_phase": FunctionInfo(lambda f: f["eclipse_phase"], "asset"),
}
_ARITY = {"unary": 1, "binary": 2, "asset": 1}


class TypeRule(BaseModel):
    """Channels whose name fully matches `match` have channel type `type`."""

    match: str
    type: str


class FeatureSpec(BaseModel):
    """One feature: a function, its parameters and where it may be applied.

    - `unary`: applied to every channel whose type is in `applies_to`, or only to
      `inputs` when given (each of which must then have a listed type). The
      column is `<channel>__<name>`.
    - `binary`: `inputs` are two channels and `applies_to` the type each must
      have, in order. The column is `name`.
    - `asset`: derived from the asset's orbit (needs a TLE). The column is `name`.

    An `optional` spec is skipped when its channels or orbit are unavailable.
    """

    name: str
    function: str
    params: dict[str, Any] = Field(default_factory=dict)
    applies_to: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    optional: bool = False

    @model_validator(mode="after")
    def _check_shape(self) -> "FeatureSpec":
        if self.function not in FUNCTIONS:
            raise ValueError(
                f"unknown function {self.function!r}; known: {sorted(FUNCTIONS)}"
            )
        kind = FUNCTIONS[self.function].kind
        if kind == "unary" and not self.applies_to:
            raise ValueError(f"{self.name}: a per-channel feature needs applies_to")
        if kind == "binary" and (len(self.inputs) != 2 or len(self.applies_to) != 2):
            raise ValueError(
                f"{self.name}: needs two inputs and one applies_to type for each"
            )
        if kind == "asset" and (self.inputs or self.applies_to):
            raise ValueError(f"{self.name}: an orbit feature takes no channels")
        try:
            inspect.signature(FUNCTIONS[self.function].function).bind(
                *([None] * _ARITY[kind]), **self.params
            )
        except TypeError as error:
            raise ValueError(
                f"{self.name}: bad params for {self.function}: {error}"
            ) from error
        return self


@dataclass(frozen=True)
class Planned:
    """A feature column and how to compute it."""

    column: str
    spec: FeatureSpec
    channels: tuple[str, ...]


class FeatureSet(BaseModel):
    """Channel types and the feature specs applied to them."""

    channel_types: list[TypeRule] = Field(default_factory=list)
    features: list[FeatureSpec]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FeatureSet":
        """Build from a mapping, expanding `foreach: {i: [1, 2]}` into one spec per value."""
        features: list[dict[str, Any]] = []
        for spec in raw.get("features", []):
            spec = dict(spec)
            loop = spec.pop("foreach", None)
            if loop is None:
                features.append(spec)
                continue
            (variable, values), *rest = loop.items()
            if rest:
                raise ConfigError("foreach supports one variable")
            for value in values:
                features.append(_substitute(spec, variable, value))
        try:
            return cls.model_validate({**raw, "features": features})
        except ValueError as error:
            raise ConfigError(f"invalid feature set: {error}") from error

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureSet":
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"cannot read feature set {path}: {error}") from error
        if not isinstance(raw, dict):
            raise ConfigError(f"feature set {path} must be a mapping")
        return cls.from_mapping(raw)

    def channel_type(self, channel: str) -> str | None:
        for rule in self.channel_types:
            if re.fullmatch(rule.match, channel):
                return rule.type
        return None

    def plan(
        self, channels: list[str], *, orbit_available: bool = False
    ) -> list[Planned]:
        """The feature columns for these channels, in a fixed order.

        Specs are taken in the order declared; within a spec, channels are
        taken in sorted order. Raises `ConfigError` if a spec is applied to a
        channel without one of its type tags.
        """
        available = sorted(channels)
        types = {c: self.channel_type(c) for c in available}
        planned: list[Planned] = []
        for spec in self.features:
            kind = FUNCTIONS[spec.function].kind
            if kind == "asset":
                if orbit_available:
                    planned.append(Planned(spec.name, spec, ()))
                elif not spec.optional:
                    raise ConfigError(f"{spec.name} needs an orbit (TLE) for the asset")
            elif kind == "binary":
                missing = [c for c in spec.inputs if c not in available]
                if missing:
                    if spec.optional:
                        continue
                    raise ConfigError(f"{spec.name}: channels not available: {missing}")
                for channel, tag in zip(spec.inputs, spec.applies_to, strict=True):
                    _require_tag(spec, channel, types[channel], [tag])
                planned.append(Planned(spec.name, spec, tuple(spec.inputs)))
            else:
                if spec.inputs:
                    missing = [c for c in spec.inputs if c not in available]
                    if missing and not spec.optional:
                        raise ConfigError(
                            f"{spec.name}: channels not available: {missing}"
                        )
                    targets = [c for c in spec.inputs if c in available]
                    for channel in targets:
                        _require_tag(spec, channel, types[channel], spec.applies_to)
                else:
                    targets = [c for c in available if types[c] in spec.applies_to]
                planned.extend(
                    Planned(f"{channel}__{spec.name}", spec, (channel,))
                    for channel in sorted(targets)
                )
        names = [p.column for p in planned]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigError(f"duplicate feature names: {duplicates}")
        return planned

    def feature_names(
        self, channels: list[str], *, orbit_available: bool = False
    ) -> list[str]:
        """Column names `compute` produces for these channels."""
        return [p.column for p in self.plan(channels, orbit_available=orbit_available)]


def _substitute(value: Any, variable: str, replacement: Any) -> Any:
    if isinstance(value, str):
        return value.replace("{" + variable + "}", str(replacement))
    if isinstance(value, list):
        return [_substitute(v, variable, replacement) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, variable, replacement) for k, v in value.items()}
    return value


def _require_tag(
    spec: FeatureSpec, channel: str, actual: str | None, allowed: list[str]
) -> None:
    if actual not in allowed:
        raise ConfigError(
            f"{spec.name}: applies to channel types {allowed}, but {channel!r} "
            f"has type {actual!r}"
        )


def compute(
    frames: Mapping[str, pd.DataFrame],
    feature_set: FeatureSet,
    tles: Mapping[str, orbit.Tle] | None = None,
) -> pd.DataFrame:
    """Compute the feature set for every asset frame.

    Returns one frame indexed by `(asset_id, t)` whose columns are the feature
    names, in the deterministic order of the plan; the names are also stored in
    `result.attrs["feature_names"]`. Every feature is causal: its value at `t`
    uses only frame rows at or before `t`.
    """
    tles = tles or {}
    channels_by_asset = {
        asset: [c for c in frame.columns if c != "valid_fraction"]
        for asset, frame in frames.items()
    }
    union = sorted({c for channels in channels_by_asset.values() for c in channels})
    names = feature_set.feature_names(union, orbit_available=bool(tles))

    parts: list[pd.DataFrame] = []
    for asset in sorted(frames):
        frame = frames[asset]
        plan = feature_set.plan(channels_by_asset[asset], orbit_available=asset in tles)
        columns: dict[str, pd.Series] = {}
        orbit_frame = (
            orbit.orbit_features(tles[asset], pd.DatetimeIndex(frame.index))
            if asset in tles
            else None
        )
        for item in plan:
            info = FUNCTIONS[item.spec.function]
            if info.kind == "asset":
                assert orbit_frame is not None
                columns[item.column] = info.function(orbit_frame, **item.spec.params)
            else:
                inputs = [frame[channel] for channel in item.channels]
                columns[item.column] = info.function(*inputs, **item.spec.params)
        part = pd.DataFrame(columns, index=frame.index).reindex(columns=names)
        part.index = pd.MultiIndex.from_arrays(
            [pd.Index([asset] * len(part)), part.index], names=["asset_id", "t"]
        )
        parts.append(part)

    result = pd.concat(parts) if parts else pd.DataFrame(columns=names)
    result.attrs["feature_names"] = names
    return result
