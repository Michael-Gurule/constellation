"""Column contracts for readings and labels, and validators for them."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pyarrow as pa

from constellation.core.errors import FeatureMismatchError, SchemaError

__all__ = [
    "LABELS",
    "LABEL_KINDS",
    "READINGS",
    "READINGS_ARROW",
    "SOURCES",
    "ColumnSpec",
    "FeatureMismatchError",
    "SchemaError",
    "empty_labels",
    "empty_readings",
    "validate_labels",
    "validate_readings",
]

Kind = Literal["str", "float", "time"]

SOURCES = ("iss", "satnogs", "sim", "benchmark")
LABEL_KINDS = ("anomaly", "fault", "failure")


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a contract: its name, logical type and nullability."""

    name: str
    kind: Kind
    nullable: bool = False


READINGS: tuple[ColumnSpec, ...] = (
    ColumnSpec("source", "str"),
    ColumnSpec("asset_id", "str"),
    ColumnSpec("channel", "str"),
    ColumnSpec("t", "time"),
    ColumnSpec("value", "float", nullable=True),
    ColumnSpec("value_text", "str", nullable=True),
    ColumnSpec("status", "str", nullable=True),
    ColumnSpec("ingested_at", "time"),
)

LABELS: tuple[ColumnSpec, ...] = (
    ColumnSpec("asset_id", "str"),
    ColumnSpec("channel", "str", nullable=True),
    ColumnSpec("start", "time"),
    ColumnSpec("end", "time"),
    ColumnSpec("kind", "str"),
    ColumnSpec("fault_class", "str", nullable=True),
    ColumnSpec("severity", "str", nullable=True),
    ColumnSpec("source", "str"),
)

_UTC_NS = pa.timestamp("ns", tz="UTC")

READINGS_ARROW = pa.schema(
    [
        ("source", pa.string()),
        ("asset_id", pa.string()),
        ("channel", pa.string()),
        ("t", _UTC_NS),
        ("value", pa.float64()),
        ("value_text", pa.string()),
        ("status", pa.string()),
        ("ingested_at", _UTC_NS),
    ]
)


def _empty(spec: tuple[ColumnSpec, ...]) -> pd.DataFrame:
    dtypes: dict[Kind, str] = {
        "str": "object",
        "float": "float64",
        "time": "datetime64[ns, UTC]",
    }
    return pd.DataFrame(
        {c.name: pd.Series(dtype=dtypes[c.kind]) for c in spec},
    )


def empty_readings() -> pd.DataFrame:
    """A readings frame with no rows and the contract's dtypes."""
    return _empty(READINGS)


def empty_labels() -> pd.DataFrame:
    """A labels frame with no rows and the contract's dtypes."""
    return _empty(LABELS)


def _check_column(df: pd.DataFrame, spec: ColumnSpec) -> list[str]:
    series = df[spec.name]
    problems: list[str] = []

    if spec.kind == "time":
        if str(series.dtype) != "datetime64[ns, UTC]":
            return [f"{spec.name}: expected datetime64[ns, UTC], got {series.dtype}"]
    elif spec.kind == "float":
        if series.dtype != "float64":
            return [f"{spec.name}: expected float64, got {series.dtype}"]
    else:
        if not (series.dtype == object or pd.api.types.is_string_dtype(series.dtype)):
            return [f"{spec.name}: expected strings, got {series.dtype}"]
        present = series.dropna()
        if not present.map(lambda v: isinstance(v, str)).all():
            problems.append(f"{spec.name}: contains non-string values")

    if not spec.nullable and series.isna().any():
        problems.append(f"{spec.name}: contains {int(series.isna().sum())} null values")
    return problems


def _validate(df: pd.DataFrame, spec: tuple[ColumnSpec, ...]) -> list[str]:
    expected = [c.name for c in spec]
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    problems: list[str] = []
    if missing:
        problems.append(f"missing columns: {missing}")
    if unexpected:
        problems.append(f"unexpected columns: {list(unexpected)}")
    if missing:
        return problems
    for column in spec:
        problems.extend(_check_column(df, column))
    return problems


def _raise(name: str, problems: list[str]) -> None:
    if problems:
        raise SchemaError(f"{name} frame is invalid:\n  " + "\n  ".join(problems))


def validate_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Check a frame against the READINGS contract and return it unchanged."""
    problems = _validate(df, READINGS)
    if not problems:
        bad = sorted(set(df["source"]) - set(SOURCES))
        if bad:
            problems.append(f"source: unknown values {bad}; allowed {list(SOURCES)}")
        if (df["asset_id"] == "").any() or (df["channel"] == "").any():
            problems.append("asset_id and channel must not be empty")
    _raise("readings", problems)
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Check a frame against the LABELS contract and return it unchanged."""
    problems = _validate(df, LABELS)
    if not problems:
        bad = sorted(set(df["kind"]) - set(LABEL_KINDS))
        if bad:
            problems.append(f"kind: unknown values {bad}; allowed {list(LABEL_KINDS)}")
        if (df["end"] < df["start"]).any():
            problems.append("end must not be earlier than start")
    _raise("labels", problems)
    return df
