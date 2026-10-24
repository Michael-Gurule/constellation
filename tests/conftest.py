"""Shared fixtures and Hypothesis configuration."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

# Deterministic property tests: the same examples on every run and machine.
settings.register_profile("default", max_examples=50, deadline=None, derandomize=True)
settings.load_profile("default")

ReadingsFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261019)


@pytest.fixture
def make_readings() -> ReadingsFactory:
    """Build a valid READINGS frame from `(t, channel, value)` rows."""

    def build(
        rows: list[tuple[str, str, float]],
        *,
        source: str = "iss",
        asset_id: str = "ISS",
    ) -> pd.DataFrame:
        times = pd.to_datetime(
            [r[0] for r in rows], utc=True, format="ISO8601"
        ).as_unit("ns")
        return pd.DataFrame(
            {
                "source": source,
                "asset_id": asset_id,
                "channel": [r[1] for r in rows],
                "t": times,
                "value": [r[2] for r in rows],
                "value_text": pd.Series([None] * len(rows), dtype=object),
                "status": pd.Series(["ok"] * len(rows), dtype=object),
                "ingested_at": pd.Timestamp("2026-03-05", tz="UTC").as_unit("ns"),
            }
        )

    return build
