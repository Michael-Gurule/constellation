"""Latency budget for the feature registry.

One week of 10-second frames for 50 channels, with the shipped feature set and
orbit features, must be computed in under 10 seconds on a laptop.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constellation import orbit
from constellation.features.registry import FeatureSet, compute
from constellation.ingest.channels import load_channels

ROOT = Path(__file__).resolve().parents[1]
BUDGET_SECONDS = 10.0
CHANNELS = 50


@pytest.fixture(scope="module")
def week():
    feature_set = FeatureSet.from_yaml(ROOT / "configs/features/default.yaml")
    names = [c.name for c in load_channels(ROOT / "configs/ingest/iss.yaml").channels]
    typed = [n for n in names if feature_set.channel_type(n)]
    untyped = [n for n in names if not feature_set.channel_type(n)]
    channels = (typed + untyped)[:CHANNELS]
    assert len(channels) == CHANNELS

    rng = np.random.default_rng(7)
    index = pd.date_range(
        "2026-10-12", periods=7 * 8640, freq="10s", tz="UTC", name="t"
    )
    data = {c: 100.0 + rng.normal(size=len(index)).cumsum() * 0.01 for c in channels}
    frame = pd.DataFrame(data, index=index)
    frame["valid_fraction"] = 1.0
    tle = orbit.load_tle((ROOT / "tests/fixtures/iss_25544.tle").read_text())
    return feature_set, {"ISS": frame}, {"ISS": tle}


def test_one_week_of_fifty_channels_is_computed_within_budget(benchmark, week):
    feature_set, frames, tles = week

    result = benchmark.pedantic(
        compute, args=(frames, feature_set, tles), rounds=3, iterations=1
    )

    assert result.shape[0] == 7 * 8640
    assert result.shape[1] > 200
    assert benchmark.stats["max"] < BUDGET_SECONDS
