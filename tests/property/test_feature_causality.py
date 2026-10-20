"""Features at time t depend only on data at or before t.

For random inputs, replace every value after a cut point with different values
and check that every feature at or before the cut is unchanged.
"""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from constellation.features.frames import build_frames
from constellation.features.registry import FeatureSet, compute

TYPES = [
    {"match": "speed", "type": "speed"},
    {"match": "current", "type": "current"},
    {"match": "mode", "type": "state"},
]


def _spec(name, function, tags, **params):
    return {"name": name, "function": function, "params": params, "applies_to": tags}


FEATURE_SET = FeatureSet.from_mapping(
    {
        "channel_types": TYPES,
        "features": [
            _spec("mean", "roll_mean", ["speed", "current"], window="5min"),
            _spec("std", "roll_std", ["speed", "current"], window="5min"),
            _spec("min", "roll_min", ["speed"], window="5min"),
            _spec("max", "roll_max", ["speed"], window="5min"),
            _spec("slope", "roll_slope", ["speed", "current"], window="5min"),
            _spec("rz", "robust_z", ["speed"], window="10min", min_periods=5),
            _spec("rate", "rate_per_s", ["current"]),
            _spec("changes", "roll_changes", ["state"], window="5min"),
            _spec("code", "level", ["state"]),
            {
                "name": "ratio",
                "function": "safe_ratio",
                "params": {"eps": 1e-3},
                "inputs": ["current", "speed"],
                "applies_to": ["current", "speed"],
            },
            {
                "name": "gap",
                "function": "difference",
                "inputs": ["speed", "current"],
                "applies_to": ["speed", "current"],
            },
        ],
    }
)

VALUES = st.one_of(
    st.just(float("nan")),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)


@st.composite
def frame_pairs(draw):
    """A frame, and a copy whose rows after a cut point are different."""
    n = draw(st.integers(min_value=20, max_value=80))
    cut = draw(st.integers(min_value=5, max_value=n - 5))
    index = pd.date_range("2026-03-01", periods=n, freq="10s", tz="UTC", name="t")
    columns = {}
    altered = {}
    for name in ("speed", "current", "mode"):
        values = np.array(draw(st.lists(VALUES, min_size=n, max_size=n)))
        future = np.array(
            draw(st.lists(VALUES, min_size=n - cut - 1, max_size=n - cut - 1))
        )
        columns[name] = values
        changed = values.copy()
        changed[cut + 1 :] = future
        altered[name] = changed
    return (
        pd.DataFrame(columns, index=index),
        pd.DataFrame(altered, index=index),
        index[cut],
    )


@given(frame_pairs())
def test_features_up_to_t_ignore_everything_after_t(pair):
    original, altered, cut = pair

    before = compute({"A": original}, FEATURE_SET)
    after = compute({"A": altered}, FEATURE_SET)

    mask = before.index.get_level_values("t") <= cut
    pd.testing.assert_frame_equal(before[mask], after[mask], check_exact=True)


@st.composite
def reading_pairs(draw):
    n = draw(st.integers(min_value=10, max_value=60))
    offsets = sorted(draw(st.lists(st.integers(0, 1800), min_size=n, max_size=n)))
    cut_offset = draw(st.integers(300, 1500))
    values = draw(st.lists(VALUES, min_size=n, max_size=n))
    other = draw(st.lists(VALUES, min_size=n, max_size=n))
    times = pd.Timestamp("2026-03-01", tz="UTC") + pd.to_timedelta(offsets, unit="s")

    def frame(vals):
        return pd.DataFrame(
            {
                "source": "sim",
                "asset_id": "A",
                "channel": "speed",
                "t": pd.DatetimeIndex(times).as_unit("ns"),
                "value": vals,
                "value_text": pd.Series([None] * n, dtype=object),
                "status": pd.Series([None] * n, dtype=object),
                "ingested_at": pd.Timestamp("2026-03-05", tz="UTC").as_unit("ns"),
            }
        )

    changed = [
        v if o <= cut_offset else w
        for o, v, w in zip(offsets, values, other, strict=True)
    ]
    return (
        frame(values),
        frame(changed),
        pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(cut_offset, unit="s"),
    )


@given(reading_pairs())
def test_frames_up_to_t_ignore_readings_after_t(pair):
    original, altered, cut = pair

    before = build_frames(original, "10s", "30s")["A"]
    after = build_frames(altered, "10s", "30s")["A"]

    common = before.index.intersection(after.index)
    common = common[common <= cut]
    assert len(common) > 0
    pd.testing.assert_frame_equal(
        before.loc[common], after.loc[common], check_exact=True
    )
