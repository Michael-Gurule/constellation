import numpy as np
import pandas as pd

from constellation.core.schemas import validate_labels
from constellation.data.labels import (
    intervals_from_mask,
    labels_frame,
    mask_from_intervals,
    merge_intervals,
)


def test_mask_and_intervals_round_trip():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)

    intervals = intervals_from_mask(mask)

    assert intervals == [(1, 3), (5, 6), (7, 10)]
    assert np.array_equal(mask_from_intervals(len(mask), intervals), mask)


def test_edge_cases_of_masks():
    assert intervals_from_mask([]) == []
    assert intervals_from_mask([False, False]) == []
    assert intervals_from_mask([True, True]) == [(0, 2)]
    assert mask_from_intervals(5, [(-3, 2), (4, 99)]).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert not mask_from_intervals(4, [(3, 1)]).any()


def test_merge_joins_overlapping_and_touching_intervals():
    assert merge_intervals([(10, 12), (0, 3), (3, 5), (4, 6), (20, 21)]) == [
        (0, 6),
        (10, 12),
        (20, 21),
    ]
    assert merge_intervals([]) == []


def test_labels_frame_follows_the_labels_contract():
    frame = labels_frame(
        [(2, 5), (10, 11)],
        asset_id="SMAP",
        channel="P-1",
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        cadence=pd.Timedelta(60, unit="s"),
        source="smap_msl",
    )

    validate_labels(frame)
    assert frame["start"].tolist() == [
        pd.Timestamp("2000-01-01T00:02:00", tz="UTC"),
        pd.Timestamp("2000-01-01T00:10:00", tz="UTC"),
    ]
    # `end` is the last anomalous sample, inclusive.
    assert frame["end"].tolist() == [
        pd.Timestamp("2000-01-01T00:04:00", tz="UTC"),
        pd.Timestamp("2000-01-01T00:10:00", tz="UTC"),
    ]
    assert set(frame["kind"]) == {"anomaly"}
    assert frame["source"].tolist() == ["smap_msl", "smap_msl"]


def test_labels_frame_without_intervals_is_an_empty_valid_frame():
    frame = labels_frame(
        [],
        asset_id="A",
        channel=None,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        cadence=pd.Timedelta(1, unit="s"),
        source="x",
    )

    assert frame.empty
    validate_labels(frame)


def test_asset_wide_labels_have_a_null_channel():
    frame = labels_frame(
        [(0, 1)],
        asset_id="A",
        channel=None,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        cadence=pd.Timedelta(1, unit="s"),
        source="x",
        kind="fault",
    )

    assert frame["channel"].isna().all()
    assert frame["kind"].tolist() == ["fault"]
