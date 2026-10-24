import numpy as np
import pandas as pd
import pytest

from constellation.core.errors import ConfigError
from constellation.data.splits import group_kfold, rolling_origin, temporal_split


def test_temporal_split_cuts_consecutive_blocks_in_order():
    blocks = temporal_split(100, [0.6, 0.2, 0.2])

    assert [len(b) for b in blocks] == [60, 20, 20]
    assert np.array_equal(np.concatenate(blocks), np.arange(100))


def test_temporal_split_accepts_a_sorted_time_index_and_drops_a_gap():
    index = pd.date_range("2026-03-01", periods=100, freq="1min", tz="UTC")

    train, test = temporal_split(index, [0.7, 0.3], gap=5)

    assert train[-1] == 69
    assert test[0] == 75  # five positions between the blocks are dropped
    assert test[-1] == 99


def test_temporal_split_rejects_bad_input():
    with pytest.raises(ConfigError, match="sum to 1"):
        temporal_split(10, [0.5, 0.4])
    with pytest.raises(ConfigError, match="positive"):
        temporal_split(10, [1.2, -0.2])
    with pytest.raises(ConfigError, match="sorted"):
        temporal_split(
            pd.DatetimeIndex(["2026-03-02", "2026-03-01"], tz="UTC"), [0.5, 0.5]
        )
    with pytest.raises(ConfigError, match="empty"):
        temporal_split(10, [0.5, 0.5], gap=5)
    with pytest.raises(ConfigError, match="negative"):
        temporal_split(10, [0.5, 0.5], gap=-1)


def test_rolling_origin_folds_expand_and_tile_the_end_of_the_series():
    folds = rolling_origin(100, horizon=10, folds=3)

    tests = [t for _, t in folds]
    assert [t[0] for t in tests] == [70, 80, 90]
    assert tests[-1][-1] == 99
    assert [len(train) for train, _ in folds] == [70, 80, 90]  # expanding window


def test_rolling_origin_gap_separates_train_from_test():
    folds = rolling_origin(100, horizon=10, folds=3, gap=4)

    for train, test in folds:
        assert test[0] - train[-1] == 5  # gap positions are skipped
        assert train[0] == 0


def test_rolling_origin_rejects_series_that_are_too_short():
    with pytest.raises(ConfigError, match="not enough"):
        rolling_origin(20, horizon=10, folds=2)
    with pytest.raises(ConfigError):
        rolling_origin(100, horizon=0, folds=2)


def test_group_kfold_keeps_every_group_in_one_fold():
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], [10, 20, 5, 15, 25, 25])

    folds = group_kfold(groups, k=3, seed=1)

    assert len(folds) == 3
    seen = np.concatenate([test for _, test in folds])
    assert sorted(seen) == list(range(len(groups)))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert len(train) + len(test) == len(groups)
    sizes = sorted(len(test) for _, test in folds)
    assert sizes[-1] - sizes[0] <= 25  # balanced to within the largest group


def test_group_kfold_depends_on_the_seed_only():
    groups = np.repeat(np.arange(20), 3)

    first = group_kfold(groups, 4, seed=7)
    again = group_kfold(groups, 4, seed=7)
    other = group_kfold(groups, 4, seed=8)

    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, again, strict=True))
    assert any(
        not np.array_equal(a[1], b[1]) for a, b in zip(first, other, strict=True)
    )


def test_group_kfold_needs_enough_groups():
    with pytest.raises(ConfigError, match="between 2 and"):
        group_kfold([1, 1, 2, 2], k=3, seed=0)
    with pytest.raises(ConfigError):
        group_kfold([1, 2, 3], k=1, seed=0)
