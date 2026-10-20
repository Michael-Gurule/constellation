"""Split utilities never leak a position or a group across the boundary."""

from itertools import pairwise

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from constellation.data.splits import group_kfold, rolling_origin, temporal_split


@given(
    n=st.integers(50, 400),
    horizon=st.integers(1, 20),
    folds=st.integers(1, 5),
    gap=st.integers(0, 15),
)
def test_rolling_origin_never_mixes_train_and_test_and_respects_the_gap(
    n, horizon, folds, gap
):
    assume(n - folds * horizon - gap >= 1)

    result = rolling_origin(n, horizon, folds, gap)

    for train, test in result:
        assert not set(train) & set(test)
        assert test[0] - train[-1] > gap  # at least `gap` positions in between
        assert train.max() < test.min()
        assert test.max() < n
    ends = [test[-1] for _, test in result]
    assert ends == sorted(ends)
    assert ends[-1] == n - 1


@given(
    n=st.integers(20, 500),
    cuts=st.lists(st.floats(0.05, 1.0), min_size=2, max_size=4),
    gap=st.integers(0, 3),
)
def test_temporal_blocks_are_ordered_and_disjoint(n, cuts, gap):
    fractions = [c / sum(cuts) for c in cuts]
    assume(min(fractions) * n > gap + 2)

    blocks = temporal_split(n, fractions, gap=gap)

    flat = np.concatenate(blocks)
    assert len(flat) == len(set(flat))
    assert all(a[-1] < b[0] for a, b in pairwise(blocks))


@given(
    sizes=st.lists(st.integers(1, 12), min_size=3, max_size=30),
    seed=st.integers(0, 10_000),
    data=st.data(),
)
def test_group_kfold_never_splits_a_group(sizes, seed, data):
    k = data.draw(st.integers(2, len(sizes)))
    groups = np.repeat(np.arange(len(sizes)), sizes)

    folds = group_kfold(groups, k, seed)

    covered = np.concatenate([test for _, test in folds])
    assert sorted(covered) == list(range(len(groups)))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert len(test) > 0
