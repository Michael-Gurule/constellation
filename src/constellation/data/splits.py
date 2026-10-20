"""Split utilities that keep time order and group membership intact."""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from constellation.core.errors import ConfigError
from constellation.core.rng import make_rng

Split = tuple[np.ndarray, np.ndarray]


def _length(index: pd.Index | int) -> int:
    if isinstance(index, int):
        return index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ConfigError("a temporal split needs an index sorted in time")
    return len(index)


def temporal_split(
    index: pd.Index | int, fractions: Sequence[float], gap: int = 0
) -> list[np.ndarray]:
    """Cut positions `0..n-1` into consecutive blocks, in time order.

    `fractions` must sum to 1. `gap` positions are dropped at the start of every
    block after the first, so a window that overlaps a block boundary cannot see
    data from the previous block. Returns one array of positions per block.
    """
    n = _length(index)
    if (
        not fractions
        or any(f <= 0 for f in fractions)
        or abs(sum(fractions) - 1.0) > 1e-9
    ):
        raise ConfigError("fractions must be positive and sum to 1")
    if gap < 0:
        raise ConfigError("gap must not be negative")
    edges = np.round(np.cumsum([0.0, *fractions]) * n).astype(int)
    blocks = []
    for position, (start, stop) in enumerate(pairwise(edges)):
        start = start + gap if position > 0 else start
        if stop <= start:
            raise ConfigError(
                "a block is empty; use fewer blocks, larger fractions or a smaller gap"
            )
        blocks.append(np.arange(start, stop))
    return blocks


def rolling_origin(n: int, horizon: int, folds: int, gap: int = 0) -> list[Split]:
    """Expanding-window folds for forecasting backtests.

    The last `folds` blocks of `horizon` positions are the test sets, in order.
    The training set of a fold is everything before its test block, minus `gap`
    positions directly in front of it (so targets that reach `gap` steps ahead
    never touch the test block).
    """
    if horizon < 1 or folds < 1 or gap < 0:
        raise ConfigError(
            "horizon and folds must be positive and gap must not be negative"
        )
    first_test = n - folds * horizon
    if first_test - gap < 1:
        raise ConfigError(
            f"{n} samples are not enough for {folds} folds of {horizon} with gap {gap}"
        )
    result: list[Split] = []
    for fold in range(folds):
        start = first_test + fold * horizon
        train = np.arange(0, start - gap)
        test = np.arange(start, start + horizon)
        result.append((train, test))
    return result


def group_kfold(
    groups: Sequence[object] | np.ndarray, k: int, seed: int
) -> list[Split]:
    """K folds in which every group falls entirely into one test fold.

    Groups are assigned largest first to the currently smallest fold, which keeps
    the folds balanced; the order among equal groups depends on `seed`.
    """
    labels = np.asarray(groups)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if k < 2 or k > len(unique):
        raise ConfigError(
            f"k must be between 2 and the number of groups ({len(unique)})"
        )
    rng = make_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k, dtype=int)
    fold_of_group = np.empty(len(unique), dtype=int)
    for group in order:
        fold = int(
            np.flatnonzero(load == load.min())[
                rng.integers(0, int((load == load.min()).sum()))
            ]
        )
        fold_of_group[group] = fold
        load[fold] += counts[group]
    fold_of_row = fold_of_group[inverse]
    positions = np.arange(len(labels))
    return [
        (positions[fold_of_row != f], positions[fold_of_row == f]) for f in range(k)
    ]
