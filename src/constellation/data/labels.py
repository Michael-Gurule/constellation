"""Anomaly labels as masks, half-open intervals and the LABELS contract.

Intervals are half-open index pairs `(start, stop)`: `start` is the first
anomalous position and `stop` the first normal one after it.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from constellation.core.schemas import empty_labels, validate_labels

Interval = tuple[int, int]


def intervals_from_mask(mask: Sequence[bool] | np.ndarray) -> list[Interval]:
    """Runs of True values as half-open intervals."""
    flags = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], flags, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(change[::2], change[1::2], strict=True)]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Sort intervals and join those that overlap or touch."""
    merged: list[Interval] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def mask_from_intervals(n: int, intervals: Iterable[Interval]) -> np.ndarray:
    """A boolean mask of length `n` that is True inside the intervals."""
    mask = np.zeros(n, dtype=bool)
    for start, stop in intervals:
        mask[max(start, 0) : max(min(stop, n), 0)] = True
    return mask


def labels_frame(
    intervals: Iterable[Interval],
    *,
    asset_id: str,
    channel: str | None,
    start: pd.Timestamp,
    cadence: pd.Timedelta,
    source: str,
    kind: str = "anomaly",
) -> pd.DataFrame:
    """Convert index intervals into the LABELS contract.

    Position `i` is at time `start + i * cadence`. A label runs from its first
    anomalous sample to its last one, both inclusive.
    """
    rows = [
        {
            "asset_id": asset_id,
            "channel": channel,
            "start": start + a * cadence,
            "end": start + (b - 1) * cadence,
            "kind": kind,
            "fault_class": None,
            "severity": None,
            "source": source,
        }
        for a, b in intervals
    ]
    if not rows:
        return empty_labels()
    frame = pd.DataFrame(rows)
    for column in ("start", "end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    for column in ("channel", "fault_class", "severity"):
        frame[column] = frame[column].astype(object)
    return validate_labels(frame)
