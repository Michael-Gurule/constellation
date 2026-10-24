"""Fixed-cadence frames built from event-time readings."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from constellation.core.schemas import validate_readings
from constellation.core.timeutils import parse_duration


def _hold_for(
    channel: str, default: pd.Timedelta, overrides: Mapping[str, str | None]
) -> pd.Timedelta | None:
    if channel not in overrides:
        return default
    override = overrides[channel]
    return None if override is None else parse_duration(override)


def build_frames(
    readings: pd.DataFrame,
    cadence: str = "10s",
    max_hold: str = "60s",
    *,
    hold: Mapping[str, str | None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Resample readings onto a UTC grid, one frame per asset.

    Each frame has the grid as index `t`, one float column per channel, and a
    `valid_fraction` column. A grid point takes the last reading at or before
    it (sample and hold), but only if that reading is at most `max_hold` old;
    otherwise the value is NaN. `hold` overrides the limit per channel, with
    `None` meaning no limit (useful for state channels that change rarely).

    State channels (those with text values) are encoded as integer codes: the
    numeric value when the source provides one, otherwise the index of the text
    among the channel's sorted distinct values. They are listed in
    `frame.attrs["state_channels"]`.
    """
    validate_readings(readings)
    step = parse_duration(cadence)
    default_hold = parse_duration(max_hold)
    overrides = hold or {}
    frames: dict[str, pd.DataFrame] = {}

    for asset, group in readings.groupby("asset_id", sort=True):
        grid = pd.date_range(
            group["t"].min().floor(step),
            group["t"].max().ceil(step),
            freq=step,
            name="t",
            tz="UTC",
        ).as_unit("ns")
        columns: dict[str, pd.Series] = {}
        states: list[str] = []
        for channel, part in group.groupby("channel", sort=True):
            series, is_state = _channel_series(part)
            if is_state:
                states.append(str(channel))
            columns[str(channel)] = _sample_and_hold(
                series, grid, _hold_for(str(channel), default_hold, overrides)
            )
        frame = pd.DataFrame(columns, index=grid)
        frame["valid_fraction"] = frame.notna().mean(axis=1) if len(columns) else 0.0
        frame.attrs["state_channels"] = states
        frames[str(asset)] = frame
    return frames


def _channel_series(part: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Event-time series of one channel, with states turned into integer codes."""
    ordered = part.sort_values("t", kind="stable")
    is_state = bool(ordered["value_text"].notna().any())
    values = ordered["value"].to_numpy(dtype=float)
    if is_state and np.isnan(values).any():
        text = ordered["value_text"]
        categories = {v: i for i, v in enumerate(sorted(text.dropna().unique()))}
        coded = text.map(categories).to_numpy(dtype=float)
        values = np.where(np.isnan(values), coded, values)
    series = pd.Series(values, index=pd.DatetimeIndex(ordered["t"]), dtype=float)
    return series[~series.index.duplicated(keep="last")], is_state


def _sample_and_hold(
    series: pd.Series, grid: pd.DatetimeIndex, limit: pd.Timedelta | None
) -> pd.Series:
    observed = series.dropna().rename("value").rename_axis("t").reset_index()
    target = pd.DataFrame({"t": grid})
    merged = pd.merge_asof(
        target,
        observed,
        on="t",
        direction="backward",
        tolerance=limit,
    )
    return pd.Series(merged["value"].to_numpy(dtype=float), index=grid, dtype=float)
