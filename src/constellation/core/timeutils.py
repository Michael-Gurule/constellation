"""Timezone-aware UTC helpers. Every timestamp in the package is UTC."""

import re
from datetime import datetime
from typing import overload

import numpy as np
import pandas as pd

from constellation.core.errors import ConfigError

_DURATION = re.compile(r"^(?P<number>\d+(?:\.\d+)?)(?P<unit>s|m|h|d|w)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


@overload
def ensure_utc(value: pd.Series) -> pd.Series: ...
@overload
def ensure_utc(value: pd.DatetimeIndex) -> pd.DatetimeIndex: ...
@overload
def ensure_utc(
    value: str | datetime | np.datetime64 | pd.Timestamp,
) -> pd.Timestamp: ...


def ensure_utc(
    value: pd.Series | pd.DatetimeIndex | str | datetime | np.datetime64 | pd.Timestamp,
) -> pd.Series | pd.DatetimeIndex | pd.Timestamp:
    """Return `value` as a UTC timestamp.

    Timezone-aware values are converted to UTC. Naive values are taken to be in
    UTC already, which is how every source in this project reports time.
    """
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True)
    if isinstance(value, pd.DatetimeIndex):
        return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def parse_duration(text: str) -> pd.Timedelta:
    """Parse a duration such as `10s`, `5m`, `1h`, `7d` or `2w`."""
    match = _DURATION.match(text.strip())
    if match is None:
        raise ConfigError(
            f"invalid duration {text!r}: expected a number and one of s, m, h, d, w"
        )
    seconds = float(match["number"]) * _UNIT_SECONDS[match["unit"]]
    if seconds <= 0:
        raise ConfigError(f"duration must be positive, got {text!r}")
    return pd.Timedelta(seconds, unit="s")
