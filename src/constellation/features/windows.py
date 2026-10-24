"""Causal, time-based window functions.

Every function looks only backwards: the value at time `t` depends on samples in
the half-open interval `(t - window, t]`. Windows are durations (`"10min"`,
`"1h"`), never row counts, so results do not change when the sampling cadence
does. All functions accept a Series or a DataFrame with a `DatetimeIndex`.
"""

import numpy as np
import pandas as pd

from constellation.core.errors import ConfigError
from constellation.core.timeutils import parse_duration

_MAD_TO_SIGMA = 1.4826


def _window(window: str | pd.Timedelta) -> pd.Timedelta:
    if isinstance(window, pd.Timedelta):
        return window
    try:
        return parse_duration(window)
    except ConfigError as error:
        raise ConfigError(
            f"invalid window {window!r}: use a number and s, min, h, d or w"
        ) from error


def _check[T: (pd.Series, pd.DataFrame)](x: T) -> None:
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ConfigError("window functions need a DatetimeIndex")
    if not x.index.is_monotonic_increasing:
        raise ConfigError("the index must be sorted in time")


def roll_mean[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 1
) -> T:
    """Mean of the valid samples in the trailing window."""
    _check(x)
    return x.rolling(_window(window), min_periods=min_periods).mean()


def roll_std[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 2
) -> T:
    """Sample standard deviation in the trailing window."""
    _check(x)
    return x.rolling(_window(window), min_periods=min_periods).std()


def roll_min[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 1
) -> T:
    """Minimum in the trailing window."""
    _check(x)
    return x.rolling(_window(window), min_periods=min_periods).min()


def roll_max[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 1
) -> T:
    """Maximum in the trailing window."""
    _check(x)
    return x.rolling(_window(window), min_periods=min_periods).max()


def roll_slope[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 2
) -> T:
    """Least-squares slope (units per second) over the trailing window.

    Computed in O(N) from rolling sums of t, x, t^2 and t*x over the valid
    samples, instead of fitting a line in every window.
    """
    _check(x)
    span = _window(window)
    index = x.index
    seconds = pd.Series((index - index[0]).total_seconds(), index=index)

    valid = x.notna()
    values = x.where(valid, 0.0)
    t = valid.mul(seconds, axis=0)

    def total(frame: T) -> T:
        return frame.rolling(span, min_periods=1).sum()

    n = total(valid.astype(float))
    sum_t = total(t)
    sum_x = total(values)
    sum_tt = total(t.mul(seconds, axis=0))
    sum_tx = total(t * values)

    denominator = n * sum_tt - sum_t * sum_t
    numerator = n * sum_tx - sum_t * sum_x
    # A window needs enough points and a spread of times; otherwise the slope is undefined.
    scale = n * sum_tt
    usable = (
        (n >= min_periods) & (denominator > 1e-12 * scale.abs()) & (denominator > 0)
    )
    return (numerator / denominator).where(usable)


def robust_z[T: (pd.Series, pd.DataFrame)](
    x: T, window: str | pd.Timedelta, min_periods: int = 10
) -> T:
    """Distance from the trailing median in robust standard deviations.

    The scale is the trailing median of the absolute deviations from the
    trailing median, multiplied by 1.4826 so it matches a standard deviation
    for Gaussian data. Where the scale is zero the result is NaN.
    """
    _check(x)
    span = _window(window)
    median = x.rolling(span, min_periods=min_periods).median()
    deviation = (x - median).abs()
    mad = deviation.rolling(span, min_periods=min_periods).median()
    return safe_ratio(x - median, _MAD_TO_SIGMA * mad, eps=1e-12)


def rate_per_s[T: (pd.Series, pd.DataFrame)](x: T) -> T:
    """Change per second since the previous sample."""
    _check(x)
    index = x.index
    seconds = pd.Series(
        index.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9, index=index
    ).diff()
    return x.diff().div(seconds, axis=0)


def safe_ratio[T: (pd.Series, pd.DataFrame)](num: T, den: T, eps: float = 1e-9) -> T:
    """`num / den`, or NaN where `|den| <= eps` or an input is missing. Never inf."""
    small = den.abs() <= eps
    ratio = num / den.where(~small)
    return ratio.replace([np.inf, -np.inf], np.nan)


def roll_changes[T: (pd.Series, pd.DataFrame)](x: T, window: str | pd.Timedelta) -> T:
    """Number of times the value changed within the trailing window."""
    _check(x)
    changed = (x.diff().abs() > 0).astype(float).where(x.notna() & x.shift().notna())
    return changed.rolling(_window(window), min_periods=1).sum()
