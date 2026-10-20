import numpy as np
import pandas as pd
import pytest

from constellation.core.errors import ConfigError
from constellation.features import windows

HOUR = pd.Timedelta(1, unit="h")


def irregular(n=200, seed=1):
    rng = np.random.default_rng(seed)
    offsets = np.cumsum(rng.integers(20, 400, size=n))
    index = pd.Timestamp("2026-03-01", tz="UTC") + pd.to_timedelta(offsets, unit="s")
    return pd.Series(rng.normal(size=n), index=pd.DatetimeIndex(index), name="x")


def trailing(series, window, fn):
    """Reference implementation: apply `fn` to the samples in (t - window, t]."""
    out = []
    for t in series.index:
        chunk = series[(series.index > t - window) & (series.index <= t)].dropna()
        out.append(fn(chunk.to_numpy()))
    return np.array(out, dtype=float)


# -- plain rolling statistics ------------------------------------------------------


def test_statistics_match_the_samples_in_the_previous_window():
    s = irregular()

    np.testing.assert_allclose(windows.roll_mean(s, "1h"), trailing(s, HOUR, np.mean))
    np.testing.assert_allclose(windows.roll_min(s, "1h"), trailing(s, HOUR, np.min))
    np.testing.assert_allclose(windows.roll_max(s, "1h"), trailing(s, HOUR, np.max))
    std = trailing(s, HOUR, lambda v: np.std(v, ddof=1) if len(v) > 1 else np.nan)
    np.testing.assert_allclose(windows.roll_std(s, "1h"), std)


def test_a_window_covers_its_duration_not_a_row_count():
    # One slow period and one fast period: a row-count window would differ.
    slow = pd.date_range("2026-03-01", periods=10, freq="10min", tz="UTC")
    fast = pd.date_range("2026-03-01T02:00", periods=60, freq="1min", tz="UTC")
    s = pd.Series(np.r_[np.zeros(10), np.ones(60)], index=slow.append(fast))

    result = windows.roll_mean(s, "10min")

    assert result.iloc[-1] == 1.0  # the last 10 minutes hold only ones
    assert result.iloc[5] == 0.0


def test_the_window_is_half_open_on_the_left():
    s = pd.Series(
        [100.0, 1.0, 3.0],
        index=pd.DatetimeIndex(
            ["2026-03-01T00:00", "2026-03-01T00:30", "2026-03-01T01:00"], tz="UTC"
        ),
    )

    # At 01:00 the window is (00:00, 01:00]: the sample at exactly 00:00 is out.
    assert windows.roll_mean(s, "1h").iloc[-1] == 2.0


def test_missing_values_are_ignored_and_min_periods_applies():
    index = pd.date_range("2026-03-01", periods=5, freq="1min", tz="UTC")
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan], index=index)

    assert windows.roll_mean(s, "10min").iloc[2] == 2.0
    assert np.isnan(windows.roll_std(s, "10min").iloc[0])  # one sample, std needs two
    assert windows.roll_std(s, "10min", min_periods=1).iloc[0] == 0.0 or np.isnan(
        windows.roll_std(s, "10min", min_periods=1).iloc[0]
    )


def test_dataframes_are_handled_column_by_column():
    s = irregular()
    frame = pd.DataFrame({"a": s, "b": -s})

    result = windows.roll_mean(frame, "30min")

    pd.testing.assert_series_equal(
        result["a"], windows.roll_mean(s, "30min"), check_names=False
    )
    np.testing.assert_allclose(result["b"], -result["a"])


@pytest.mark.parametrize(
    "function", [windows.roll_mean, windows.roll_slope, windows.robust_z]
)
def test_bad_windows_and_indexes_are_rejected(function):
    s = irregular()

    with pytest.raises(ConfigError, match="invalid window"):
        function(s, "soon")
    with pytest.raises(ConfigError, match="DatetimeIndex"):
        function(pd.Series([1.0, 2.0]), "1h")
    with pytest.raises(ConfigError, match="sorted"):
        function(s.iloc[::-1], "1h")


def test_windows_accept_both_duration_styles():
    s = irregular()

    pd.testing.assert_series_equal(
        windows.roll_mean(s, "10m"), windows.roll_mean(s, "10min")
    )
    pd.testing.assert_series_equal(
        windows.roll_mean(s, "1h"), windows.roll_mean(s, HOUR)
    )


# -- slope ---------------------------------------------------------------------------


def test_slope_is_exact_on_linear_data_with_irregular_sampling():
    s = irregular()
    seconds = (s.index - s.index[0]).total_seconds().to_numpy()
    line = pd.Series(3.0 - 0.002 * seconds, index=s.index)

    slope = windows.roll_slope(line, "1h").dropna()

    assert len(slope) > 100
    np.testing.assert_allclose(slope, -0.002, rtol=1e-8)


def test_slope_matches_a_least_squares_fit_on_noisy_data_with_gaps():
    s = irregular(150, seed=4)
    s.iloc[30:40] = np.nan
    expected = []
    for t in s.index:
        chunk = s[(s.index > t - HOUR) & (s.index <= t)].dropna()
        if len(chunk) < 2:
            expected.append(np.nan)
            continue
        x = (chunk.index - chunk.index[0]).total_seconds().to_numpy()
        expected.append(np.polyfit(x, chunk.to_numpy(), 1)[0])

    slope = windows.roll_slope(s, "1h")

    np.testing.assert_allclose(slope, expected, rtol=1e-6, atol=1e-9)


def test_slope_is_accurate_for_a_week_of_data_far_from_the_origin():
    index = pd.date_range("2026-03-01", periods=7 * 8640, freq="10s", tz="UTC")
    seconds = (index - index[0]).total_seconds().to_numpy()
    line = pd.Series(1000.0 + 5e-4 * seconds, index=index)

    slope = windows.roll_slope(line, "10min").dropna()

    np.testing.assert_allclose(slope, 5e-4, rtol=1e-5)


def test_slope_is_undefined_without_enough_points():
    index = pd.DatetimeIndex(["2026-03-01T00:00", "2026-03-01T02:00"], tz="UTC")
    s = pd.Series([1.0, 2.0], index=index)

    assert windows.roll_slope(s, "1h").isna().all()  # one point per window
    assert windows.roll_slope(s, "3h").iloc[-1] == pytest.approx(1.0 / 7200)


def test_slope_of_a_constant_is_zero():
    index = pd.date_range("2026-03-01", periods=50, freq="1min", tz="UTC")

    slope = windows.roll_slope(pd.Series(4.2, index=index), "10min").dropna()

    np.testing.assert_allclose(slope, 0.0, atol=1e-12)


# -- robust z, rate, ratio, changes ---------------------------------------------------


def test_robust_z_flags_an_outlier_and_ignores_the_future():
    rng = np.random.default_rng(2)
    index = pd.date_range("2026-03-01", periods=600, freq="10s", tz="UTC")
    values = rng.normal(size=600)
    values[500] = 12.0
    s = pd.Series(values, index=index)

    z = windows.robust_z(s, "30min")

    assert z.iloc[500] > 6
    assert z.iloc[100:500].abs().max() < 5
    changed_future = s.copy()
    changed_future.iloc[501:] += 50.0
    pd.testing.assert_series_equal(
        windows.robust_z(changed_future, "30min").iloc[:501], z.iloc[:501]
    )


def test_robust_z_of_a_constant_is_nan_not_inf():
    index = pd.date_range("2026-03-01", periods=100, freq="10s", tz="UTC")

    z = windows.robust_z(pd.Series(3.0, index=index), "10min")

    assert not np.isinf(z).any()
    assert z.isna().all()


def test_rate_per_second_uses_the_real_time_step():
    index = pd.DatetimeIndex(
        [
            "2026-03-01T00:00:00",
            "2026-03-01T00:00:10",
            "2026-03-01T00:01:10",
            "2026-03-01T00:01:20",
        ],
        tz="UTC",
    )
    s = pd.Series([0.0, 5.0, 65.0, np.nan], index=index)

    rate = windows.rate_per_s(s)

    assert np.isnan(rate.iloc[0])
    assert rate.iloc[1] == pytest.approx(0.5)
    assert rate.iloc[2] == pytest.approx(1.0)
    assert np.isnan(rate.iloc[3])


def test_safe_ratio_never_returns_infinity():
    num = pd.Series([1.0, 1.0, np.inf, np.nan, 4.0, -3.0])
    den = pd.Series([2.0, 0.0, 1.0, 1.0, np.inf, 1e-12])

    ratio = windows.safe_ratio(num, den, eps=1e-9)

    assert not np.isinf(ratio).any()
    assert ratio.iloc[0] == 0.5
    assert ratio.iloc[[1, 3, 5]].isna().all()  # zero denominator, missing, below eps
    assert ratio.iloc[4] == 0.0  # finite over infinite


def test_safe_ratio_works_on_frames():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"x": [0.0, 4.0]})

    result = windows.safe_ratio(a, b)

    assert np.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 0.5


def test_roll_changes_counts_value_changes_in_the_window():
    index = pd.date_range("2026-03-01", periods=8, freq="10min", tz="UTC")
    s = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, np.nan, 3.0], index=index)

    changes = windows.roll_changes(s, "1h")

    assert changes.iloc[1] == 0.0
    assert changes.iloc[2] == 1.0
    assert changes.iloc[5] == 2.0
    assert changes.iloc[4] == 1.0
