from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from constellation.core.errors import ConfigError
from constellation.core.timeutils import ensure_utc, parse_duration


def test_naive_timestamps_are_taken_as_utc():
    result = ensure_utc(datetime(2026, 3, 1, 12, 0))

    assert result == pd.Timestamp("2026-03-01T12:00:00", tz="UTC")
    assert str(result.tz) == "UTC"


def test_aware_timestamps_are_converted():
    eastern = timezone(timedelta(hours=-5))

    result = ensure_utc(datetime(2026, 3, 1, 12, 0, tzinfo=eastern))

    assert result == pd.Timestamp("2026-03-01T17:00:00", tz="UTC")


def test_strings_and_numpy_values():
    assert ensure_utc("2026-03-01T00:00:00Z") == pd.Timestamp("2026-03-01", tz="UTC")
    assert ensure_utc(np.datetime64("2026-03-01T06:00")) == pd.Timestamp(
        "2026-03-01T06:00", tz="UTC"
    )


def test_series_and_index():
    series = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-02"]))
    index = pd.DatetimeIndex(["2026-03-01 10:00"], tz="Europe/Paris")

    assert str(ensure_utc(series).dtype) == "datetime64[ns, UTC]"
    converted = ensure_utc(index)
    assert converted[0] == pd.Timestamp("2026-03-01 09:00", tz="UTC")
    assert str(converted.tz) == "UTC"
    assert str(ensure_utc(pd.DatetimeIndex(["2026-03-01"])).tz) == "UTC"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("10s", 10),
        ("5m", 300),
        ("1h", 3600),
        ("7d", 604800),
        ("2w", 1209600),
        ("1.5h", 5400),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == pd.Timedelta(seconds, unit="s")


@pytest.mark.parametrize("text", ["", "10", "s", "10x", "-5s", "0s", "1 h h"])
def test_parse_duration_rejects_bad_input(text):
    with pytest.raises(ConfigError):
        parse_duration(text)
