import numpy as np
import pandas as pd
import pytest

from constellation.core.errors import SchemaError
from constellation.features import windows
from constellation.features.frames import build_frames


def readings(rows, asset="ISS", source="iss"):
    """rows: (iso time, channel, value[, text])"""
    frame = pd.DataFrame(
        {
            "source": source,
            "asset_id": asset,
            "channel": [r[1] for r in rows],
            "t": pd.to_datetime(
                [r[0] for r in rows], utc=True, format="ISO8601"
            ).as_unit("ns"),
            "value": [r[2] for r in rows],
            "value_text": pd.Series(
                [r[3] if len(r) > 3 else None for r in rows], dtype=object
            ),
            "status": pd.Series(["24"] * len(rows), dtype=object),
            "ingested_at": pd.Timestamp("2026-03-05", tz="UTC").as_unit("ns"),
        }
    )
    frame["value"] = frame["value"].astype(float)
    return frame


def test_grid_is_regular_utc_and_covers_the_data():
    r = readings([("2026-03-01T00:00:03", "a", 1.0), ("2026-03-01T00:00:47", "a", 2.0)])

    frame = build_frames(r, "10s")["ISS"]

    assert frame.index.name == "t"
    assert str(frame.index.dtype) == "datetime64[ns, UTC]"
    assert frame.index[0] == pd.Timestamp("2026-03-01T00:00:00", tz="UTC")
    assert frame.index[-1] == pd.Timestamp("2026-03-01T00:00:50", tz="UTC")
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta(10, unit="s")).all()


def test_values_are_held_for_at_most_max_hold():
    r = readings([("2026-03-01T00:00:00", "a", 1.0), ("2026-03-01T00:03:00", "a", 5.0)])

    frame = build_frames(r, "10s", "60s")["ISS"]
    a = frame["a"]

    assert a.loc["2026-03-01T00:00:00"] == 1.0
    assert a.loc["2026-03-01T00:01:00"] == 1.0  # exactly max_hold old: kept
    assert np.isnan(a.loc["2026-03-01T00:01:10"])  # older than max_hold
    assert np.isnan(a.loc["2026-03-01T00:02:50"])
    assert a.loc["2026-03-01T00:03:00"] == 5.0


def test_hold_is_causal_and_never_uses_a_later_sample():
    r = readings([("2026-03-01T00:00:00", "a", 1.0), ("2026-03-01T00:00:15", "a", 2.0)])

    a = build_frames(r, "10s", "60s")["ISS"]["a"]

    assert (
        a.loc["2026-03-01T00:00:10"] == 1.0
    )  # the 00:00:15 sample is still in the future
    assert a.loc["2026-03-01T00:00:20"] == 2.0


def test_hold_can_be_overridden_per_channel_and_can_be_unlimited():
    r = readings(
        [
            ("2026-03-01T00:00:00", "fast", 1.0),
            ("2026-03-01T00:00:00", "slow", 1.0),
            ("2026-03-01T00:00:00", "mode", 4.0),
            ("2026-03-01T01:00:00", "fast", 2.0),
        ]
    )

    frame = build_frames(r, "10m", "1m", hold={"slow": "30m", "mode": None})["ISS"]

    assert np.isnan(frame["fast"].loc["2026-03-01T00:20:00"])
    assert frame["slow"].loc["2026-03-01T00:20:00"] == 1.0
    assert np.isnan(frame["slow"].loc["2026-03-01T00:40:00"])
    assert frame["mode"].loc["2026-03-01T01:00:00"] == 4.0  # held for the whole hour


def test_valid_fraction_is_the_share_of_channels_with_a_value():
    r = readings(
        [
            ("2026-03-01T00:00:00", "a", 1.0),
            ("2026-03-01T00:00:00", "b", 1.0),
            ("2026-03-01T00:02:00", "a", 2.0),
        ]
    )

    frame = build_frames(r, "1m", "90s")["ISS"]

    assert frame["valid_fraction"].loc["2026-03-01T00:00:00"] == 1.0
    assert frame["valid_fraction"].loc["2026-03-01T00:02:00"] == 0.5
    assert list(frame.columns) == ["a", "b", "valid_fraction"]


def test_duplicate_timestamps_keep_the_last_reading():
    r = readings([("2026-03-01T00:00:00", "a", 1.0), ("2026-03-01T00:00:00", "a", 9.0)])

    assert build_frames(r)["ISS"]["a"].iloc[0] == 9.0


def test_state_channels_become_integer_codes():
    r = readings(
        [
            ("2026-03-01T00:00:00", "mode", 5.0, "5"),  # numeric code from the source
            ("2026-03-01T00:00:30", "mode", 2.0, "2"),
            ("2026-03-01T00:00:00", "on", float("nan"), "on"),  # text only
            ("2026-03-01T00:00:30", "on", float("nan"), "off"),
            ("2026-03-01T00:01:00", "on", float("nan"), "on"),
            ("2026-03-01T00:00:00", "temp", 20.0),
        ]
    )

    frame = build_frames(r, "30s", "60s")["ISS"]

    assert frame.attrs["state_channels"] == ["mode", "on"]
    assert frame["mode"].tolist() == [5.0, 2.0, 2.0]
    assert frame["on"].tolist() == [1.0, 0.0, 1.0]  # sorted texts: off=0, on=1
    assert frame["temp"].iloc[0] == 20.0


def test_each_asset_gets_its_own_frame_and_grid():
    r = pd.concat(
        [
            readings([("2026-03-01T00:00:00", "a", 1.0)], asset="SIM-1", source="sim"),
            readings([("2026-03-02T00:00:00", "b", 2.0)], asset="SIM-2", source="sim"),
        ],
        ignore_index=True,
    )

    frames = build_frames(r, "1h")

    assert set(frames) == {"SIM-1", "SIM-2"}
    assert list(frames["SIM-1"].columns) == ["a", "valid_fraction"]
    assert frames["SIM-2"].index[0] == pd.Timestamp("2026-03-02", tz="UTC")


def test_empty_readings_give_no_frames():
    from constellation.core.schemas import empty_readings

    assert build_frames(empty_readings()) == {}


def test_invalid_readings_are_rejected():
    r = readings([("2026-03-01", "a", 1.0)]).drop(columns="status")

    with pytest.raises(SchemaError):
        build_frames(r)


def test_changing_data_after_a_time_never_changes_earlier_rows():
    rng = np.random.default_rng(3)
    times = pd.date_range("2026-03-01", periods=200, freq="7s", tz="UTC")
    base = readings(
        [
            (t.isoformat(), "a", float(v))
            for t, v in zip(times, rng.normal(size=200), strict=True)
        ]
    )
    cut = times[120]
    altered = base.copy()
    altered.loc[altered["t"] > cut, "value"] = 999.0

    before = build_frames(base, "10s", "30s")["ISS"]
    after = build_frames(altered, "10s", "30s")["ISS"]

    mask = before.index <= cut
    pd.testing.assert_frame_equal(before[mask], after[mask])


def test_hourly_means_agree_across_cadences():
    seconds = np.arange(0, 8 * 3600)
    t = pd.Timestamp("2026-03-01", tz="UTC") + pd.to_timedelta(seconds, unit="s")
    value = 10.0 + 3.0 * np.sin(
        2 * np.pi * seconds / 5400.0
    )  # an orbital-period signal
    r = readings(
        [(ts.isoformat(), "a", v) for ts, v in zip(t[::2], value[::2], strict=True)]
    )

    five = windows.roll_mean(build_frames(r, "5s", "60s")["ISS"]["a"], "1h")
    ten = windows.roll_mean(build_frames(r, "10s", "60s")["ISS"]["a"], "1h")

    common = five.index.intersection(ten.index)
    common = common[common >= common[0] + pd.Timedelta(1, unit="h")]
    assert len(common) > 1000
    assert np.abs(five.loc[common] - ten.loc[common]).max() < 0.01  # amplitude is 3
