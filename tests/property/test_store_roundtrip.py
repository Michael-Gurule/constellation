"""Any valid readings frame survives a write and a read exactly."""

import tempfile

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from constellation.core.schemas import READINGS, SOURCES, validate_readings
from constellation.core.store import ParquetStore

NAMES = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,11}", fullmatch=True)
# 2000-01-01 to 2100-01-01 in nanoseconds since the epoch.
EPOCH_NS = st.integers(min_value=946_684_800 * 10**9, max_value=4_102_444_800 * 10**9)
TEXT = st.none() | st.text(max_size=12)
VALUES = st.floats(allow_nan=True, allow_infinity=True, width=64)


@st.composite
def readings_frames(draw: st.DrawFn) -> pd.DataFrame:
    keys = draw(
        st.lists(
            st.tuples(st.sampled_from(SOURCES), NAMES, NAMES, EPOCH_NS),
            min_size=1,
            max_size=25,
            unique=True,
        )
    )
    rows = [
        {
            "source": source,
            "asset_id": asset,
            "channel": channel,
            "t": pd.Timestamp(ns, unit="ns", tz="UTC"),
            "value": draw(VALUES),
            "value_text": draw(TEXT),
            "status": draw(TEXT),
            "ingested_at": pd.Timestamp(draw(EPOCH_NS), unit="ns", tz="UTC"),
        }
        for source, asset, channel, ns in keys
    ]
    frame = pd.DataFrame(rows, columns=[c.name for c in READINGS])
    for column in ("t", "ingested_at"):
        frame[column] = frame[column].astype("datetime64[ns, UTC]")
    for column in ("value_text", "status"):
        frame[column] = frame[column].astype(object)
    return frame


@given(readings_frames())
def test_valid_frames_round_trip_exactly(frame: pd.DataFrame) -> None:
    validate_readings(frame)
    keys = ["source", "asset_id", "channel", "t"]
    expected = frame.sort_values(keys).reset_index(drop=True)

    with tempfile.TemporaryDirectory() as root:
        store = ParquetStore(root)
        store.write_readings(frame)
        result = store.read_readings()

    pd.testing.assert_frame_equal(result, expected, check_exact=True)


@given(readings_frames())
def test_partitions_match_the_utc_event_date(frame: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as root:
        written = ParquetStore(root).write_readings(frame)
        directories = {p.parent.name for p in written}

    expected = set("date=" + frame["t"].dt.strftime("%Y-%m-%d"))
    assert directories == expected
