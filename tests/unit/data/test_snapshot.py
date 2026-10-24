import json
from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

from constellation.cli import app
from constellation.core.errors import ConfigError
from constellation.core.manifest import file_sha256
from constellation.core.settings import get_settings
from constellation.core.store import ParquetStore
from constellation.data.snapshot import (
    build_snapshot,
    find_gaps,
    verify_snapshot,
    write_snapshot,
)


def live(frame):
    """Mark every reading as arriving one second after its event time."""
    frame = frame.copy()
    frame["ingested_at"] = frame["t"] + pd.Timedelta(1, unit="s")
    return frame


def minutes(day, start, stop, channel="a", step=1):
    return [
        (f"{day}T00:{m:02d}:00", channel, float(m)) for m in range(start, stop, step)
    ]


@pytest.fixture
def store(tmp_path, make_readings):
    store = ParquetStore(tmp_path / "readings")
    # Day 1: two files, with a ten-minute hole in the feed (00:20 to 00:30) and a repeat.
    store.write_readings(live(make_readings(minutes("2026-03-01", 0, 21))))
    store.write_readings(
        live(make_readings(minutes("2026-03-01", 30, 40) + minutes("2026-03-01", 0, 3)))
    )
    store.write_readings(live(make_readings(minutes("2026-03-01", 0, 5, channel="b"))))
    store.write_readings(live(make_readings(minutes("2026-03-02", 0, 10))))
    store.write_readings(
        live(make_readings(minutes("2026-03-04", 0, 10)))
    )  # outside the range below
    return store


def test_snapshot_describes_partitions_files_channels_and_totals(store):
    snapshot = build_snapshot(
        store,
        start="2026-03-01",
        end="2026-03-02",
        name="demo",
        today=date(2026, 3, 10),
    )

    assert snapshot["name"] == "demo"
    assert [p["date"] for p in snapshot["partitions"]] == ["2026-03-01", "2026-03-02"]
    first = snapshot["partitions"][0]
    assert len(first["files"]) == 3
    path = (
        store.root
        / "source=iss"
        / "asset=ISS"
        / "date=2026-03-01"
        / first["files"][0]["name"]
    )
    assert first["files"][0]["sha256"] == file_sha256(path)
    assert first["files"][0]["rows"] == len(pd.read_parquet(path))
    assert first["rows_raw"] == 21 + 13 + 5
    assert first["rows_unique"] == 21 + 10 + 5  # three repeated readings dropped
    assert snapshot["channels"]["a"]["first_t"] == "2026-03-01T00:00:00+00:00"
    assert snapshot["channels"]["a"]["last_t"] == "2026-03-02T00:09:00+00:00"
    assert snapshot["channels"]["b"]["rows"] == 5
    totals = snapshot["totals"]
    assert (totals["partitions"], totals["files"], totals["channels"]) == (2, 4, 2)
    assert totals["rows_unique"] == 36 + 10
    assert totals["first_t"] == "2026-03-01T00:00:00+00:00"


def test_snapshot_flags_incomplete_days_and_uncompacted_partitions(store):
    open_range = build_snapshot(
        store, start="2026-03-01", end="2026-03-02", name="x", today=date(2026, 3, 2)
    )
    finished = build_snapshot(
        store, start="2026-03-02", end="2026-03-02", name="x", today=date(2026, 3, 3)
    )

    assert not open_range["complete"]
    assert not open_range["compacted"]  # 2026-03-01 has three files
    assert finished["complete"]
    assert finished["compacted"]


def test_gaps_longer_than_five_minutes_are_listed(store):
    snapshot = build_snapshot(
        store, start="2026-03-01", end="2026-03-01", name="x", today=date(2026, 3, 10)
    )

    assert snapshot["gaps"] == [
        {
            "start": "2026-03-01T00:20:00+00:00",
            "end": "2026-03-01T00:30:00+00:00",
            "seconds": 600.0,
        }
    ]
    assert snapshot["gap_threshold_s"] == 300.0


def test_readings_that_arrive_late_do_not_hide_a_gap(make_readings):
    frame = live(
        make_readings(minutes("2026-03-01", 0, 5) + minutes("2026-03-01", 30, 33))
    )
    resent = make_readings([("2026-03-01T00:15:00", "slow_state", 1.0)])
    resent["ingested_at"] = pd.Timestamp("2026-03-01T01:00:00", tz="UTC").as_unit(
        "ns"
    )  # a re-sent snapshot

    gaps = find_gaps(pd.concat([frame, resent], ignore_index=True))

    assert [g["seconds"] for g in gaps] == [
        1560.0
    ]  # 00:04 to 00:30, despite the reading at 00:15


def test_a_short_series_has_no_gaps(make_readings):
    assert find_gaps(live(make_readings([("2026-03-01T00:00:00", "a", 1.0)]))) == []


def test_verify_detects_changed_missing_and_new_files(store, make_readings):
    snapshot = build_snapshot(
        store, start="2026-03-01", end="2026-03-02", name="x", today=date(2026, 3, 10)
    )
    assert verify_snapshot(store, snapshot) == []

    day = store.root / "source=iss" / "asset=ISS" / "date=2026-03-01"
    victim, removed = sorted(day.glob("part-*.parquet"))[:2]
    victim.write_bytes(victim.read_bytes() + b"x")
    removed.unlink()
    store.write_readings(live(make_readings(minutes("2026-03-02", 20, 22))))

    problems = verify_snapshot(store, snapshot)

    assert sorted(p.split(":")[0] for p in problems) == [
        "changed",
        "missing",
        "new file",
    ]


def test_write_snapshot_names_the_file_after_source_and_name(tmp_path):
    path = write_snapshot(tmp_path / "manifests", "oct", {"source": "iss", "x": 1})

    assert path.name == "iss_oct.json"
    assert json.loads(path.read_text()) == {"source": "iss", "x": 1}
    assert [p.name for p in path.parent.iterdir()] == ["iss_oct.json"]


def test_invalid_ranges_are_rejected(store):
    with pytest.raises(ConfigError, match="end must not be before start"):
        build_snapshot(store, start="2026-03-02", end="2026-03-01", name="x")
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        build_snapshot(store, start="March", end="2026-03-01", name="x")


def test_snapshot_of_an_empty_range_is_valid(store):
    snapshot = build_snapshot(
        store, start="2025-01-01", end="2025-01-02", name="x", today=date(2026, 3, 10)
    )

    assert snapshot["partitions"] == []
    assert snapshot["totals"]["rows_unique"] == 0
    assert snapshot["totals"]["first_t"] is None
    assert snapshot["compacted"]


# -- command line -----------------------------------------------------------------------


@pytest.fixture
def cli_store(tmp_path, make_readings, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    store = ParquetStore(tmp_path / "data" / "readings")
    store.write_readings(live(make_readings(minutes("2026-03-01", 0, 10))))
    store.write_readings(live(make_readings(minutes("2026-03-01", 20, 30))))
    yield tmp_path
    get_settings.cache_clear()


def test_cli_snapshot_writes_a_manifest_and_verify_checks_it(cli_store):
    runner = CliRunner()
    args = [
        "data",
        "snapshot",
        "iss",
        "--start",
        "2026-03-01",
        "--end",
        "2026-03-01",
        "--name",
        "demo",
    ]

    written = runner.invoke(app, args)
    verified = runner.invoke(app, [*args, "--verify"])

    manifest = cli_store / "data" / "manifests" / "iss_demo.json"
    assert written.exit_code == 0
    assert manifest.exists()
    assert (
        "1 partitions, 2 files, 20 unique readings on 1 channels, 1 gaps over 300 s"
        in written.stdout
    )
    assert "several files" in written.stdout  # not compacted yet
    assert verified.exit_code == 0
    assert "all files match" in verified.stdout

    part = next((cli_store / "data" / "readings").rglob("part-*.parquet"))
    part.write_bytes(part.read_bytes() + b"x")
    modified = runner.invoke(app, [*args, "--verify"])
    assert modified.exit_code == 1
    assert "changed:" in modified.stdout


def test_cli_snapshot_default_name_and_errors(cli_store):
    runner = CliRunner()

    runner.invoke(
        app, ["data", "snapshot", "iss", "--start", "2026-03-01", "--end", "2026-03-02"]
    )
    other = runner.invoke(
        app,
        ["data", "snapshot", "satnogs", "--start", "2026-03-01", "--end", "2026-03-02"],
    )
    missing = runner.invoke(
        app,
        [
            "data",
            "snapshot",
            "iss",
            "--start",
            "2026-03-01",
            "--end",
            "2026-03-02",
            "--name",
            "nope",
            "--verify",
        ],
    )

    assert (
        cli_store / "data" / "manifests" / "iss_2026-03-01_2026-03-02.json"
    ).exists()
    assert other.exit_code == 2
    assert missing.exit_code == 2
