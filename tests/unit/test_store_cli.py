import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

from constellation.cli import app
from constellation.core.settings import get_settings
from constellation.core.store import ParquetStore


@pytest.fixture
def workspace(tmp_path, make_readings, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    store = ParquetStore(tmp_path / "readings")
    for minute in range(3):
        frame = make_readings([(f"2026-03-01T00:0{minute}:00", "a", float(minute))])
        store.write_readings(frame)
    store.write_readings(make_readings([("2026-03-01T00:00:00", "a", 0.0)]))  # a repeat
    yield store
    root.handlers[:] = handlers
    root.setLevel(level)
    get_settings.cache_clear()


def files(store):
    return list(store.root.rglob("part-*.parquet"))


def test_compact_dry_run_reports_without_changing_anything(workspace):
    before = files(workspace)

    result = CliRunner().invoke(app, ["store", "compact", "--dry-run"])

    assert result.exit_code == 0
    assert (
        "would compact iss/ISS/2026-03-01: 4 files, 4 rows -> 1 file, 3 rows"
        in result.stdout
    )
    assert "would compact: 4 files -> 1, 4 rows -> 3 (1 duplicates)" in result.stdout
    assert sorted(files(workspace)) == sorted(before)


def test_compact_rewrites_finished_days(workspace):
    result = CliRunner().invoke(app, ["store", "compact"])

    assert result.exit_code == 0
    assert "compacted: 4 files -> 1, 4 rows -> 3 (1 duplicates)" in result.stdout
    assert len(files(workspace)) == 1
    assert len(workspace.read_readings()) == 3

    again = CliRunner().invoke(app, ["store", "compact"])
    assert "0 files -> 0" in again.stdout


def test_compact_options_and_errors(workspace):
    runner = CliRunner()

    skipped = runner.invoke(app, ["store", "compact", "--before", "2026-03-01"])
    other = runner.invoke(app, ["store", "compact", "--source", "sim"])
    bad = runner.invoke(app, ["store", "compact", "--before", "soon"])

    assert (
        "0 files -> 0" in skipped.stdout
    )  # strictly before: the day itself is not finished
    assert "0 files -> 0" in other.stdout
    assert bad.exit_code == 2
    assert len(files(workspace)) == 4


def test_compact_never_touches_today(tmp_path, make_readings, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    store = ParquetStore(tmp_path / "readings")
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT00:00:00")
    store.write_readings(make_readings([(today, "a", 1.0)]))
    store.write_readings(make_readings([(today, "b", 1.0)]))

    result = CliRunner().invoke(app, ["store", "compact"])

    get_settings.cache_clear()
    assert "0 files -> 0" in result.stdout
    assert len(files(store)) == 2
