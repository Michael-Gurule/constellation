import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

from constellation import __version__
from constellation.cli import app
from constellation.core.settings import get_settings
from constellation.core.store import ParquetStore
from constellation.ingest import isslive

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONSTELLATION_ISS_FLUSH_SECONDS", "5")
    get_settings.cache_clear()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    get_settings.cache_clear()


def _store(tmp_path) -> ParquetStore:
    return ParquetStore(tmp_path / "data" / "readings")


def test_version_prints_the_package_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_status_reports_an_empty_store(tmp_path):
    result = runner.invoke(app, ["collect", "status"])

    assert result.exit_code == 1
    assert "no readings under" in result.stdout


def test_status_summarizes_freshness_and_channels(tmp_path, make_readings):
    now = pd.Timestamp.now(tz="UTC")
    fresh = (now - pd.Timedelta(20, unit="s")).strftime("%Y-%m-%dT%H:%M:%S")
    old = (now - pd.Timedelta(3, unit="h")).strftime("%Y-%m-%dT%H:%M:%S")
    store = _store(tmp_path)
    store.write_readings(make_readings([(fresh, "cmg1_wheel_speed", 6600.0)]))
    store.write_readings(
        make_readings(
            [(old, "cmg1_wheel_speed", 6601.0)], source="sim", asset_id="SIM-1"
        )
    )

    summary = runner.invoke(app, ["collect", "status"])
    detail = runner.invoke(app, ["collect", "status", "--by-channel"])

    assert summary.exit_code == 0
    assert "iss/ISS: 1 rows, 1 channels" in summary.stdout
    assert "[ok]" in summary.stdout.split("sim/SIM-1")[0]
    assert "sim/SIM-1: 1 rows" in summary.stdout
    assert "[STALE]" in summary.stdout.split("sim/SIM-1")[1]
    assert "cmg1_wheel_speed" in detail.stdout


def test_probe_reports_silent_and_static_items(tmp_path, monkeypatch):
    seen = {}

    def fake_probe(items, seconds, out):
        seen.update(items=items, seconds=seconds, out=out)
        return {"Z1000009": 40, "USLAB000012": 1, "TIME_000001": 90}

    monkeypatch.setattr(isslive, "probe", fake_probe)
    out = tmp_path / "probe.jsonl"

    result = runner.invoke(
        app, ["collect", "iss", "--probe", "7", "--probe-out", str(out)]
    )

    assert result.exit_code == 0
    assert seen["seconds"] == 7
    assert seen["out"] == out
    assert seen["items"][-2:] == ["TIME_000001", "TIME_000002"]
    assert "131 updates on 3 items" in result.stdout
    assert "one update only (snapshot): ['USLAB000012']" in result.stdout
    assert "Z1000010" in result.stdout.split("no update at all:")[1].split("\n")[0]


def test_collect_runs_until_the_duration_with_the_configured_flush(
    tmp_path, monkeypatch
):
    calls = {}

    def fake_run(self, duration=None):
        calls.update(
            duration=duration,
            flush=self.flush_seconds,
            root=self.store.root,
            channels=len(self.channels),
        )

    monkeypatch.setattr(isslive.IssLiveCollector, "run", fake_run)

    result = runner.invoke(app, ["collect", "iss", "--duration", "2h"])

    assert result.exit_code == 0
    assert calls == {
        "duration": 7200.0,
        "flush": 5,
        "root": tmp_path / "data" / "readings",
        "channels": 91,
    }


def test_collect_without_a_duration_runs_until_stopped(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        isslive.IssLiveCollector,
        "run",
        lambda self, duration=None: calls.update(d=duration),
    )

    assert runner.invoke(app, ["collect", "iss"]).exit_code == 0
    assert calls == {"d": None}


def test_a_bad_duration_is_rejected(monkeypatch):
    monkeypatch.setattr(
        isslive.IssLiveCollector, "run", lambda self, duration=None: None
    )

    result = runner.invoke(app, ["collect", "iss", "--duration", "soon"])

    assert result.exit_code != 0
