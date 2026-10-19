import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from constellation.core.schemas import validate_readings
from constellation.core.store import ParquetStore
from constellation.ingest import isslive
from constellation.ingest.channels import load_channels
from constellation.ingest.isslive import (
    IssLiveCollector,
    Observation,
    parse_source_time,
    resolve_source_time,
)

PROBE = Path(__file__).resolve().parents[2] / "fixtures" / "isslive_probe.jsonl"
CHANNELS = load_channels("configs/ingest/iss.yaml").channels
NOW = pd.Timestamp("2026-10-19T23:40:00", tz="UTC")


class FakeUpdate:
    """The slice of the SDK's ItemUpdate that the collector reads."""

    def __init__(self, item, value, hours=7031.5, status="24", snapshot=False):
        self._item = item
        self._fields = {"Value": value, "TimeStamp": hours, "Status.Class": status}
        self._snapshot = snapshot

    def getItemName(self):
        return self._item

    def getValue(self, field):
        value = self._fields.get(field)
        return None if value is None else str(value)

    def isSnapshot(self):
        return self._snapshot


@pytest.fixture
def store(tmp_path):
    return ParquetStore(tmp_path / "readings")


@pytest.fixture
def collector(store):
    return IssLiveCollector(CHANNELS, store, flush_seconds=60, clock=lambda: NOW)


# -- time conversion ---------------------------------------------------------------


def test_hour_24_is_the_start_of_january_first():
    assert parse_source_time(24.0, 2026) == pd.Timestamp("2026-01-01", tz="UTC")
    assert parse_source_time(0.0, 2026) == pd.Timestamp("2025-12-31", tz="UTC")


def test_source_time_matches_a_recorded_update():
    # Recorded 2026-10-19 23:3x UTC with TimeStamp 7031.5597...
    t = parse_source_time(7031.559722222222, 2026)

    assert t == pd.Timestamp("2026-10-19T23:33:35", tz="UTC")
    assert t.tz is not None


def test_source_time_is_rounded_to_microseconds():
    t = parse_source_time(7031.559722222222, 2026)

    assert t.value % 1000 == 0


def test_unknown_year_is_inferred_from_the_clock():
    assert resolve_source_time(7031.5, None, NOW).year == 2026


def test_year_rollover_picks_the_year_nearest_to_now():
    # Just after midnight on 1 January the year item may still say the old year.
    new_year = pd.Timestamp("2027-01-01T00:00:30", tz="UTC")

    stale = resolve_source_time(24.0 + 30 / 3600, 2026, new_year)

    assert stale == pd.Timestamp("2027-01-01T00:00:30", tz="UTC")
    assert (
        resolve_source_time(24.0, 2026, pd.Timestamp("2026-01-01T00:01", tz="UTC")).year
        == 2026
    )


# -- update handling ---------------------------------------------------------------


def test_update_is_attributed_to_the_item_it_names(collector, store):
    outcome = collector.handle_update(FakeUpdate("Z1000010", 6600.5, hours=7031.0))

    assert outcome.outcome == "stored"
    assert collector.flush() == 1
    (row,) = store.read_readings().itertuples()
    assert row.channel == "cmg2_wheel_speed"
    assert row.value == 6600.5
    assert row.source == "iss"
    assert row.asset_id == "ISS"
    assert row.t == parse_source_time(7031.0, 2026)  # source time, not the clock
    assert row.ingested_at == NOW
    assert row.status == "24"
    assert row.value_text is None


def test_state_channels_keep_the_raw_text(collector, store):
    collector.handle_update(FakeUpdate("USLAB000012", "5"))
    collector.handle_update(FakeUpdate("S1000009", "on"))
    collector.flush()

    rows = store.read_readings().set_index("channel")
    assert rows.loc["gnc_mode", "value"] == 5.0
    assert rows.loc["gnc_mode", "value_text"] == "5"
    assert pd.isna(rows.loc["sband_rfg1_on", "value"])
    assert rows.loc["sband_rfg1_on", "value_text"] == "on"


def test_unparsable_numeric_value_is_kept_as_text(collector, store):
    collector.handle_update(FakeUpdate("Z1000009", "n/a"))
    collector.flush()

    (row,) = store.read_readings().itertuples()
    assert pd.isna(row.value)
    assert row.value_text == "n/a"


def test_time_items_set_the_year_and_are_not_stored(collector, store):
    collector.handle_update(FakeUpdate("TIME_000002", "2027"))
    collector.handle_update(FakeUpdate("TIME_000001", "25313615000"))
    # With the year known to be 2027, hour 7031 is in October 2027, which is
    # more than half a year from the clock, so the nearest year (2026) wins.
    collector.handle_update(FakeUpdate("Z1000009", 1.0, hours=7031.0))

    assert collector.counts["time"] == 2
    assert collector._year == 2027
    assert collector.flush() == 1
    assert store.read_readings()["t"].iloc[0].year == 2026


def test_garbage_year_is_ignored(collector):
    collector.handle_update(FakeUpdate("TIME_000002", "not a year"))
    collector.handle_update(FakeUpdate("TIME_000002", "12"))

    assert collector._year is None


@pytest.mark.parametrize(
    ("update", "outcome"),
    [
        (FakeUpdate("NOPE", 1), "unmapped"),
        (FakeUpdate("Z1000009", 1, hours="soon"), "invalid"),
        (FakeUpdate("Z1000009", 1, hours="nan"), "invalid"),
        (FakeUpdate("Z1000009", None), "invalid"),
        (FakeUpdate("P4000002", 0, hours=0, status="2"), "nodata"),
    ],
)
def test_updates_that_cannot_become_readings_are_counted(collector, update, outcome):
    assert collector.handle_update(update).outcome == outcome
    assert collector.counts[outcome] == 1
    assert collector.flush() == 0


def test_repeated_snapshots_are_not_stored_twice(collector):
    first = collector.handle_update(FakeUpdate("USLAB000012", 5, snapshot=True))
    again = collector.handle_update(FakeUpdate("USLAB000012", 5, snapshot=True))
    changed = collector.handle_update(FakeUpdate("USLAB000012", 6, hours=7032.0))

    assert (first.outcome, again.outcome, changed.outcome) == (
        "stored",
        "duplicate",
        "stored",
    )


def test_observer_sees_every_update(store):
    seen: list[Observation] = []
    collector = IssLiveCollector(
        CHANNELS, store, clock=lambda: NOW, observer=seen.append
    )

    collector.handle_update(FakeUpdate("Z1000009", 1, snapshot=True))
    collector.handle_update(FakeUpdate("NOPE", 1))

    assert [(o.item, o.outcome, o.snapshot) for o in seen] == [
        ("Z1000009", "stored", True),
        ("NOPE", "unmapped", False),
    ]
    assert seen[0].t is not None


def test_unmapped_items_are_logged_only_a_few_times(collector, caplog):
    with caplog.at_level(logging.WARNING, logger=isslive.__name__):
        for _ in range(20):
            collector.handle_update(FakeUpdate("NOPE", 1))

    assert len(caplog.records) == 5


def test_items_to_subscribe_include_the_time_items(collector):
    assert "Z1000009" in collector.items
    assert collector.items[-2:] == ["TIME_000001", "TIME_000002"]
    assert len(collector.items) == len(CHANNELS) + 2


# -- the recorded probe ------------------------------------------------------------


def _probe_records():
    return [json.loads(line) for line in PROBE.read_text().splitlines()]


def test_recorded_probe_maps_every_update(store):
    records = _probe_records()
    clock = pd.Timestamp(records[-1]["wall"])
    collector = IssLiveCollector(CHANNELS, store, clock=lambda: clock)

    for record in records:
        update = FakeUpdate(
            record["item"],
            record["Value"],
            hours=record["TimeStamp"],
            status=record["Status.Class"],
            snapshot=record["snapshot"],
        )
        collector.handle_update(update)
    collector.flush()

    assert collector.counts["unmapped"] == 0
    assert collector.counts["invalid"] == 0
    assert collector.counts["nodata"] == 6  # the six solar array current channels
    assert collector.counts["stored"] > 800

    readings = validate_readings(store.read_readings())
    assert readings["channel"].nunique() == 85  # 91 configured minus 6 with no data
    assert readings["t"].max() <= clock
    live = readings[readings["status"] == "24"]
    assert (clock - live["t"].max()) < pd.Timedelta(30, unit="s")
    speed = readings[readings["channel"] == "cmg1_wheel_speed"]["value"]
    assert speed.between(6500, 6700).all()
    assert set(readings["status"]) == {"24"}


# -- buffering and flushing ----------------------------------------------------------


def test_buffer_flushes_when_full(store):
    collector = IssLiveCollector(CHANNELS, store, max_buffer=3, clock=lambda: NOW)

    for i in range(3):
        collector.handle_update(FakeUpdate("Z1000009", i, hours=7031.0 + i))

    assert collector.counts["flushed"] == 3
    assert len(store.read_readings()) == 3


def test_failed_flush_keeps_readings_for_the_next_attempt(
    collector, store, monkeypatch, caplog
):
    collector.handle_update(FakeUpdate("Z1000009", 1.0))
    real_write = store.write_readings
    monkeypatch.setattr(
        store, "write_readings", lambda df: (_ for _ in ()).throw(OSError("full"))
    )

    with caplog.at_level(logging.ERROR, logger=isslive.__name__):
        assert collector.flush() == 0
    assert "flush failed" in caplog.text

    monkeypatch.setattr(store, "write_readings", real_write)
    assert collector.flush() == 1
    assert len(store.read_readings()) == 1


def test_flush_with_nothing_buffered_is_a_no_op(collector):
    assert collector.flush() == 0


def test_background_flush_writes_periodically(store):
    collector = IssLiveCollector(CHANNELS, store, flush_seconds=0.02, clock=lambda: NOW)
    collector.handle_update(FakeUpdate("Z1000009", 1.0))
    thread = threading.Thread(target=collector._flush_loop)
    thread.start()
    time.sleep(0.3)
    collector._stop.set()
    thread.join(timeout=2)

    assert len(store.read_readings()) == 1


# -- gaps ------------------------------------------------------------------------------


def test_long_silence_is_logged_as_an_ingest_gap(collector, monkeypatch, caplog):
    collector._last_update = time.monotonic() - 400

    with caplog.at_level(logging.WARNING, logger=isslive.__name__):
        collector._check_gap()
        collector._check_gap()  # logged once per gap
        collector.handle_update(FakeUpdate("Z1000009", 1.0))

    messages = [r.getMessage() for r in caplog.records]
    assert (
        sum("ingest_gap: no updates for" in m and "resumed" not in m for m in messages)
        == 1
    )
    assert any("resumed" in m for m in messages)


def test_short_silence_is_not_a_gap(collector, caplog):
    with caplog.at_level(logging.WARNING, logger=isslive.__name__):
        collector._check_gap()
        collector.handle_update(FakeUpdate("Z1000009", 1.0))

    assert caplog.records == []


# -- running -------------------------------------------------------------------------


class FakeClient:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def fake_session(monkeypatch):
    session = {}

    def open_session(items, on_update, on_status):
        session.update(
            items=items, on_update=on_update, on_status=on_status, client=FakeClient()
        )
        on_status("CONNECTED:WS-STREAMING")
        return session["client"]

    monkeypatch.setattr(isslive, "open_session", open_session)
    return session


def test_run_stops_after_the_duration_and_flushes(collector, store, fake_session):
    # The SDK calls `isSnapshot` on the update; the collector's own callback reads it.
    collector.handle_update(FakeUpdate("Z1000009", 1.0))

    collector.run(duration=0.05)

    assert fake_session["client"].disconnected
    assert fake_session["items"] == collector.items
    assert len(store.read_readings()) == 1


def test_updates_arriving_through_the_session_callback_are_stored(
    collector, store, fake_session
):
    def deliver():
        while "on_update" not in fake_session:
            time.sleep(0.005)
        fake_session["on_update"](FakeUpdate("Z1000011", 6601, snapshot=True))

    threading.Thread(target=deliver).start()
    collector.run(duration=0.2)

    assert store.read_readings()["channel"].tolist() == ["cmg3_wheel_speed"]


def test_sigint_stops_the_collector_cleanly(collector, store, fake_session):
    collector.handle_update(FakeUpdate("Z1000009", 1.0))
    before = signal.getsignal(signal.SIGINT)
    timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()

    collector.run(duration=None)  # returns once the signal arrives

    timer.join()
    assert fake_session["client"].disconnected
    assert len(store.read_readings()) == 1
    assert signal.getsignal(signal.SIGINT) is before


# -- probe -----------------------------------------------------------------------------


def test_probe_trims_each_item_and_reports_full_counts(tmp_path, monkeypatch):
    def open_session(items, on_update, on_status):
        for i in range(30):
            on_update(FakeUpdate("Z1000009", i, hours=7031.0 + i, snapshot=i == 0))
        on_update(FakeUpdate("USLAB000012", 5, snapshot=True))
        return FakeClient()

    monkeypatch.setattr(isslive, "open_session", open_session)
    out = tmp_path / "probe.jsonl"

    counts = isslive.probe(["Z1000009", "USLAB000012"], 0.01, out, max_per_item=10)

    assert counts == {"Z1000009": 30, "USLAB000012": 1}
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert sum(r["item"] == "Z1000009" for r in lines) == 10
    assert lines[0]["snapshot"] is True
    assert {"TimeStamp", "Value", "Status.Class", "Status.Indicator", "wall"} <= set(
        lines[0]
    )
