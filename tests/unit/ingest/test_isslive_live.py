"""Soak test against NASA's public ISSLive feed. Run with `pytest --run-live`."""

import pandas as pd
import pytest

from constellation.core.store import ParquetStore
from constellation.ingest.channels import load_channels
from constellation.ingest.isslive import IssLiveCollector

SOAK_SECONDS = 120


@pytest.mark.live
@pytest.mark.slow
def test_two_minute_soak_maps_updates_and_keeps_event_time_near_now(tmp_path):
    channels = load_channels("configs/ingest/iss.yaml").channels
    observed = []
    store = ParquetStore(tmp_path / "readings")
    collector = IssLiveCollector(
        channels, store, flush_seconds=30, observer=observed.append
    )

    collector.run(duration=SOAK_SECONDS)

    updates = [o for o in observed if o.outcome != "time"]
    mapped = [o for o in updates if o.outcome in {"stored", "duplicate", "nodata"}]
    assert len(updates) > 500, "the feed delivered almost nothing"
    assert len(mapped) / len(updates) >= 0.95

    now = pd.Timestamp.now(tz="UTC")
    live_times = [o.t for o in observed if o.outcome == "stored" and not o.snapshot]
    assert live_times
    assert all(abs(now - t) <= pd.Timedelta(5, unit="min") for t in live_times)

    stored = store.read_readings()
    assert not stored.empty
    assert stored["channel"].nunique() >= 60
    assert len(stored) == collector.counts["stored"]
