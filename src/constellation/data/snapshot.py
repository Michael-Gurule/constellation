"""Dated, checksummed snapshots of a slice of the readings store.

A snapshot is a manifest of the files that make up a date range: per partition
the file names, sha256 hashes and row counts, per channel the first and last
event time, and the gaps in the feed. An experiment records which snapshot it
used, and `verify_snapshot` shows whether the files have changed since.
"""

import json
import os
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from constellation.core.errors import ConfigError
from constellation.core.manifest import file_sha256
from constellation.core.store import ParquetStore, drop_duplicate_readings

GAP_SECONDS = 300.0
# A reading counts as live (part of the feed's continuity) when it arrived soon
# after its event time. Re-sent snapshots after a reconnect carry old event times.
LIVE_LAG = pd.Timedelta(10, unit="min")


def _day(value: date | str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(value)
    except ValueError as error:
        raise ConfigError(f"invalid date {value!r}: use YYYY-MM-DD") from error


def _partitions(
    store: ParquetStore, source: str, asset: str, start: date, end: date
) -> list[Path]:
    root = store.root / f"source={source}" / f"asset={asset}"
    days = [
        d
        for d in sorted(root.glob("date=*"))
        if start.isoformat() <= d.name.split("=", 1)[1] <= end.isoformat()
    ]
    return days


def _iso(moment: pd.Timestamp) -> str:
    return moment.isoformat()


def find_gaps(
    frame: pd.DataFrame, threshold_s: float = GAP_SECONDS
) -> list[dict[str, Any]]:
    """Gaps longer than `threshold_s` between consecutive live readings."""
    live = frame[(frame["ingested_at"] - frame["t"]) < LIVE_LAG]
    times = live["t"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) < 2:
        return []
    spacing = times.diff()
    gaps = []
    for index in spacing[spacing > pd.Timedelta(threshold_s, unit="s")].index:
        before, after = times[index - 1], times[index]
        gaps.append(
            {
                "start": _iso(before),
                "end": _iso(after),
                "seconds": round((after - before).total_seconds(), 3),
            }
        )
    return gaps


def build_snapshot(
    store: ParquetStore,
    *,
    start: date | str,
    end: date | str,
    name: str,
    source: str = "iss",
    asset: str = "ISS",
    today: date | None = None,
) -> dict[str, Any]:
    """Describe the store's files for `source`/`asset` from `start` to `end` (inclusive)."""
    first, last = _day(start), _day(end)
    if last < first:
        raise ConfigError("end must not be before start")
    current = today or datetime.now(UTC).date()

    partitions: list[dict[str, Any]] = []
    for directory in _partitions(store, source, asset, first, last):
        files = sorted(directory.glob("part-*.parquet"))
        if not files:
            continue
        records = [
            {
                "name": f.name,
                "sha256": file_sha256(f),
                "rows": pq.ParquetFile(f).metadata.num_rows,
                "size": f.stat().st_size,
            }
            for f in files
        ]
        partitions.append({"date": directory.name.split("=", 1)[1], "files": records})

    frame = store.read_readings(
        source=source,
        assets=[asset],
        start=pd.Timestamp(first, tz="UTC"),
        end=pd.Timestamp(last + timedelta(days=1), tz="UTC"),
    )
    unique_by_day = frame.groupby(frame["t"].dt.strftime("%Y-%m-%d")).size().to_dict()
    for partition in partitions:
        partition["rows_raw"] = sum(f["rows"] for f in partition["files"])
        partition["rows_unique"] = int(unique_by_day.get(partition["date"], 0))

    channels = {
        str(channel): {
            "rows": len(group),
            "first_t": _iso(group["t"].min()),
            "last_t": _iso(group["t"].max()),
        }
        for channel, group in frame.groupby("channel", sort=True)
    }
    files_total = sum(len(p["files"]) for p in partitions)
    return {
        "name": name,
        "source": source,
        "asset": asset,
        "start": first.isoformat(),
        "end": last.isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "complete": last < current,
        "compacted": all(len(p["files"]) == 1 for p in partitions),
        "gap_threshold_s": GAP_SECONDS,
        "gap_definition": (
            "event-time gaps between consecutive readings that arrived within 10 minutes "
            "of their event time (any channel)"
        ),
        "partitions": partitions,
        "channels": channels,
        "gaps": find_gaps(frame),
        "totals": {
            "partitions": len(partitions),
            "files": files_total,
            "rows_raw": sum(p["rows_raw"] for p in partitions),
            "rows_unique": len(drop_duplicate_readings(frame)),
            "channels": len(channels),
            "size_bytes": sum(f["size"] for p in partitions for f in p["files"]),
            "first_t": _iso(frame["t"].min()) if len(frame) else None,
            "last_t": _iso(frame["t"].max()) if len(frame) else None,
        },
    }


def write_snapshot(directory: Path, name: str, snapshot: dict[str, Any]) -> Path:
    """Write `iss_<name>.json` (or `<source>_<name>.json`) atomically."""
    path = directory / f"{snapshot['source']}_{name}.json"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def verify_snapshot(store: ParquetStore, snapshot: dict[str, Any]) -> list[str]:
    """Compare the files on disk with a snapshot. Returns the problems found."""
    problems: list[str] = []
    base = store.root / f"source={snapshot['source']}" / f"asset={snapshot['asset']}"
    recorded: set[Path] = set()
    for partition in snapshot["partitions"]:
        directory = base / f"date={partition['date']}"
        for record in partition["files"]:
            path = directory / record["name"]
            recorded.add(path)
            if not path.exists():
                problems.append(f"missing: {path.relative_to(store.root)}")
            elif file_sha256(path) != record["sha256"]:
                problems.append(f"changed: {path.relative_to(store.root)}")
    for directory in _partitions(
        store,
        snapshot["source"],
        snapshot["asset"],
        date.fromisoformat(snapshot["start"]),
        date.fromisoformat(snapshot["end"]),
    ):
        for path in sorted(directory.glob("part-*.parquet")):
            if path not in recorded:
                problems.append(f"new file: {path.relative_to(store.root)}")
    return problems
