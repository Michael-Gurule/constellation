"""Compaction of finished days: one deduplicated file per `date=` partition.

The collector writes a small part file every flush, so a day ends up as hundreds
of files. Compaction merges the files of a *finished* day into one, removes
duplicate readings and sorts by `(channel, t)`.

The order of operations protects the data: the new file is written under a
temporary name, moved into place, read back and checked against the expected row
count, and only then are the old files deleted. If anything fails before that
point, the new file is removed and the old files are left as they were.
"""

import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from constellation.core.errors import ConfigError, StoreError
from constellation.core.schemas import READINGS_ARROW
from constellation.core.store import ParquetStore, drop_duplicate_readings

logger = logging.getLogger(__name__)

_COLUMNS = READINGS_ARROW.names


@dataclass(frozen=True)
class PartitionPlan:
    """One finished day that can be compacted."""

    source: str
    asset: str
    date: str
    directory: Path
    files: list[Path]
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class CompactionReport:
    """What a compaction run did (or, with `dry_run`, would do)."""

    dry_run: bool
    partitions: list[PartitionPlan] = field(default_factory=list)
    skipped_open: list[str] = field(default_factory=list)

    @property
    def files_before(self) -> int:
        return sum(len(p.files) for p in self.partitions)

    @property
    def files_after(self) -> int:
        return len(self.partitions)

    @property
    def rows_before(self) -> int:
        return sum(p.rows_before for p in self.partitions)

    @property
    def rows_after(self) -> int:
        return sum(p.rows_after for p in self.partitions)


def _read_partition(files: list[Path]) -> pd.DataFrame:
    tables = [pq.read_table(f, schema=READINGS_ARROW) for f in files]
    frame: pd.DataFrame = pa.concat_tables(tables).to_pandas()
    return frame


def _candidates(
    store: ParquetStore, before: str, source: str | None, asset: str | None
) -> list[tuple[str, str, str, Path]]:
    found: list[tuple[str, str, str, Path]] = []
    for source_dir in sorted(store.root.glob("source=*")):
        source_name = source_dir.name.split("=", 1)[1]
        if source is not None and source_name != source:
            continue
        for asset_dir in sorted(source_dir.glob("asset=*")):
            asset_name = asset_dir.name.split("=", 1)[1]
            if asset is not None and asset_name != asset:
                continue
            for day_dir in sorted(asset_dir.glob("date=*")):
                day = day_dir.name.split("=", 1)[1]
                if day < before:
                    found.append((source_name, asset_name, day, day_dir))
    return found


def plan_compaction(
    store: ParquetStore,
    *,
    before: date | str | None = None,
    source: str | None = None,
    asset: str | None = None,
    today: date | None = None,
) -> CompactionReport:
    """Find the partitions that compaction would rewrite, and count their rows.

    A partition qualifies when its date is strictly before `before` (default: the
    current UTC date, so today's open partition is never touched), it holds more
    than one part file, and no writer has left a `.*.tmp` file in it.
    """
    if before is not None:
        try:
            limit = date.fromisoformat(str(before)).isoformat()
        except ValueError as error:
            raise ConfigError(f"invalid date {before!r}: use YYYY-MM-DD") from error
    else:
        limit = (today or datetime.now(UTC).date()).isoformat()
    report = CompactionReport(dry_run=True)
    for source_name, asset_name, day, directory in _candidates(
        store, limit, source, asset
    ):
        files = sorted(directory.glob("part-*.parquet"))
        if len(files) < 2:
            continue
        if any(directory.glob(".*.tmp")):
            report.skipped_open.append(f"{source_name}/{asset_name}/{day}")
            continue
        frame = _read_partition(files)
        report.partitions.append(
            PartitionPlan(
                source_name,
                asset_name,
                day,
                directory,
                files,
                rows_before=len(frame),
                rows_after=len(drop_duplicate_readings(frame)),
            )
        )
    return report


def _compact_partition(plan: PartitionPlan) -> None:
    frame = drop_duplicate_readings(_read_partition(plan.files))
    frame = frame.sort_values(["channel", "t"], kind="stable").reset_index(drop=True)
    if len(frame) != plan.rows_after:
        raise StoreError(
            f"{plan.directory}: expected {plan.rows_after} unique rows, found {len(frame)}; "
            "the partition changed during compaction"
        )
    name = f"part-{uuid.uuid4()}.parquet"
    final = plan.directory / name
    temporary = plan.directory / f".{name}.tmp"
    originals = set(plan.files)
    try:
        table = pa.Table.from_pandas(
            frame[_COLUMNS], schema=READINGS_ARROW, preserve_index=False
        )
        pq.write_table(table, temporary)
        os.replace(temporary, final)
        if pq.read_table(final).num_rows != len(frame):
            raise StoreError(
                f"{final}: row count after writing does not match; keeping the original files"
            )
        # The compacted file is complete and verified. Anything from here on only
        # removes data that it already holds.
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        with contextlib.suppress(OSError):
            if final.exists() and final not in originals:
                final.unlink()
        raise
    for original in plan.files:
        original.unlink()


def compact(
    store: ParquetStore,
    *,
    before: date | str | None = None,
    source: str | None = None,
    asset: str | None = None,
    dry_run: bool = False,
    today: date | None = None,
) -> CompactionReport:
    """Compact every qualifying partition; see `plan_compaction` for the scope.

    Idempotent: a compacted day holds one file and is skipped next time.
    """
    report = plan_compaction(
        store, before=before, source=source, asset=asset, today=today
    )
    report.dry_run = dry_run
    if dry_run:
        return report
    for plan in report.partitions:
        _compact_partition(plan)
        logger.info(
            "compacted %s/%s/%s: %d files, %d rows to 1 file, %d rows",
            plan.source,
            plan.asset,
            plan.date,
            len(plan.files),
            plan.rows_before,
            plan.rows_after,
        )
    return report
