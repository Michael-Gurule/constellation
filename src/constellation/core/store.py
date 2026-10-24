"""Event-time Parquet store for readings.

Layout: `<root>/source=<source>/asset=<asset_id>/date=<YYYY-MM-DD>/part-<uuid>.parquet`,
where the date is the UTC date of the event time `t`, never the time of writing.
"""

import contextlib
import os
import re
import uuid
from collections.abc import Collection
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from constellation.core.errors import SchemaError, StoreError
from constellation.core.schemas import (
    READINGS,
    READINGS_ARROW,
    empty_readings,
    validate_readings,
)
from constellation.core.timeutils import ensure_utc

_UTC_NS_TYPE = pa.timestamp("ns", tz="UTC")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_PARTITIONING = ds.partitioning(
    pa.schema([("source", pa.string()), ("asset", pa.string()), ("date", pa.string())]),
    flavor="hive",
)
_DATASET_SCHEMA = READINGS_ARROW.append(pa.field("asset", pa.string())).append(
    pa.field("date", pa.string())
)
_COLUMNS = [c.name for c in READINGS]
_SORT_KEYS = ["source", "asset_id", "channel", "t"]
# Two rows that agree on all of these are the same reading, however often it was written.
DEDUPE_KEYS = [*_SORT_KEYS, "value", "value_text", "status"]


def drop_duplicate_readings(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated readings, keeping the copy with the earliest `ingested_at`.

    Rows are duplicates when they are identical on `source`, `asset_id`, `channel`,
    `t`, `value`, `value_text` and `status` (missing values compare equal). Rows
    that differ in any of those are all kept.
    """
    if frame.empty:
        return frame
    ordered = frame.sort_values("ingested_at", kind="stable")
    unique: pd.DataFrame = ordered.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    return unique


def _date_of(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d")


class ParquetStore:
    """Partitioned, append-only storage for READINGS frames."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- writing -----------------------------------------------------------

    def write_readings(self, df: pd.DataFrame) -> list[Path]:
        """Append readings, one file per (source, asset, event-date) partition.

        Every file is written under a temporary name and renamed once all
        files of the call have been written, so a failure leaves no partial
        files behind. Returns the paths of the files written.
        """
        validate_readings(df)
        if df.empty:
            return []
        for column in ("source", "asset_id"):
            unsafe = sorted({v for v in df[column].unique() if not _SAFE_NAME.match(v)})
            if unsafe:
                raise SchemaError(
                    f"{column}: values {unsafe} cannot be used as partition names "
                    "(allowed characters: letters, digits, '.', '_', '-')"
                )

        frame = df[_COLUMNS].copy()
        for column in ("value_text", "status"):
            values = frame[column].astype(object).to_numpy(copy=True)
            values[frame[column].isna().to_numpy()] = None
            frame[column] = values
        frame["_date"] = frame["t"].dt.strftime("%Y-%m-%d")

        staged: list[tuple[Path, Path]] = []
        try:
            for (source, asset, date), part in frame.groupby(
                ["source", "asset_id", "_date"], sort=True
            ):
                directory = (
                    self.root / f"source={source}" / f"asset={asset}" / f"date={date}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                name = f"part-{uuid.uuid4()}.parquet"
                final = directory / name
                temporary = directory / f".{name}.tmp"
                table = pa.Table.from_pandas(
                    part.drop(columns="_date"),
                    schema=READINGS_ARROW,
                    preserve_index=False,
                )
                pq.write_table(table, temporary)
                staged.append((temporary, final))
            for temporary, final in staged:
                os.replace(temporary, final)
        except BaseException:
            for temporary, _ in staged:
                with contextlib.suppress(OSError):
                    temporary.unlink()
            raise
        return [final for _, final in staged]

    # -- reading -----------------------------------------------------------

    def _dataset(self) -> ds.Dataset | None:
        if not self.root.exists():
            return None
        try:
            return ds.dataset(
                self.root,
                format="parquet",
                schema=_DATASET_SCHEMA,
                partitioning=_PARTITIONING,
            )
        except (FileNotFoundError, pa.ArrowInvalid) as error:
            raise StoreError(f"cannot open store at {self.root}: {error}") from error

    def read_readings(
        self,
        *,
        source: str | None = None,
        assets: Collection[str] | None = None,
        channels: Collection[str] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        dedupe: bool = True,
    ) -> pd.DataFrame:
        """Read readings matching the filters, sorted by source, asset, channel, t.

        `start` is inclusive and `end` is exclusive. Filters are pushed down to
        partition pruning and Parquet statistics. With `dedupe` (the default),
        readings that were written more than once, for example by overlapping
        part files in a synced copy, are returned once; see `drop_duplicate_readings`.
        """
        dataset = self._dataset()
        if dataset is None:
            return empty_readings()

        expression: ds.Expression | None = None

        def add(clause: ds.Expression) -> None:
            nonlocal expression
            expression = clause if expression is None else expression & clause

        if source is not None:
            add(ds.field("source") == source)
        if assets is not None:
            add(ds.field("asset").isin(list(assets)))
        if channels is not None:
            add(ds.field("channel").isin(list(channels)))
        if start is not None:
            start_ts = ensure_utc(start)
            add(ds.field("date") >= _date_of(start_ts))
            add(ds.field("t") >= pa.scalar(start_ts.to_pydatetime(), _UTC_NS_TYPE))
        if end is not None:
            end_ts = ensure_utc(end)
            add(ds.field("date") <= _date_of(end_ts))
            add(ds.field("t") < pa.scalar(end_ts.to_pydatetime(), _UTC_NS_TYPE))

        table = dataset.to_table(columns=_COLUMNS, filter=expression)
        if table.num_rows == 0:
            return empty_readings()
        frame = table.to_pandas()
        if dedupe:
            frame = drop_duplicate_readings(frame)
        result: pd.DataFrame = frame.sort_values(_SORT_KEYS, kind="stable").reset_index(
            drop=True
        )
        return result

    # -- inspection --------------------------------------------------------

    def list_assets(self) -> list[tuple[str, str]]:
        """Return the (source, asset_id) pairs present, sorted."""
        pairs: list[tuple[str, str]] = []
        if not self.root.exists():
            return pairs
        for source_dir in sorted(self.root.glob("source=*")):
            for asset_dir in sorted(source_dir.glob("asset=*")):
                if any(asset_dir.rglob("part-*.parquet")):
                    pairs.append(
                        (
                            source_dir.name.split("=", 1)[1],
                            asset_dir.name.split("=", 1)[1],
                        )
                    )
        return pairs

    def stats(self, dedupe: bool = True) -> pd.DataFrame:
        """Rows and first/last event time for every (source, asset, channel).

        Duplicate readings (see `drop_duplicate_readings`) are counted once unless
        `dedupe` is false, which is cheaper on a large store.
        """
        columns = ["source", "asset_id", "channel", "rows", "first_t", "last_t"]
        dataset = self._dataset()
        if dataset is None:
            return pd.DataFrame(columns=columns)
        table = dataset.to_table(columns=DEDUPE_KEYS if dedupe else _SORT_KEYS)
        if table.num_rows == 0:
            return pd.DataFrame(columns=columns)
        frame = table.to_pandas()
        if dedupe:
            frame = frame.drop_duplicates(subset=DEDUPE_KEYS)
        grouped = (
            frame.groupby(["source", "asset_id", "channel"], sort=True)["t"]
            .agg(rows="size", first_t="min", last_t="max")
            .reset_index()
        )
        result: pd.DataFrame = grouped[columns]
        return result
