"""OPSSAT-AD: telemetry segments from ESA's OPS-SAT CubeSat (Zenodo 12588359).

`segments.csv` has one row per sample (`channel, timestamp, value, label,
sampling, anomaly, segment, train`); a segment is labeled as a whole, and the
`train` flag gives the split that the dataset's authors recommend. Segments
are short windows (8 to about 1000 samples, 1 s or 5 s apart) that are not
contiguous in time.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from constellation.core.errors import StoreError
from constellation.core.schemas import empty_labels, validate_labels
from constellation.data.benchmarks.base import BenchmarkSeries, Interval

NAME = "opssat"
SOURCE_FILE = "segments.csv"
ASSET_ID = "OPS-SAT"


class OpsSatBenchmark:
    name = NAME

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._samples: pd.DataFrame | None = None

    def samples(self) -> pd.DataFrame:
        """All samples, ordered by segment and time."""
        if self._samples is None:
            path = self.root / SOURCE_FILE
            if not path.exists():
                raise StoreError(
                    f"{path} not found; run `constellation data fetch {NAME}`"
                )
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype(
                "datetime64[ns, UTC]"
            )
            self._samples = frame.sort_values(
                ["segment", "timestamp"], kind="stable"
            ).reset_index(drop=True)
        return self._samples

    def segments(self) -> pd.DataFrame:
        """One row per segment with its label, split and extent."""
        grouped = self.samples().groupby("segment", sort=True)
        table = grouped.agg(
            channel=("channel", "first"),
            anomaly=("anomaly", "first"),
            train=("train", "first"),
            sampling_s=("sampling", "first"),
            n=("value", "size"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        ).reset_index()
        table["anomaly"] = table["anomaly"].astype(bool)
        table["train"] = table["train"].astype(bool)
        return table

    def labels(self) -> pd.DataFrame:
        """Anomalous segments as the LABELS contract (test and train split alike)."""
        table = self.segments()
        anomalous = table[table["anomaly"]]
        if anomalous.empty:
            return empty_labels()
        frame = pd.DataFrame(
            {
                "asset_id": ASSET_ID,
                "channel": anomalous["channel"].astype(object).to_numpy(),
                "start": anomalous["start"].to_numpy(),
                "end": anomalous["end"].to_numpy(),
                "kind": "anomaly",
                "fault_class": pd.Series(
                    [None] * len(anomalous), dtype=object
                ).to_numpy(),
                "severity": pd.Series([None] * len(anomalous), dtype=object).to_numpy(),
                "source": NAME,
            }
        )
        for column in ("start", "end"):
            frame[column] = pd.to_datetime(frame[column], utc=True).astype(
                "datetime64[ns, UTC]"
            )
        for column in ("channel", "fault_class", "severity"):
            frame[column] = frame[column].astype(object)
        return validate_labels(frame)

    def load(self, channels: Sequence[str] | None = None) -> list[BenchmarkSeries]:
        """One series per channel, built from the recommended split.

        `train` joins the normal training segments and `test` joins all test
        segments, each in time order. `test_labels` mark the anomalous test
        segments, and `meta` holds the segment boundaries and sample times.
        """
        table = self.segments()
        samples = self.samples()
        wanted = (
            sorted(table["channel"].unique()) if channels is None else list(channels)
        )
        series: list[BenchmarkSeries] = []
        for channel in wanted:
            segments = table[table["channel"] == channel].sort_values(
                "start", kind="stable"
            )
            if segments.empty:
                raise StoreError(f"unknown OPSSAT-AD channel {channel!r}")
            rows = segments.to_dict("records")
            channel_samples = samples[samples["channel"] == channel]
            by_segment: dict[int, pd.DataFrame] = {
                int(str(seg)): group
                for seg, group in channel_samples.groupby("segment")
            }
            train_parts = [
                by_segment[int(r["segment"])]["value"].to_numpy(dtype=float)
                for r in rows
                if r["train"] and not r["anomaly"]
            ]
            test_rows = [r for r in rows if not r["train"]]
            test_parts = [by_segment[int(r["segment"])] for r in test_rows]
            test = (
                np.concatenate([p["value"].to_numpy(dtype=float) for p in test_parts])
                if test_parts
                else np.empty(0)
            )
            bounds: list[Interval] = []
            labels: list[Interval] = []
            cursor = 0
            for row, part in zip(test_rows, test_parts, strict=True):
                bounds.append((cursor, cursor + len(part)))
                if row["anomaly"]:
                    labels.append((cursor, cursor + len(part)))
                cursor += len(part)
            test_times = (
                np.concatenate(
                    [
                        p["timestamp"]
                        .dt.tz_convert("UTC")
                        .dt.tz_localize(None)
                        .to_numpy()
                        for p in test_parts
                    ]
                )
                if test_parts
                else np.empty(0, dtype="datetime64[ns]")
            )
            series.append(
                BenchmarkSeries(
                    dataset=NAME,
                    asset_id=ASSET_ID,
                    channel=str(channel),
                    train=np.concatenate(train_parts) if train_parts else np.empty(0),
                    test=test,
                    test_labels=labels,
                    meta={
                        "segment_ids": [int(r["segment"]) for r in test_rows],
                        "segment_bounds": bounds,
                        "test_times": test_times,
                        "sampling_s": [int(r["sampling_s"]) for r in test_rows],
                        "n_train_segments": len(train_parts),
                    },
                )
            )
        return series
