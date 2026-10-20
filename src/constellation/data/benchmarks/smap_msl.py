"""NASA SMAP and MSL telemetry with expert-labeled anomalies (Telemanom).

Source: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding", KDD 2018. The data are anonymized (channel
names, time and scale) and arrive as one `.npy` array per channel in `train/` and
`test/`; the first column is the telemetry value and the others are one-hot
encoded commands. The authors scaled the values per channel; in the files as
published, column 0 lies between -1.48 and 4.16 in the training arrays and
between -1.42 and 258.1 in the test arrays (channel M-6), so the values are not
confined to (-1, 1). `labeled_anomalies.csv` lists the anomalous index ranges in
the test arrays, with inclusive ends.

The published label file has three quirks that the loader handles:

- `class` is written with unquoted words, `[contextual, point]`, which is not
  valid Python or JSON, so it is parsed by hand (`anomaly_sequences` is JSON);
- channel P-2 has two rows with different ranges; their intervals are merged
  into one series and `meta["label_rows"]` records that there were two;
- channel T-10 has arrays but no label row; it is skipped and logged at debug
  level, which leaves 81 labeled series.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from constellation.core.errors import StoreError
from constellation.data.benchmarks.base import BenchmarkSeries
from constellation.data.labels import merge_intervals

logger = logging.getLogger(__name__)

NAME = "smap_msl"
LABELS_FILE = "labeled_anomalies.csv"
# The source anonymizes time, so positions are placed on a synthetic timeline.
TIMELINE_START = "2000-01-01T00:00:00Z"
CADENCE_S = 1.0


def _find_data_dir(root: Path) -> Path:
    """The directory holding `train/` and `test/`, wherever the archive put it."""
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise StoreError(f"no train/ and test/ directories found under {root}")


def parse_classes(text: str) -> list[str]:
    """Anomaly classes from the `class` column, quoted or not: `[a, b]` or `['a', 'b']`."""
    inner = text.strip().strip("[]")
    return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]


class SmapMslBenchmark:
    name = NAME

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _labels_path(self) -> Path:
        direct = self.root / LABELS_FILE
        if direct.exists():
            return direct
        found = sorted(self.root.rglob(LABELS_FILE))
        if not found:
            raise StoreError(f"{LABELS_FILE} not found under {self.root}")
        return found[0]

    def channels(self) -> pd.DataFrame:
        """The label table: `chan_id, spacecraft, anomaly_sequences, class, num_values`."""
        return pd.read_csv(self._labels_path())

    def load(self, channels: Sequence[str] | None = None) -> list[BenchmarkSeries]:
        """One series per labeled channel, in the order of the label file.

        A channel that appears in several rows has the intervals of all rows merged.
        """
        table = self.channels()
        data = _find_data_dir(self.root)
        unlabeled = sorted(
            {p.stem for p in (data / "test").glob("*.npy")} - set(table["chan_id"])
        )
        for channel in unlabeled:
            logger.debug("skipping %s: arrays without a label row", channel)
        if channels is not None:
            unknown = sorted(set(channels) - set(table["chan_id"]))
            if unknown:
                raise StoreError(f"unknown SMAP/MSL channels: {unknown}")
            table = table[table["chan_id"].isin(channels)]
        series: list[BenchmarkSeries] = []
        for key, rows in table.groupby("chan_id", sort=False):
            channel = str(key)
            train_file, test_file = (
                data / "train" / f"{channel}.npy",
                data / "test" / f"{channel}.npy",
            )
            if not (train_file.exists() and test_file.exists()):
                raise StoreError(f"missing arrays for channel {channel} under {data}")
            train_all, test_all = np.load(train_file), np.load(test_file)
            test = np.asarray(test_all.reshape(len(test_all), -1)[:, 0], dtype=float)
            train = np.asarray(train_all.reshape(len(train_all), -1)[:, 0], dtype=float)

            declared = {int(n) for n in rows["num_values"]}
            if declared != {len(test)}:
                logger.warning(
                    "%s: test array has %d values but the label file says %s",
                    channel,
                    len(test),
                    sorted(declared),
                )
            intervals = merge_intervals(
                (int(a), min(int(b) + 1, len(test)))  # ends are inclusive
                for text in rows["anomaly_sequences"]
                for a, b in json.loads(str(text))
            )
            classes = [c for text in rows["class"] for c in parse_classes(str(text))]
            if len(rows) > 1:
                logger.debug(
                    "%s has %d label rows; intervals merged", channel, len(rows)
                )
            series.append(
                BenchmarkSeries(
                    dataset=NAME,
                    asset_id=str(rows["spacecraft"].iloc[0]),
                    channel=channel,
                    train=train,
                    test=test,
                    test_labels=intervals,
                    meta={
                        "cadence_s": CADENCE_S,
                        "test_start": TIMELINE_START,
                        "timeline": "synthetic: the source anonymizes time",
                        "anomaly_classes": classes,
                        "label_rows": len(rows),
                        "num_values": len(test),
                        "n_input_columns": int(
                            test_all.reshape(len(test_all), -1).shape[1]
                        ),
                    },
                )
            )
        return series
