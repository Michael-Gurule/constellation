"""NASA SMAP and MSL telemetry with expert-labeled anomalies (Telemanom).

Source: Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding", KDD 2018. The data are anonymized (channel
names, time and scale), pre-scaled to (-1, 1), and arrive as one `.npy` array
per channel in `train/` and `test/`; the first column is the telemetry value
and the others are one-hot encoded commands. `labeled_anomalies.csv` lists the
anomalous index ranges in the test arrays, with inclusive ends.
"""

import ast
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from constellation.core.errors import StoreError
from constellation.data.benchmarks.base import BenchmarkSeries
from constellation.data.labels import merge_intervals

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
        table = self.channels()
        data = _find_data_dir(self.root)
        if channels is not None:
            unknown = sorted(set(channels) - set(table["chan_id"]))
            if unknown:
                raise StoreError(f"unknown SMAP/MSL channels: {unknown}")
            table = table[table["chan_id"].isin(channels)]
        series: list[BenchmarkSeries] = []
        for row in table.itertuples():
            channel = str(row.chan_id)
            train_file, test_file = (
                data / "train" / f"{channel}.npy",
                data / "test" / f"{channel}.npy",
            )
            if not (train_file.exists() and test_file.exists()):
                raise StoreError(f"missing arrays for channel {channel} under {data}")
            train_all, test_all = np.load(train_file), np.load(test_file)
            test = np.asarray(test_all.reshape(len(test_all), -1)[:, 0], dtype=float)
            train = np.asarray(train_all.reshape(len(train_all), -1)[:, 0], dtype=float)
            sequences = ast.literal_eval(str(row.anomaly_sequences))
            intervals = merge_intervals(
                (int(a), min(int(b) + 1, len(test)))
                for a, b in sequences  # ends are inclusive
            )
            series.append(
                BenchmarkSeries(
                    dataset=NAME,
                    asset_id=str(row.spacecraft),
                    channel=channel,
                    train=train,
                    test=test,
                    test_labels=intervals,
                    meta={
                        "cadence_s": CADENCE_S,
                        "test_start": TIMELINE_START,
                        "timeline": "synthetic: the source anonymizes time",
                        "anomaly_classes": ast.literal_eval(str(row._4))
                        if hasattr(row, "_4")
                        else [],
                        "n_input_columns": int(
                            test_all.reshape(len(test_all), -1).shape[1]
                        ),
                    },
                )
            )
        return series
