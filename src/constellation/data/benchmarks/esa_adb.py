"""ESA Anomaly Dataset (ESA-AD) and the ESA-ADB benchmark (Zenodo 12528696).

Real telemetry from anonymized ESA missions. A mission folder holds
`labels.csv` (`ID, Channel, StartTime, EndTime`), `anomaly_types.csv`
(`ID, Category, ...`), `telecommands.csv` and `channels/<channel>.zip`, a pickled
data frame with a time index and one column named after the channel.

The preprocessing follows the ESA-ADB repository (https://github.com/kplabs-pl/ESA-ADB,
MIT licence), script `Mission1_semisupervised_prep_from_raw.py`: zero-order-hold
resampling to 30 s, the first difference of the monotonic channels 4 to 11, and
a train/test cut at 2007-01-01 for the 84-month setting. Annotated events that
fall between two grid points are kept by marking every grid cell they touch.
The default `channels` are the lightweight subset of Mission1 (channels 41 to
46); pass `channels` for others.

The channel files are pickles. They are only read after the archive's checksum
has been verified against the one published with the Zenodo record.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from constellation.core.errors import StoreError
from constellation.core.timeutils import parse_duration
from constellation.data.benchmarks.base import BenchmarkSeries, Interval
from constellation.data.labels import intervals_from_mask

NAME = "esa_adb"
ASSET_PREFIX = "ESA"
LIGHTWEIGHT_CHANNELS = tuple(f"channel_{n}" for n in range(41, 47))
MONOTONIC_CHANNELS = range(4, 12)
SPLIT = "2007-01-01"
CATEGORY_ANOMALY = "Anomaly"
CATEGORY_RARE_EVENT = "Rare Event"


class EsaAdbBenchmark:
    name = NAME

    def __init__(
        self,
        root: str | Path,
        mission: str = "Mission1",
        channels: Sequence[str] = LIGHTWEIGHT_CHANNELS,
        resample: str = "30s",
        split: str = SPLIT,
    ) -> None:
        self.root = Path(root)
        self.mission = mission
        self.default_channels = tuple(channels)
        self.step = parse_duration(resample)
        self.freq = f"{int(self.step.total_seconds())}s"
        self.split = pd.Timestamp(split)

    @property
    def folder(self) -> Path:
        direct = self.root / f"ESA-{self.mission}"
        return direct if direct.is_dir() else self.root

    def events(self) -> pd.DataFrame:
        """Annotated events with their category: `ID, Channel, StartTime, EndTime, Category`."""
        labels = pd.read_csv(self.folder / "labels.csv")
        for column in ("StartTime", "EndTime"):
            labels[column] = pd.to_datetime(labels[column], utc=True).dt.tz_localize(
                None
            )
        types = pd.read_csv(self.folder / "anomaly_types.csv")[["ID", "Category"]]
        return labels.merge(types, on="ID", how="left")

    def _read_channel(self, channel: str) -> pd.Series:
        path = self.folder / "channels" / f"{channel}.zip"
        if not path.exists():
            raise StoreError(f"{path} not found; run `constellation data fetch {NAME}`")
        frame = pd.read_pickle(path)
        series = frame[channel] if channel in frame.columns else frame.iloc[:, 0]
        series = series.astype(float)
        index = pd.DatetimeIndex(series.index)
        series.index = (
            index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
        )
        number = int(channel.split("_")[1])
        if number in MONOTONIC_CHANNELS:
            series = pd.Series(
                np.diff(series.to_numpy(), append=series.to_numpy()[-1]),
                index=series.index,
            )
        ordered: pd.Series = series.sort_index()
        return ordered

    def _resample(self, series: pd.Series) -> pd.Series:
        grid = pd.date_range(
            series.index[0].floor(self.freq),
            series.index[-1].ceil(self.freq),
            freq=self.freq,
        )
        held = series.reindex(grid, method="ffill")
        held.iloc[0] = series.iloc[0]
        return held

    def _mask(self, grid: np.ndarray, events: pd.DataFrame) -> np.ndarray:
        """Grid cells touched by any of the events."""
        mask = np.zeros(len(grid), dtype=bool)
        for first_time, last_time in zip(
            events["StartTime"], events["EndTime"], strict=True
        ):
            start = np.datetime64(
                pd.Timestamp(first_time).floor(self.freq).to_datetime64()
            )
            end = np.datetime64(
                pd.Timestamp(last_time).floor(self.freq).to_datetime64()
            )
            mask[
                np.searchsorted(grid, start, side="left") : np.searchsorted(
                    grid, end, side="right"
                )
            ] = True
        return mask

    def load(self, channels: Sequence[str] | None = None) -> list[BenchmarkSeries]:
        wanted = list(channels) if channels is not None else list(self.default_channels)
        events = self.events()
        series: list[BenchmarkSeries] = []
        for channel in wanted:
            held = self._resample(self._read_channel(channel))
            grid = held.index.to_numpy(dtype="datetime64[ns]")
            mine = events[events["Channel"] == channel]
            categories = {
                "anomaly": mine[mine["Category"] == CATEGORY_ANOMALY],
                "rare_event": mine[mine["Category"] == CATEGORY_RARE_EVENT],
                "gap": mine[
                    ~mine["Category"].isin([CATEGORY_ANOMALY, CATEGORY_RARE_EVENT])
                ],
            }
            masks = {
                name: self._mask(grid, frame) for name, frame in categories.items()
            }
            # Train: up to and including the split instant. Test: everything after it.
            cut = int(np.searchsorted(grid, np.datetime64(self.split), side="right"))
            intervals: dict[str, list[Interval]] = {
                name: intervals_from_mask(mask[cut:]) for name, mask in masks.items()
            }
            values = held.to_numpy(dtype=float)
            series.append(
                BenchmarkSeries(
                    dataset=NAME,
                    asset_id=f"{ASSET_PREFIX}-{self.mission}",
                    channel=channel,
                    train=values[:cut],
                    test=values[cut:],
                    test_labels=intervals["anomaly"],
                    meta={
                        "cadence_s": self.step.total_seconds(),
                        "test_start": _iso(grid[cut]) if cut < len(grid) else None,
                        "train_start": _iso(grid[0]),
                        "split": self.split.isoformat(),
                        "rare_events": intervals["rare_event"],
                        "gaps": intervals["gap"],
                        "train_anomaly_intervals": intervals_from_mask(
                            masks["anomaly"][:cut]
                        ),
                        "mission": self.mission,
                    },
                )
            )
        return series


def _iso(moment: np.datetime64) -> str:
    return pd.Timestamp(moment).tz_localize("UTC").isoformat()
