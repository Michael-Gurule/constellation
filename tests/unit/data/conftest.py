"""Builds small benchmark trees with the same file layout as the real downloads."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "benchmarks"


@pytest.fixture
def smap_root(tmp_path):
    root = tmp_path / "smap_msl"
    root.mkdir()
    shutil.copy(
        FIXTURES / "smap_msl" / "labeled_anomalies.csv", root / "labeled_anomalies.csv"
    )
    # The Kaggle release nests the arrays one level down, under data/data/.
    for split, rows in (("train", 300), ("test", 200)):
        folder = root / "data" / "data" / split
        folder.mkdir(parents=True)
        # T-10 has arrays but no row in the label file, as in the published data.
        for channel, columns in (
            ("P-1", 25),
            ("E-1", 25),
            ("T-1", 55),
            ("P-2", 25),
            ("T-10", 55),
        ):
            values = np.linspace(-1, 1, rows)
            array = np.zeros((rows, columns))
            array[:, 0] = values
            array[::7, 1] = 1.0  # one-hot command columns
            np.save(folder / f"{channel}.npy", array)
    return root


@pytest.fixture
def opssat_root(tmp_path):
    root = tmp_path / "opssat"
    root.mkdir()
    shutil.copy(FIXTURES / "opssat" / "segments.csv", root / "segments.csv")
    return root


@pytest.fixture
def esa_root(tmp_path):
    root = tmp_path / "esa_adb"
    mission = root / "ESA-Mission1"
    (mission / "channels").mkdir(parents=True)
    for name in ("labels.csv", "anomaly_types.csv", "telecommands.csv"):
        shutil.copy(FIXTURES / "esa" / "ESA-Mission1" / name, mission / name)
    rng = np.random.default_rng(5)
    for channel in ("channel_41", "channel_42", "channel_5"):
        # Irregular sampling, as in the real channel files, across the train/test cut.
        offsets = np.sort(
            rng.choice(np.arange(0, 4 * 86400, 5), size=4000, replace=False)
        )
        index = pd.Timestamp("2006-12-30") + pd.to_timedelta(offsets, unit="s")
        values = (
            np.cumsum(rng.normal(size=len(index)))
            if channel == "channel_5"
            else rng.normal(size=len(index))
        )
        frame = pd.DataFrame(
            {channel: values}, index=pd.DatetimeIndex(index, name="timestamp")
        )
        frame.to_pickle(mission / "channels" / f"{channel}.zip")
    return root
