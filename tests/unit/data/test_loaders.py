import logging

import numpy as np
import pandas as pd
import pytest

from constellation.core.errors import StoreError
from constellation.core.schemas import validate_labels
from constellation.data.benchmarks import (
    BENCHMARKS,
    EsaAdbBenchmark,
    OpsSatBenchmark,
    SmapMslBenchmark,
    get_benchmark,
    smap_msl,
)
from constellation.data.benchmarks.smap_msl import parse_classes
from constellation.data.labels import labels_frame, mask_from_intervals

# -- SMAP / MSL -------------------------------------------------------------------------


def test_smap_msl_series_use_the_first_column_and_inclusive_label_ends(smap_root):
    series = {s.channel: s for s in SmapMslBenchmark(smap_root).load()}

    assert set(series) == {"P-1", "E-1", "T-1", "P-2"}  # T-10 has no label row
    p1 = series["P-1"]
    assert (p1.dataset, p1.asset_id) == ("smap_msl", "SMAP")
    assert len(p1.train) == 300
    assert len(p1.test) == 200
    assert p1.test.ndim == 1
    assert p1.test[0] == -1.0
    assert p1.test_labels == [
        (30, 40),
        (120, 151),
    ]  # [30, 39] and [120, 150], inclusive
    assert p1.meta["n_input_columns"] == 25
    assert p1.meta["anomaly_classes"] == ["point", "contextual"]


def test_smap_msl_clips_and_merges_labels(smap_root):
    series = {s.channel: s for s in SmapMslBenchmark(smap_root).load()}

    assert series["E-1"].test_labels == [(190, 200)]  # runs past the end of the data
    assert series["T-1"].test_labels == [(10, 26)]  # [10, 20] and [15, 25] overlap
    assert series["T-1"].asset_id == "MSL"
    assert series["T-1"].meta["n_input_columns"] == 55


def test_smap_msl_classes_are_parsed_from_unquoted_lists(smap_root):
    series = {s.channel: s for s in SmapMslBenchmark(smap_root).load()}

    assert series["E-1"].meta["anomaly_classes"] == ["contextual"]
    assert series["T-1"].meta["anomaly_classes"] == ["point", "point"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[contextual, contextual]", ["contextual", "contextual"]),
        ("[point]", ["point"]),
        ("['point', 'contextual']", ["point", "contextual"]),
        ('["point"]', ["point"]),
        ("[]", []),
        ("", []),
    ],
)
def test_parse_classes_accepts_quoted_and_unquoted_lists(text, expected):
    assert parse_classes(text) == expected


def test_smap_msl_merges_the_rows_of_a_duplicated_channel(smap_root, caplog):
    with caplog.at_level(logging.DEBUG, logger=smap_msl.__name__):
        series = SmapMslBenchmark(smap_root).load()

    p2 = [s for s in series if s.channel == "P-2"]
    assert len(p2) == 1  # one series, not two
    assert p2[0].test_labels == [
        (50, 100)
    ]  # [50, 80] and [70, 99], merged, ends inclusive
    assert p2[0].meta["label_rows"] == 2
    assert p2[0].meta["anomaly_classes"] == ["contextual", "point"]
    assert "P-2 has 2 label rows" in caplog.text


def test_smap_msl_skips_arrays_without_a_label_row_and_says_so(smap_root, caplog):
    with caplog.at_level(logging.DEBUG, logger=smap_msl.__name__):
        series = SmapMslBenchmark(smap_root).load()

    assert "T-10" not in {s.channel for s in series}
    assert "skipping T-10" in caplog.text
    assert [s.meta["label_rows"] for s in series if s.channel == "P-1"] == [1]


def test_smap_msl_warns_when_an_array_disagrees_with_the_label_file(smap_root, caplog):
    labels = smap_root / "labeled_anomalies.csv"
    labels.write_text(
        labels.read_text().replace("[contextual],200", "[contextual],999")
    )

    with caplog.at_level(logging.WARNING, logger=smap_msl.__name__):
        SmapMslBenchmark(smap_root).load(["E-1"])

    assert "E-1: test array has 200 values but the label file says [999]" in caplog.text


def test_smap_msl_channel_selection_and_errors(smap_root, tmp_path):
    benchmark = SmapMslBenchmark(smap_root)

    assert [s.channel for s in benchmark.load(["E-1"])] == ["E-1"]
    with pytest.raises(StoreError, match="unknown SMAP/MSL channels"):
        benchmark.load(["X-9"])
    with pytest.raises(StoreError, match=r"labeled_anomalies\.csv not found"):
        SmapMslBenchmark(tmp_path / "empty").load()


def test_smap_msl_needs_the_arrays(smap_root):
    for path in smap_root.rglob("P-1.npy"):
        path.unlink()

    with pytest.raises(StoreError, match="missing arrays for channel P-1"):
        SmapMslBenchmark(smap_root).load()


def test_smap_msl_without_train_and_test_folders(tmp_path):
    root = tmp_path / "smap"
    root.mkdir()
    (root / "labeled_anomalies.csv").write_text(
        "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
    )

    with pytest.raises(StoreError, match="train/ and test/"):
        SmapMslBenchmark(root).load()


def test_smap_labels_convert_to_the_contract(smap_root):
    series = SmapMslBenchmark(smap_root).load(["P-1"])[0]

    frame = labels_frame(
        series.test_labels,
        asset_id=series.asset_id,
        channel=series.channel,
        start=pd.Timestamp(series.meta["test_start"]),
        cadence=pd.Timedelta(series.meta["cadence_s"], unit="s"),
        source=series.dataset,
    )

    validate_labels(frame)
    assert len(frame) == 2


# -- OPSSAT-AD ------------------------------------------------------------------------------


def test_opssat_segments_table(opssat_root):
    table = OpsSatBenchmark(opssat_root).segments()

    assert table["segment"].tolist() == [1, 2, 3, 4, 5, 6]
    assert table.set_index("segment")["anomaly"].to_dict() == {
        1: False,
        2: True,
        3: False,
        4: True,
        5: False,
        6: False,
    }
    assert table.set_index("segment")["train"].to_dict() == {
        1: True,
        2: True,
        3: False,
        4: False,
        5: True,
        6: False,
    }
    seg3 = table[table["segment"] == 3].iloc[0]
    assert (seg3["sampling_s"], seg3["n"]) == (5, 2)
    assert seg3["start"] == pd.Timestamp("2022-06-03", tz="UTC")


def test_opssat_series_follow_the_recommended_split(opssat_root):
    series = {s.channel: s for s in OpsSatBenchmark(opssat_root).load()}

    first = series["CADC0872"]
    assert (first.dataset, first.asset_id) == ("opssat", "OPS-SAT")
    assert first.train.tolist() == [
        1.0,
        2.0,
        3.0,
    ]  # the anomalous training segment is left out
    assert first.test.tolist() == [4.0, 5.0, 20.0, 21.0, 22.0]
    assert first.test_labels == [(2, 5)]
    assert first.meta["segment_ids"] == [3, 4]
    assert first.meta["segment_bounds"] == [(0, 2), (2, 5)]
    assert first.meta["sampling_s"] == [5, 1]
    assert len(first.meta["test_times"]) == 5
    second = series["CADC0873"]
    assert second.train.tolist() == [-1.0, -2.0]
    assert second.test.tolist() == [-3.0, -4.0]
    assert second.test_labels == []


def test_opssat_labels_follow_the_contract(opssat_root):
    labels = OpsSatBenchmark(opssat_root).labels()

    validate_labels(labels)
    assert labels["channel"].tolist() == ["CADC0872", "CADC0872"]
    assert labels["start"].tolist() == [
        pd.Timestamp("2022-06-02T00:00:00", tz="UTC"),
        pd.Timestamp("2022-06-04T00:00:00", tz="UTC"),
    ]
    assert labels["end"].iloc[1] == pd.Timestamp("2022-06-04T00:00:02", tz="UTC")
    assert set(labels["asset_id"]) == {"OPS-SAT"}


def test_opssat_errors(opssat_root, tmp_path):
    with pytest.raises(StoreError, match="unknown OPSSAT-AD channel"):
        OpsSatBenchmark(opssat_root).load(["nope"])
    with pytest.raises(StoreError, match="data fetch opssat"):
        OpsSatBenchmark(tmp_path / "missing").segments()


# -- ESA-ADB --------------------------------------------------------------------------------


CHANNELS = ["channel_41", "channel_42"]


def test_esa_events_are_joined_with_their_category(esa_root):
    events = EsaAdbBenchmark(esa_root).events()

    assert events.set_index("ID")["Category"].to_dict() == {
        "id_a": "Anomaly",
        "id_b": "Anomaly",
        "id_c": "Rare Event",
        "id_d": "Anomaly",
        "id_e": "Communication Gap",
    }
    assert events["StartTime"].dt.tz is None


def test_esa_series_are_resampled_with_zero_order_hold_and_split_at_the_cut(esa_root):
    series = {s.channel: s for s in EsaAdbBenchmark(esa_root).load(CHANNELS)}

    s = series["channel_41"]
    assert (s.dataset, s.asset_id) == ("esa_adb", "ESA-Mission1")
    assert s.meta["cadence_s"] == 30.0
    assert s.meta["split"].startswith("2007-01-01")
    assert s.meta["test_start"] == "2007-01-01T00:00:30+00:00"
    # Train runs to the split inclusive, so the first test cell is the one after.
    assert s.meta["train_start"] <= "2006-12-30T00:00:30+00:00"
    assert len(s.train) > 0
    assert len(s.test) > 0
    assert not np.isnan(s.train).any()
    assert not np.isnan(s.test).any()

    raw = pd.read_pickle(esa_root / "ESA-Mission1" / "channels" / "channel_41.zip")[
        "channel_41"
    ]
    for stamp in ("2007-01-01T06:00:00", "2007-01-02T11:59:30"):
        position = (
            pd.Timestamp(stamp) - pd.Timestamp("2007-01-01T00:00:30")
        ) / pd.Timedelta(30, unit="s")
        assert s.test[int(position)] == pytest.approx(raw.asof(pd.Timestamp(stamp)))


def test_esa_labels_include_events_shorter_than_one_cell(esa_root):
    s = {s.channel: s for s in EsaAdbBenchmark(esa_root).load(CHANNELS)}["channel_41"]

    # id_a lasts 10 s, inside the single 30 s cell starting at 06:00:00 (test position 719).
    # id_b spans 12:00:00 to 12:10:00 inclusive: 21 cells starting at position 1439.
    assert s.test_labels == [(719, 720), (1439, 1460)]
    assert s.meta["rare_events"] == [
        (2159, 2170)
    ]  # id_c, kept apart from the anomalies
    assert s.meta["gaps"] == []
    mask = mask_from_intervals(len(s.test), s.test_labels)
    assert mask.sum() == 22


def test_esa_labels_before_the_split_are_kept_for_training_only(esa_root):
    s = {s.channel: s for s in EsaAdbBenchmark(esa_root).load(CHANNELS)}["channel_41"]

    assert len(s.meta["train_anomaly_intervals"]) == 1  # id_d, on 2006-12-30
    assert all(stop <= len(s.test) for _, stop in s.test_labels)


def test_esa_communication_gaps_are_not_anomalies(esa_root):
    s = {s.channel: s for s in EsaAdbBenchmark(esa_root).load(CHANNELS)}["channel_42"]

    assert s.test_labels == []
    assert len(s.meta["gaps"]) == 1  # id_e


def test_esa_monotonic_channels_are_differenced(esa_root):
    s = EsaAdbBenchmark(esa_root).load(["channel_5"])[0]

    raw = pd.read_pickle(esa_root / "ESA-Mission1" / "channels" / "channel_5.zip")[
        "channel_5"
    ]
    assert np.abs(s.test).max() < 0.5 * raw.abs().max()  # steps, not a running total
    assert abs(s.test.mean()) < 0.5


def test_esa_channels_default_to_the_lightweight_subset(esa_root):
    benchmark = EsaAdbBenchmark(esa_root)

    assert benchmark.default_channels == tuple(f"channel_{n}" for n in range(41, 47))
    with pytest.raises(StoreError, match=r"channel_43\.zip not found"):
        benchmark.load()  # only channels 41 and 42 exist in the fixture


def test_esa_split_and_resolution_are_configurable(esa_root):
    s = EsaAdbBenchmark(esa_root, resample="1m", split="2006-12-31").load(
        ["channel_41"]
    )[0]

    assert s.meta["cadence_s"] == 60.0
    assert s.meta["test_start"] == "2006-12-31T00:01:00+00:00"


def test_esa_root_may_be_the_mission_folder_itself(esa_root):
    direct = EsaAdbBenchmark(esa_root / "ESA-Mission1")

    assert [s.channel for s in direct.load(["channel_41"])] == ["channel_41"]


# -- registry ---------------------------------------------------------------------------------


def test_get_benchmark_builds_loaders_for_the_external_folder(tmp_path):
    for name, cls in BENCHMARKS.items():
        benchmark = get_benchmark(name, tmp_path)
        assert isinstance(benchmark, cls)
        assert benchmark.name == name
        assert benchmark.root == tmp_path / "external" / name  # type: ignore[attr-defined]


def test_get_benchmark_rejects_unknown_names(tmp_path):
    from constellation.core.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown benchmark"):
        get_benchmark("nope", tmp_path)
