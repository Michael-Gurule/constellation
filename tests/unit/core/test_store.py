import pandas as pd
import pyarrow.parquet as pq
import pytest

from constellation.core.errors import SchemaError
from constellation.core.schemas import READINGS, validate_readings
from constellation.core.store import ParquetStore


def _parts(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("part-*.parquet"))


def test_partitions_follow_event_time_across_midnight(tmp_path, make_readings):
    frame = make_readings(
        [
            ("2026-03-01T23:59:59", "temp", 1.0),
            ("2026-03-02T00:00:00", "temp", 2.0),
            ("2026-03-02T00:00:01", "temp", 3.0),
        ]
    )

    written = ParquetStore(tmp_path).write_readings(frame)

    assert len(written) == 2
    directories = sorted({p.parent.relative_to(tmp_path).as_posix() for p in written})
    assert directories == [
        "source=iss/asset=ISS/date=2026-03-01",
        "source=iss/asset=ISS/date=2026-03-02",
    ]


def test_files_are_named_part_uuid_and_hold_the_full_contract(tmp_path, make_readings):
    ParquetStore(tmp_path).write_readings(make_readings([("2026-03-01", "temp", 1.0)]))

    (path,) = tmp_path.rglob("*.parquet")

    assert path.name.startswith("part-")
    assert pq.read_schema(path).names == [c.name for c in READINGS]


def test_round_trip_preserves_every_value(tmp_path, make_readings):
    frame = make_readings(
        [
            ("2026-03-01T10:00:00.123456789", "temp", 1.5),
            ("2026-03-01T10:00:01", "state", float("nan")),
        ]
    )
    frame.loc[1, "value_text"] = "ON"
    frame.loc[0, "status"] = None
    store = ParquetStore(tmp_path)

    store.write_readings(frame)
    result = store.read_readings()

    validate_readings(result)
    expected = frame.sort_values(["channel", "t"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_filters_select_source_assets_channels_and_time(tmp_path, make_readings):
    store = ParquetStore(tmp_path)
    store.write_readings(
        make_readings(
            [
                ("2026-03-01T00:00:00", "a", 1.0),
                ("2026-03-01T12:00:00", "a", 2.0),
                ("2026-03-02T00:00:00", "b", 3.0),
            ]
        )
    )
    store.write_readings(
        make_readings([("2026-03-01T06:00:00", "a", 9.0)], asset_id="OTHER")
    )
    store.write_readings(
        make_readings(
            [("2026-03-01T06:00:00", "a", 8.0)], source="sim", asset_id="SIM-1"
        )
    )

    assert len(store.read_readings()) == 5
    assert store.read_readings(source="sim")["value"].tolist() == [8.0]
    assert set(store.read_readings(assets=["OTHER"])["value"]) == {9.0}
    assert store.read_readings(channels=["b"])["value"].tolist() == [3.0]

    window = store.read_readings(
        source="iss",
        assets=["ISS"],
        start="2026-03-01T00:00:00Z",
        end="2026-03-02T00:00:00Z",
    )
    assert window["value"].tolist() == [1.0, 2.0]  # start inclusive, end exclusive

    assert store.read_readings(start=pd.Timestamp("2026-03-03", tz="UTC")).empty


def test_reads_are_sorted_by_source_asset_channel_and_time(tmp_path, make_readings):
    store = ParquetStore(tmp_path)
    store.write_readings(
        make_readings(
            [
                ("2026-03-02", "b", 1.0),
                ("2026-03-01", "b", 2.0),
                ("2026-03-01", "a", 3.0),
            ]
        )
    )

    result = store.read_readings()

    assert result["value"].tolist() == [3.0, 2.0, 1.0]


def test_failed_write_leaves_no_files_behind(tmp_path, make_readings, monkeypatch):
    frame = make_readings([("2026-03-01", "temp", 1.0), ("2026-03-02", "temp", 2.0)])
    real_write = pq.write_table
    calls = {"n": 0}

    def crash_on_second(table, where, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_write(table, where, *args, **kwargs)

    monkeypatch.setattr("constellation.core.store.pq.write_table", crash_on_second)
    store = ParquetStore(tmp_path)

    with pytest.raises(OSError, match="disk full"):
        store.write_readings(frame)

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert store.read_readings().empty


def test_failed_rename_cleans_up_temporary_files(tmp_path, make_readings, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("constellation.core.store.os.replace", refuse)

    with pytest.raises(OSError, match="read-only"):
        ParquetStore(tmp_path).write_readings(
            make_readings([("2026-03-01", "temp", 1.0)])
        )

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_stray_temporary_files_are_ignored_by_reads(tmp_path, make_readings):
    store = ParquetStore(tmp_path)
    store.write_readings(make_readings([("2026-03-01", "temp", 1.0)]))
    directory = next(tmp_path.glob("source=*/asset=*/date=*"))
    (directory / ".part-crashed.parquet.tmp").write_bytes(b"not parquet")

    assert len(store.read_readings()) == 1


def test_invalid_frames_and_unsafe_names_are_rejected(tmp_path, make_readings):
    store = ParquetStore(tmp_path)
    frame = make_readings([("2026-03-01", "temp", 1.0)])

    with pytest.raises(SchemaError):
        store.write_readings(frame.drop(columns="status"))
    with pytest.raises(SchemaError, match="partition names"):
        store.write_readings(
            make_readings([("2026-03-01", "temp", 1.0)], asset_id="../etc")
        )
    assert not list(tmp_path.rglob("*.parquet"))


def test_empty_frame_writes_nothing(tmp_path, make_readings):
    empty = make_readings([("2026-03-01", "temp", 1.0)]).iloc[0:0]

    assert ParquetStore(tmp_path).write_readings(empty) == []


def test_missing_store_reads_as_empty(tmp_path):
    store = ParquetStore(tmp_path / "nope")

    assert store.read_readings().empty
    assert list(store.read_readings().columns) == [c.name for c in READINGS]
    assert store.list_assets() == []
    assert store.stats().empty


def test_appends_accumulate_in_separate_files(tmp_path, make_readings):
    store = ParquetStore(tmp_path)

    store.write_readings(make_readings([("2026-03-01T00:00:00", "temp", 1.0)]))
    store.write_readings(make_readings([("2026-03-01T00:00:01", "temp", 2.0)]))

    assert len(_parts(tmp_path)) == 2
    assert store.read_readings()["value"].tolist() == [1.0, 2.0]


def test_list_assets_and_stats(tmp_path, make_readings):
    store = ParquetStore(tmp_path)
    store.write_readings(
        make_readings(
            [
                ("2026-03-01T00:00:00", "a", 1.0),
                ("2026-03-02T00:00:00", "a", 2.0),
                ("2026-03-01T05:00:00", "b", 3.0),
            ]
        )
    )
    store.write_readings(
        make_readings([("2026-03-01", "x", 1.0)], source="sim", asset_id="SIM-1")
    )

    assert store.list_assets() == [("iss", "ISS"), ("sim", "SIM-1")]

    stats = store.stats().set_index(["source", "asset_id", "channel"])
    row = stats.loc[("iss", "ISS", "a")]
    assert row["rows"] == 2
    assert row["first_t"] == pd.Timestamp("2026-03-01", tz="UTC")
    assert row["last_t"] == pd.Timestamp("2026-03-02", tz="UTC")
    assert stats.loc[("sim", "SIM-1", "x"), "rows"] == 1
